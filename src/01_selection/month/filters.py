"""Which games of the month are complete cases (the filters are in src/settings.py), and what is read from them."""
import io
import re
from collections import Counter
from collections.abc import Iterator
from typing import NamedTuple

import chess.pgn
import numpy as np

from src.config import CFG
from src.common.pgn import SPEED_RE, read_game

FILTERS = CFG["filters"]
STEPS = ["scanned", "speed bullet-classical", "has engine evaluations", "passes filters", "complete case"]
GAME_START_RE = re.compile(r"^\[Event ", re.M)


class Game(NamedTuple):
    game_id: str
    speed: str
    white_elo: int
    black_elo: int
    clk: np.ndarray
    blunder: np.ndarray


# True if the cleanup described by its cleanup.json used the filters of src/settings.py
def same_filters(meta: dict) -> bool:
    return meta.get("filters") == FILTERS


# Split the text into single games
def split_games(text: str) -> Iterator[str]:
    starts = [m.start() for m in GAME_START_RE.finditer(text)]
    starts.append(len(text))
    for i in range(len(starts) - 1):
        yield text[starts[i]:starts[i + 1]]


# Return the game if it passes every filter, else None
def complete_game(pgn: str, funnel: Counter) -> Game | None:
    headers = chess.pgn.read_headers(io.StringIO(pgn))
    if headers is None:
        return None
    match = SPEED_RE.match(headers.get("Event", ""))
    speed = match.group(1).lower() if match else "other"
    if speed not in FILTERS["speeds"]:
        funnel[("scanned", "other")] += 1
        return None
    funnel[("scanned", speed)] += 1
    funnel[("speed bullet-classical", speed)] += 1

    if "[%eval " not in pgn:
        return None
    funnel[("has engine evaluations", speed)] += 1

    white_elo = headers.get("WhiteElo", "?")
    black_elo = headers.get("BlackElo", "?")
    if not white_elo.isdigit() or not black_elo.isdigit():
        return None
    if "BOT" in (headers.get("WhiteTitle"), headers.get("BlackTitle")):
        return None
    if headers.get("Termination") not in FILTERS["terminations"]:
        return None
    game = read_game(pgn)
    if game is None or game.errors:
        return None
    nodes = list(game.mainline())
    # every length counts, also an early resignation; a game without a move has nothing to predict
    if not nodes:
        return None
    funnel[("passes filters", speed)] += 1

    clocks = [node.clock() for node in nodes]
    if None in clocks:
        return None
    for node in nodes[:-1]:
        if node.eval() is None:
            return None
    last = nodes[-1]
    if last.eval() is None and not last.board().is_checkmate():
        return None
    funnel[("complete case", speed)] += 1

    blunders = [chess.pgn.NAG_BLUNDER in node.nags for node in nodes]
    return Game(
        game_id=headers.get("Site", "").rsplit("/", 1)[-1],
        speed=speed,
        white_elo=int(white_elo),
        black_elo=int(black_elo),
        clk=np.rint(clocks).astype(np.int64),
        blunder=np.array(blunders, dtype=bool),
    )
