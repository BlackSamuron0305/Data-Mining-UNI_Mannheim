"""Lichess PGN text -> games (with python-chess), and the complete-case filter of the selection.

A Lichess game looks like this (headers, blank line, one line of moves):

    [Event "Rated Rapid game"] ... [WhiteElo "1914"] [BlackElo "1894"] ... [TimeControl "600+0"]

    1. e4 { [%eval 0.18] [%clk 0:10:00] } 1... c5 { [%eval 0.32] [%clk 0:10:00] } 2. Nf3?! { ... } 1-0

`%clk` is the mover's remaining clock after the move, `%eval` the Stockfish evaluation of the position after
the move (`#3` = mate in 3); `?!`, `?` and `??` are the marks that Lichess writes for inaccuracies, mistakes and
blunders. python-chess reads them as node.clock(), node.eval() and the NAGs 6, 2 and 4 of node.nags. The target
of the project is the `??` mark (chess.pgn.NAG_BLUNDER).
"""
from __future__ import annotations

import io
import re
from collections import Counter
from collections.abc import Iterator
from typing import NamedTuple

from compression import zstd

import chess.engine
import chess.pgn
import numpy as np

from src.config import CFG

GAME_START_RE = re.compile(r"^\[Event ", re.M)    # "[Event" at the start of a line opens a game
# Event of a lobby game, an arena and a swiss tournament: "Rated Blitz game", "Rated Blitz tournament <link>",
# "Blitz swiss <link>" (swiss games lack "Rated", but the file holds rated games only) -> blitz
SPEED_RE = re.compile(r"(?:Rated )?(\w+) (?:game|tournament|swiss)\b")
STEPS = ["scanned", "speed bullet-classical", "has engine evaluations", "passes filters", "complete case"]
FILTERS = CFG["filters"]


class Game(NamedTuple):
    """What the selection needs of a complete game."""
    game_id: str
    speed: str
    white_elo: int
    black_elo: int
    time_control: str      # start time + increment in seconds, e.g. '180+2'
    clk: np.ndarray        # the mover's clock after every ply, in seconds
    blunder: np.ndarray    # every ply: Lichess marked the move with ??


def split_games(text: str) -> Iterator[str]:
    """The original text of every game in `text`, which must hold whole games. The text is kept as it is, so that
    the selection can store the games unchanged (python-chess only gives back its own rewrite of a game)."""
    starts = [m.start() for m in GAME_START_RE.finditer(text)] + [len(text)]
    for a, b in zip(starts, starts[1:]):
        yield text[a:b]


def iter_pgn(path) -> Iterator[str]:
    """The original text of every game in a .pgn.zst file, one game at a time (the file is streamed)."""
    buf: list[str] = []
    with zstd.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("[Event ") and buf:
                yield "".join(buf)
                buf = []
            buf.append(line)
    if buf:
        yield "".join(buf)


def time_control(tc: str) -> tuple[int, int]:
    """Start time and increment in seconds of a TimeControl header such as '180+2'."""
    base, _, inc = tc.partition("+")
    return int(base), int(inc or 0)


# Lichess turns an evaluation into winning chances with a logistic curve of the centipawns, capped at +-1000; a
# mate counts as the cap. A move is a blunder when it lowers the mover's winning chances (-1 to 1) by 0.3 or more,
# a mistake by 0.2, an inaccuracy by 0.1: in win probability (0 to 1) that is 0.15, 0.10 and 0.05.
CP_CAP = 1000
WIN_K = 0.00368208


def capped_cp(score: chess.engine.Score) -> int:
    """Centipawns of an evaluation, mates and large values capped at +-CP_CAP."""
    return max(-CP_CAP, min(CP_CAP, score.score(mate_score=100_000)))


def win_prob(cp) -> np.ndarray:
    """Win probability (0 to 1) of the side the centipawns are counted for, as Lichess computes it."""
    return 1.0 / (1.0 + np.exp(-WIN_K * np.clip(np.asarray(cp, dtype=float), -CP_CAP, CP_CAP)))


def read_game(pgn: str) -> chess.pgn.Game | None:
    """The first game of `pgn` as a python-chess game, or None if there is none."""
    return chess.pgn.read_game(io.StringIO(pgn))


def complete_game(pgn: str, funnel: Counter) -> Game | None:
    """The game in `pgn` (the text of one game) if it is a complete case, else None. Every filter step a game
    passes is counted in `funnel` (by speed). Complete case: a rated standard game of speed bullet to classical,
    both ratings known, no computer player, ended normally or on time, at least `min_plies` plies, all moves
    legal, a clock on every ply and an engine evaluation after every ply (the mating move needs none)."""
    headers = chess.pgn.read_headers(io.StringIO(pgn))
    if headers is None:
        return None
    m = SPEED_RE.match(headers.get("Event", ""))
    speed = m.group(1).lower() if m else "other"
    funnel[("scanned", speed if speed in FILTERS["speeds"] else "other")] += 1     # ultrabullet, correspondence
    if speed not in FILTERS["speeds"]:
        return None
    funnel[("speed bullet-classical", speed)] += 1
    if "[%eval " not in pgn:                       # checked on the text: much faster than reading the moves
        return None
    funnel[("has engine evaluations", speed)] += 1
    w_elo, b_elo = headers.get("WhiteElo", "?"), headers.get("BlackElo", "?")
    if not (w_elo.isdigit() and b_elo.isdigit()) or "BOT" in (headers.get("WhiteTitle"), headers.get("BlackTitle")):
        return None
    if headers.get("Termination") not in FILTERS["terminations"]:
        return None
    game = read_game(pgn)
    nodes = list(game.mainline()) if game is not None and not game.errors else []
    if len(nodes) < FILTERS["min_plies"]:
        return None
    funnel[("passes filters", speed)] += 1
    clk = [n.clock() for n in nodes]
    if None in clk or any(n.eval() is None for n in nodes[:-1]):
        return None
    if nodes[-1].eval() is None and not nodes[-1].board().is_checkmate():
        return None
    funnel[("complete case", speed)] += 1
    return Game(game_id=headers.get("Site", "").rsplit("/", 1)[-1], speed=speed, white_elo=int(w_elo),
                black_elo=int(b_elo), time_control=headers.get("TimeControl", "?"),
                clk=np.rint(clk).astype(np.int64),
                blunder=np.array([chess.pgn.NAG_BLUNDER in n.nags for n in nodes], dtype=bool))
