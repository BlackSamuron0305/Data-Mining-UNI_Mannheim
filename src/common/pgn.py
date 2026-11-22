# Read a Lichess game: moves, clocks ([%clk]), evaluations ([%eval]) and blunder marks (??)
from __future__ import annotations

import io
import re
from collections.abc import Iterator

from compression import zstd

import chess.engine
import chess.pgn
import numpy as np

# "Rated Blitz game", "Rated Blitz tournament ..." or "Blitz swiss ..." -> Blitz
SPEED_RE = re.compile(r"(?:Rated )?(\w+) (?:game|tournament|swiss)\b")

# Lichess turns centipawns into a win probability with a logistic curve, capped at +-1000
CP_CAP = 1000
WIN_K = 0.00368208


# Every game in a .pgn.zst file, as text
def iter_pgn(path) -> Iterator[str]:
    lines = []
    with zstd.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("[Event ") and lines:
                yield "".join(lines)
                lines = []
            lines.append(line)
    if lines:
        yield "".join(lines)


# Start time and increment in seconds, from a TimeControl such as '180+2'
def time_control(tc: str) -> tuple[int, int]:
    base, _, inc = tc.partition("+")
    return int(base), int(inc or 0)


def capped_cp(score: chess.engine.Score) -> int:
    return max(-CP_CAP, min(CP_CAP, score.score(mate_score=100_000)))


# Win probability (0 to 1) as Lichess computes it
def win_prob(cp) -> np.ndarray:
    cp = np.clip(np.asarray(cp, dtype=float), -CP_CAP, CP_CAP)
    return 1.0 / (1.0 + np.exp(-WIN_K * cp))


def read_game(pgn: str) -> chess.pgn.Game | None:
    return chess.pgn.read_game(io.StringIO(pgn))
