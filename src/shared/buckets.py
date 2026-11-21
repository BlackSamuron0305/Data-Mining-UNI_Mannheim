"""Groups of games and buckets of moves, shared by every step.

Group of a game: speed x Elo of the mean of both ratings. Bucket of a move: speed x Elo of the mover x the mover's
time left before the move.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.config import CFG, SEED

SEL = CFG["selection"]
SPEEDS = CFG["filters"]["speeds"]
N_S = len(SPEEDS)
ELO_BIN, ELO_LO, ELO_HI = SEL["elo_bin"], SEL["elo_min"], SEL["elo_max"]
N_E = (ELO_HI - ELO_LO) // ELO_BIN + 1
WIN = SEL["window"]
N_W = np.array([SEL["window_max"][s] // WIN + 1 for s in SPEEDS])
B_OFF = np.concatenate(([0], np.cumsum(N_E * N_W)))
N_B = int(B_OFF[-1])
N_G = N_S * N_E
SHRINK = 50


def elo_label(e: int) -> str:
    lo = ELO_LO + ELO_BIN * e
    if e == 0:
        return f"<{lo + ELO_BIN}"
    if e == N_E - 1:
        return f"{lo}+"
    return f"{lo}-{lo + ELO_BIN}"


def elo_bin(elo) -> np.ndarray:
    return np.clip((np.asarray(elo) - ELO_LO) // ELO_BIN, 0, N_E - 1).astype(np.int64)


def uniform(game_id: str, salt: str = "ie500-strata") -> float:
    """A fixed pseudo-random number in [0, 1) of a game, from the seed and the game id (keep the salt)."""
    h = hashlib.shake_256(f"{salt}:{SEED}:{game_id}".encode("utf-8")).digest(8)
    return int.from_bytes(h, "little") / 2.0 ** 64


def move_buckets(speed: str, white_elo: int, black_elo: int, clk: np.ndarray) -> np.ndarray:
    """Bucket of every ply: speed x Elo of the mover x the mover's clock before the move (`clk`: after every ply)."""
    n = len(clk)
    s = SPEEDS.index(speed)
    elo = elo_bin(np.where(np.arange(n) % 2 == 0, white_elo, black_elo))
    # the first move of each side counts the clock after it
    own = np.concatenate((clk[:2], clk[:-2]))
    return B_OFF[s] + elo * N_W[s] + np.minimum(own // WIN, N_W[s] - 1)


def game_group(speed: str, white_elo: int, black_elo: int) -> int:
    """Group of a game: speed x Elo of the mean of both ratings."""
    return SPEEDS.index(speed) * N_E + int(elo_bin((white_elo + black_elo) / 2))


def shrunk(blunders, moves, parent) -> pd.Series:
    """Blunder probability of a bucket pulled towards `parent` by SHRINK moves at the parent's probability."""
    return (blunders + SHRINK * parent) / (moves + SHRINK)


def time_order(labels: pd.Series) -> pd.Series:
    """Seconds at the start of a time-left label such as '1:30-2:00' or '30:00+', for sorting."""
    first = labels.str.extract(r"^(\d+):(\d+)").astype(int)
    return first[0] * 60 + first[1]
