"""Subgroups of the moves (speed, Elo band, move number, time left, win probability) for validate.py and errors.py.

The columns come from data/02_preprocessing/<selection>/without_stockfish/final/<train|test>.parquet; the bands are set in config.yaml (evaluation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import CFG, FINAL_DIRS, KEYS
from src.common.buckets import N_E, SPEEDS, elo_bin, elo_label

EV = CFG["evaluation"]
SOURCE = ["speed", "mover_elo", "move_no", "clock_rel", "eval_wp"]
DIMENSIONS = {"speed": "speed", "elo_band": "Elo of the mover", "phase": "move number",
              "time_left": "time left (share of the game's time)", "position": "mover's win probability"}


def positions(part: str, columns: list[str], game_ids=None) -> pd.DataFrame | None:
    """The columns of the moves of a part ('valid' reads the training moves), only of the given games."""
    if part == "test":
        path = FINAL_DIRS["without_stockfish"] / "test.parquet"
    else:
        path = FINAL_DIRS["without_stockfish"] / "train.parquet"
    if not path.exists():
        return None
    have = set(pq.read_schema(path).names)
    cols = list(KEYS)
    for c in columns:
        if c in have and c not in cols:
            cols.append(c)
    filters = None
    if game_ids is not None:
        filters = [("game_id", "in", list(pd.unique(np.asarray(game_ids))))]
    return pq.read_table(path, columns=cols, filters=filters).to_pandas()


def aligned(keys: pd.DataFrame, pos: pd.DataFrame) -> pd.DataFrame:
    """The columns of `pos` in the order of the moves `keys` (NaN for a move without a row)."""
    return keys[KEYS].merge(pos, on=KEYS, how="left").drop(columns=KEYS).set_index(keys.index)


def cut_bands(x: np.ndarray, edges: list[float], labels: list[str], right: bool) -> pd.Categorical:
    side = "right" if right else "left"
    codes = np.searchsorted(edges, x, side=side)
    codes = np.where(np.isnan(x), -1, codes)
    return pd.Categorical.from_codes(codes, labels, ordered=True)


def pct_labels(edges: list[float], first: str, last: str) -> list[str]:
    pc = [f"{100 * v:g}" for v in edges]
    labels = [first + pc[0] + "%"]
    for a, b in zip(pc, pc[1:]):
        labels.append(f"{a}-{b}%")
    labels.append(pc[-1] + last)
    return labels


def bands(pos: pd.DataFrame) -> pd.DataFrame:
    """One column per dimension (ordered categories), in the order of `pos`."""
    out = pd.DataFrame(index=pos.index)
    out["speed"] = pd.Categorical(pos["speed"], categories=SPEEDS, ordered=True)

    elo = pos["mover_elo"].to_numpy(dtype=float)
    codes = np.where(np.isnan(elo), -1, elo_bin(np.nan_to_num(elo)))
    elo_labels = [elo_label(e) for e in range(N_E)]
    out["elo_band"] = pd.Categorical.from_codes(codes, elo_labels, ordered=True)

    m = EV["move_bands"]
    move_labels = []
    for a, b in zip([0] + m, m):
        move_labels.append(f"{a + 1}-{b}")
    move_labels.append(f"{m[-1] + 1}+")
    out["phase"] = cut_bands(pos["move_no"].to_numpy(dtype=float), m, move_labels, right=False)

    c = EV["clock_bands"]
    out["time_left"] = cut_bands(pos["clock_rel"].to_numpy(dtype=float), c, pct_labels(c, "<", "%+"), right=True)

    e = EV["eval_bands"]
    wp_labels = pct_labels([0] + e + [1], "", "")[1:-1]
    wp = cut_bands(pos["eval_wp"].to_numpy(dtype=float), e, wp_labels, right=True)
    out["position"] = wp.add_categories("none").fillna("none")
    return out
