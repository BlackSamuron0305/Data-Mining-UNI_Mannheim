"""Bucket rates step 1 - the blunder probability of every bucket of moves, from every complete game of the month.

Usage:  uv run python -m src.preprocessing.bucket_rates.rates        (after the census)

A bucket is speed x Elo of the mover x the mover's time left before the move (756 buckets, see
src/preprocessing/selection/census.py). The census has counted the moves and blunders of every complete game of the
month per bucket; this step turns them into probabilities with a 95% interval. They serve two purposes:

- the baseline that every model has to beat: predict for a move the blunder probability of its bucket
  (src/training/bucket_rate/);
- the reference for the check of the selection (selection_check.py).

`rate_baseline` is the probability the baseline predicts: the bucket's blunders plus SHRINK moves at the probability of
its speed x Elo of the mover (all time steps), divided by its moves plus SHRINK. A bucket with millions of moves keeps its
own rate; one with a handful moves towards its speed x Elo instead of a rate of 0 or 1 that a log loss punishes
without end. A bucket without a move gets the speed x Elo rate, or the speed's if that is empty too (column
`fallback`).

Outputs (data/bucket_rates/)
rates.csv           per bucket: speed, Elo of the mover, time left, moves, blunders, games, rate, 95% interval,
                    rate_baseline, few_games
by_time_left.csv    per speed and time step (all Elo bands together): moves, blunders, rate, 95% interval
summary.md          the main numbers
logs/rates.log
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.config import BUCKET_RATES_DIR, CENSUS_DIR, LOGS_SUB, MONTH, setup_log
from src.preprocessing.selection.census import SPEEDS

log = logging.getLogger("bucket_rates")
Z95 = 1.959964
SHRINK = 50                      # moves at the parent's probability added to every bucket of the baseline


def wilson(k, n, z: float = Z95) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson interval of a probability with k hits in n trials (NaN for n = 0)."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        mid = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        # clamped, so that rounding never puts the rate outside its interval (0 blunders: lower end exactly 0)
        return np.minimum(np.maximum(mid - half, 0.0), p), np.maximum(np.minimum(mid + half, 1.0), p)


def shrunk(blunders, moves, parent) -> pd.Series:
    """Blunder probability of a bucket pulled towards `parent` by SHRINK moves at the parent's probability."""
    return (blunders + SHRINK * parent) / (moves + SHRINK)


def add_rates(df: pd.DataFrame, moves: str = "moves", blunders: str = "blunders", suffix: str = "") -> pd.DataFrame:
    m, k = df[moves].to_numpy(dtype=float), df[blunders].to_numpy(dtype=float)
    df[f"rate{suffix}"] = np.where(m > 0, k / np.where(m > 0, m, 1), np.nan)
    df[f"rate_lo{suffix}"], df[f"rate_hi{suffix}"] = wilson(k, m)
    return df


def time_order(labels: pd.Series) -> pd.Series:
    """Seconds at the start of a time-left label such as '1:30-2:00' or '30:00+', for sorting."""
    first = labels.str.extract(r"^(\d+):(\d+)").astype(int)
    return first[0] * 60 + first[1]


def bucket_rates() -> pd.DataFrame:
    """Blunder probability per bucket of the month, with the baseline probability and its fallback."""
    bt = pd.read_csv(CENSUS_DIR / "buckets.csv", index_col="bucket")
    out = add_rates(bt[["speed", "mover_elo", "time_left", "moves", "blunders", "games", "few_games"]].copy())
    by_elo = bt.groupby(["speed", "mover_elo"])[["moves", "blunders"]].transform("sum")
    by_speed = bt.groupby("speed")[["moves", "blunders"]].transform("sum")
    rate_elo = by_elo["blunders"] / by_elo["moves"].replace(0, np.nan)
    rate_speed = by_speed["blunders"] / by_speed["moves"].replace(0, np.nan)
    out["fallback"] = np.where(out["moves"] > 0, "", np.where(by_elo["moves"] > 0, "speed x elo", "speed"))
    out["rate_baseline"] = shrunk(out["blunders"], out["moves"], rate_elo.fillna(rate_speed))
    return out


def by_time_left(df: pd.DataFrame, cols: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Moves and blunders summed over the Elo bands, per speed and time step; `cols` maps a suffix to the (moves,
    blunders) columns to sum."""
    sums = [c for pair in cols.values() for c in pair]
    t = df.groupby(["speed", "time_left"], sort=False)[sums].sum().reset_index()
    for suffix, (m, k) in cols.items():
        add_rates(t, m, k, suffix)
    t["_o"], t["_s"] = t["speed"].map(SPEEDS.index), time_order(t["time_left"])
    return t.sort_values(["_o", "_s"]).drop(columns=["_o", "_s"]).reset_index(drop=True)


def run() -> None:
    meta_path = CENSUS_DIR / "census.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} is missing: run uv run python -m src.preprocessing.selection.census first")
    census = json.loads(meta_path.read_text(encoding="utf-8"))
    (BUCKET_RATES_DIR / LOGS_SUB).mkdir(parents=True, exist_ok=True)
    setup_log(BUCKET_RATES_DIR / LOGS_SUB / "rates.log")

    rates = bucket_rates()
    rates.to_csv(BUCKET_RATES_DIR / "rates.csv")
    tl = by_time_left(rates, {"": ("moves", "blunders")})
    tl.to_csv(BUCKET_RATES_DIR / "by_time_left.csv", index=False)

    used = rates["moves"] > 0
    lines = [f"# Lichess {MONTH}: blunder probability per bucket of moves", "",
             f"From every complete game of the month: {census['n_complete_games']:,} games, {census['n_moves']:,} "
             f"moves, blunder probability {100 * census['blunder_rate']:.3f}%.", "",
             f"- buckets (speed x Elo of the mover x time left): {len(rates)}, with a move: {int(used.sum())}, "
             f"with fewer than the minimum of games: {int(rates['few_games'].sum())}",
             f"- blunder probability per bucket: {100 * rates['rate'].min():.2f}% to {100 * rates['rate'].max():.2f}%",
             f"- widest 95% interval: {100 * (rates['rate_hi'] - rates['rate_lo']).max():.2f} points", "",
             "## By speed and time left (all Elo bands), blunder probability in %", "",
             "| speed | time left | moves | rate | 95% interval |", "|:--|:--|--:|--:|--:|"]
    for _, r in tl.iterrows():
        lines.append(f"| {r['speed']} | {r['time_left']} | {int(r['moves']):,} | {100 * r['rate']:.2f} | "
                     f"{100 * r['rate_lo']:.2f}-{100 * r['rate_hi']:.2f} |")
    (BUCKET_RATES_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("%d buckets, %d with a move, rates %.2f%% to %.2f%%", len(rates), int(used.sum()),
             100 * rates["rate"].min(), 100 * rates["rate"].max())


if __name__ == "__main__":
    run()
