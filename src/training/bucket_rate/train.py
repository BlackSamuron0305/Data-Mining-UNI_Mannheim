"""Baseline: predict for every move the blunder probability of its bucket (speed x Elo of the mover x time left).

Usage:  uv run python -m src.training.bucket_rate.train [--source month|train] [--name NAME]

--source month   the probabilities of every complete game of the month (data/bucket_rates/rates.csv, from
                 src/preprocessing/bucket_rates/rates.py). This is the baseline every model has to beat. The month
                 holds the test games as well, but they are about 0.4% of its moves, so they hardly move a rate.
--source train   the probabilities of the training games without the held-out ones: the same idea, learned only
                 from the data the models see. Every bucket is pulled towards its speed x Elo of the mover as in
                 rates.py (SHRINK moves at that probability), a bucket without a training move gets that one.

Writes data/runs/bucket_rate/<name>/ (name: the source), see src/training/common.py.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.config import BUCKET_RATES_DIR
from src.preprocessing.bucket_rates.rates import shrunk
from src.training.common import load, save_run

METHOD = "bucket_rate"


def rates_from_month() -> pd.Series:
    path = BUCKET_RATES_DIR / "rates.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run uv run python -m src.preprocessing.bucket_rates.rates")
    return pd.read_csv(path, index_col="bucket")["rate_baseline"]


def rates_from(df: pd.DataFrame) -> pd.Series:
    """Blunder probability per bucket of the moves in df, with the fallbacks of the month's table."""
    month = pd.read_csv(BUCKET_RATES_DIR / "rates.csv", index_col="bucket")[["speed", "mover_elo"]]
    c = df.groupby("bucket")["blunder"].agg(["sum", "count"]).reindex(month.index, fill_value=0)
    c = c.join(month)
    elo = c.groupby(["speed", "mover_elo"])[["sum", "count"]].transform("sum")
    speed = c.groupby("speed")[["sum", "count"]].transform("sum")
    rate = lambda s: s["sum"] / s["count"].replace(0, np.nan)
    parent = rate(elo).fillna(rate(speed)).fillna(df["blunder"].mean())
    return shrunk(c["sum"], c["count"], parent)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["month", "train"], default="month")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    t0 = time.time()
    train, test = load("train", "basic"), load("test", "basic")
    rates = rates_from_month() if args.source == "month" else rates_from(train[train["fold"] == "train"])
    preds = {}
    for part, df in (("valid", train[train["fold"] == "valid"]), ("test", test)):
        preds[part] = df.assign(p=df["bucket"].map(rates).to_numpy())
    d = save_run(METHOD, args.name or args.source, {"source": args.source, "features": "bucket"}, preds, t0)
    print(f"{d}: valid {len(preds['valid']):,} moves, test {len(preds['test']):,} moves")


if __name__ == "__main__":
    main()
