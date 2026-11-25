"""Baseline: predict for every move the blunder probability of its bucket (speed x Elo of the mover x time left).

Usage:  uv run python -m src.04_data_mining.bucket_rate.train [--source month|train] [--name NAME]

--source month takes the rates of the whole month, --source train learns them from the fitting moves only.
"""
import argparse
import time

import numpy as np
import pandas as pd

from src.config import MONTH_DIR
from src.common.buckets import shrunk
from src.common.server import ensure
from ..common import load, save_run

METHOD = "bucket_rate"


def read_month():
    path = ensure(MONTH_DIR / "rates.csv", "the blunder rates (python -m src.01_selection.month.rates)")
    return pd.read_csv(path, index_col="bucket")


def rate(counts):
    return counts["sum"] / counts["count"].replace(0, np.nan)


def rates_from(df):
    """Blunder probability per bucket of the moves in df, shrunk towards its speed x Elo (else speed) rate."""
    month = read_month()[["speed", "mover_elo"]]
    c = df.groupby("bucket")["blunder"].agg(["sum", "count"]).reindex(month.index, fill_value=0)
    c = c.join(month)
    by_elo = c.groupby(["speed", "mover_elo"])[["sum", "count"]].transform("sum")
    by_speed = c.groupby("speed")[["sum", "count"]].transform("sum")
    parent = rate(by_elo).fillna(rate(by_speed)).fillna(df["blunder"].mean())
    return shrunk(c["sum"], c["count"], parent)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["month", "train"], default="month")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    t0 = time.time()
    train = load("train", "basic")
    test = load("test", "basic")
    if args.source == "month":
        rates = read_month()["rate_baseline"]
    else:
        rates = rates_from(train[train["fold"] == "train"])
    preds = {}
    for part, df in [("valid", train[train["fold"] == "valid"]), ("test", test)]:
        preds[part] = df.assign(p=df["bucket"].map(rates).to_numpy())
    d = save_run(METHOD, args.name or args.source, {"source": args.source, "features": "bucket"}, preds, t0)
    print(f"{d}: valid {len(preds['valid']):,} moves, test {len(preds['test']):,} moves")


if __name__ == "__main__":
    main()
