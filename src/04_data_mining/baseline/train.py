"""Simple baselines: the blunder rate of a group of moves, learned from the fitting moves only.

Usage:  uv run python -m src.04_data_mining.baseline.train [--rules overall speed speed_elo eval_wp opp_erred]
            [--max-games N]

Each group's rate is pulled towards its parent's rate by SHRINK moves; writes data/04_data_mining/<selection>/baseline/<rule>/.
"""
import argparse
import time

import numpy as np
import pandas as pd

from src.config import CFG
from src.common.buckets import SHRINK, elo_bin, shrunk
from ..common import TARGET, folds, load, save_run

METHOD = "baseline"
RULES = ["overall", "speed", "speed_elo", "eval_wp", "opp_erred"]
COLUMNS = ["speed", "mover_elo", "eval_wp", "opp_last_loss"]
MISTAKE = CFG["features"]["marks"]["mistake"]
EVAL_BANDS = 20


def eval_band(wp):
    """The win probability band of every move, e.g. '45-50%', or 'none'."""
    labels = []
    for k in range(EVAL_BANDS):
        labels.append(f"{k * 100 // EVAL_BANDS:02d}-{(k + 1) * 100 // EVAL_BANDS}%")
    labels = np.array(labels)
    w = wp.to_numpy(dtype=float)
    band = np.clip(np.floor(np.nan_to_num(w) * EVAL_BANDS), 0, EVAL_BANDS - 1).astype(int)
    return pd.Series(np.where(np.isnan(w), "none", labels[band]), index=wp.index)


def groups(df, rule):
    """The group of every move under a rule and the group's parent."""
    everyone = pd.Series("all", index=df.index)
    if rule == "overall":
        return everyone, everyone
    speed = df["speed"].astype(str)
    if rule == "speed":
        return speed, everyone
    if rule == "speed_elo":
        elo = pd.Series(elo_bin(df["mover_elo"]), index=df.index).astype(str)
        return speed + " | Elo bin " + elo, speed
    if rule == "eval_wp":
        return eval_band(df["eval_wp"]), everyone
    if rule == "opp_erred":
        erred = df["opp_last_loss"].to_numpy(dtype=float) >= MISTAKE
        return pd.Series(np.where(erred, "opponent erred", "other"), index=df.index), everyone
    raise ValueError(f"unknown rule {rule!r}")


def fit_rule(fit_df, rule):
    """Per group: moves, blunders and the shrunk rate; and the rate of every parent."""
    group, parent = groups(fit_df, rule)
    y = fit_df[TARGET].astype(float)
    by_parent = y.groupby(parent).mean()
    t = pd.DataFrame({"group": group, "parent": parent, "y": y})
    t = t.groupby(["group", "parent"])["y"].agg(blunders="sum", moves="count").reset_index()
    t["rate"] = shrunk(t["blunders"], t["moves"], t["parent"].map(by_parent))
    return t, by_parent


def predict(df, rule, table, by_parent, overall):
    group, parent = groups(df, rule)
    p = group.map(table.set_index("group")["rate"])
    p = p.fillna(parent.map(by_parent)).fillna(overall)
    return p.to_numpy(dtype=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", nargs="+", choices=RULES, default=RULES)
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    fit_df, valid = folds(load("train", COLUMNS, args.max_games))
    test = load("test", COLUMNS)
    overall = float(fit_df[TARGET].mean())
    for rule in args.rules:
        table, by_parent = fit_rule(fit_df, rule)
        preds = {}
        for part, df in [("valid", valid), ("test", test)]:
            preds[part] = df[["game_id", "ply", TARGET]].assign(p=predict(df, rule, table, by_parent, overall))
        settings = {"rule": rule, "features": "rule", "shrink": SHRINK, "max_train_games": args.max_games}
        d = save_run(METHOD, rule, settings, preds, t0, extra={"fit_rows": len(fit_df), "groups": len(table)})
        table.to_csv(d / "rates.csv", index=False)
        print(f"{d}: {len(table)} groups from {len(fit_df):,} moves")


if __name__ == "__main__":
    main()
