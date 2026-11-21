"""Permutation importance by feature group (and optionally by feature) of a fitted run, on the held-out games.

Usage:  uv run python -m src.04_data_mining.importance [--run METHOD/NAME] [--moves 200000] [--repeats 3]
            [--by group feature]

Shuffles the columns of each group together and measures how much the log loss rises; writes
importance_groups.csv (and importance_features.csv) into the run's folder.
"""
import argparse

import joblib
import numpy as np
import pandas as pd

from src.config import RUNS_DIR, SEED, VALIDATION_DIR
from src.shared.metrics import scores
from .common import TARGET, TUNING, Fitted, folds, groups_of, load


def runs_with_model():
    runs = []
    for p in RUNS_DIR.glob("*/*/model.joblib"):
        if p.parent.parent.name != "ablation":
            runs.append(f"{p.parent.parent.name}/{p.parent.name}")
    return sorted(runs)


def best_run():
    """The run with a model and the lowest log loss on the held-out games."""
    candidates = runs_with_model()
    if not candidates:
        raise FileNotFoundError(f"no run with a model in {RUNS_DIR}: train one first (README.md)")
    path = VALIDATION_DIR / "results.csv"
    if path.exists():
        r = pd.read_csv(path)
        r = r[(r["part"] == "valid") & (r["moves_on"] == "common") & r["run"].isin(candidates)]
        if len(r):
            return str(r.sort_values("log_loss")["run"].iloc[0])
    loss = {}
    for run in candidates:
        p = pd.read_parquet(RUNS_DIR / run / "predictions_valid.parquet")
        loss[run] = scores(p[TARGET], p["p"])["log_loss"]
    return min(loss, key=loss.get)


def permutation(m, df, units, repeats):
    """Log loss rise and pr_auc fall when the columns of each unit are shuffled together."""
    y = df[TARGET].to_numpy()
    base = scores(y, m.predict(df))
    rng = np.random.default_rng(SEED)
    rows = []
    for unit, cols in units.items():
        rise = []
        fall = []
        for _ in range(repeats):
            shuffled = df.copy()
            shuffled[cols] = df[cols].iloc[rng.permutation(len(df))].set_axis(df.index)
            s = scores(y, m.predict(shuffled))
            rise.append(s["log_loss"] - base["log_loss"])
            fall.append(base["pr_auc"] - s["pr_auc"])
        row = {"unit": unit, "columns": " ".join(cols), "log_loss_rise": np.mean(rise), "log_loss_rise_sd": np.nan,
               "pr_auc_fall": np.mean(fall), "pr_auc_fall_sd": np.nan}
        if repeats > 1:
            row["log_loss_rise_sd"] = np.std(rise, ddof=1)
            row["pr_auc_fall_sd"] = np.std(fall, ddof=1)
        rows.append(row)
        print(f"{unit:24s} log loss rise {row['log_loss_rise']:+.5f}, pr_auc fall {row['pr_auc_fall']:+.4f}")

    out = pd.DataFrame(rows).sort_values("log_loss_rise", ascending=False)
    positive = out["log_loss_rise"].clip(lower=0)
    total = positive.sum()
    if total > 0:
        out["share_of_rise"] = positive / total
    else:
        out["share_of_rise"] = np.nan
    return out, base


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=None, help="METHOD/NAME of a run with model.joblib (default: the best)")
    ap.add_argument("--moves", type=int, default=TUNING.get("importance_moves", 200_000))
    ap.add_argument("--repeats", type=int, default=TUNING.get("importance_repeats", 3))
    ap.add_argument("--by", nargs="+", choices=["group", "feature"], default=["group"])
    args = ap.parse_args()

    run = args.run or best_run()
    d = RUNS_DIR / run
    m = joblib.load(d / "model.joblib")
    if not isinstance(m, Fitted):
        raise TypeError(f"{d}/model.joblib is not a Fitted model (an older run): train it again")
    valid = folds(load("train", m.columns))[1]
    if len(valid) > args.moves:
        valid = valid.sample(args.moves, random_state=SEED).reset_index(drop=True)
    print(f"{run}: permutation importance on {len(valid):,} held-out moves, {args.repeats} shuffles")

    for by in args.by:
        if by == "group":
            units = groups_of(m.columns)
        else:
            units = {c: [c] for c in m.columns}
        out, base = permutation(m, valid, units, args.repeats)
        path = d / f"importance_{by}s.csv"
        out.rename(columns={"unit": by}).to_csv(path, index=False)
        print(f"{path}: base log loss {base['log_loss']:.5f}, pr_auc {base['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
