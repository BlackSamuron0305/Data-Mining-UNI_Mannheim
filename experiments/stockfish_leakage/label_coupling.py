"""Tests A and D: does Lichess's own evaluation (eval_wp), which comes from the same engine run as the ?? label,
tell more about the label than an independent engine look at the same position (our sf_wp_best)?

D: recompute ?? from the file's evaluations (loss >= 0.15 win probability) and compare with the marks.
A: r = eval_wp - sf_wp_best. Within bands of sf_wp_best, does the blunder rate rise with r?

Usage (from the repository): uv run python -m experiments.stockfish_leakage.label_coupling [train|test]
"""
import sys

import numpy as np
import pandas as pd

part = sys.argv[1] if len(sys.argv) > 1 else "train"
cols = ["game_id", "ply", "blunder", "eval_wp", "opp_last_loss", "speed", "mover_elo"]
df = pd.read_parquet(f"data/02_preprocessing/stratified/{part}.parquet", columns=cols)
sf = pd.read_parquet(f"data/02_preprocessing/stockfish/{part}.parquet",
                     columns=["game_id", "ply", "sf_wp_best", "sf_share_blunder"])
df = df.merge(sf, on=["game_id", "ply"], how="left", validate="one_to_one")
df = df.sort_values(["game_id", "ply"], kind="stable").reset_index(drop=True)
print(f"{part}: {len(df):,} moves, {df['game_id'].nunique():,} games, blunder rate {df['blunder'].mean():.4f}, "
      f"sf missing {df['sf_wp_best'].isna().sum():,}")

# the evaluation after the move, for the mover: 1 - the next position's win probability for the other side
same_game = df["game_id"].eq(df["game_id"].shift(-1))
after = np.where(same_game, 1.0 - df["eval_wp"].shift(-1), np.nan)
loss = df["eval_wp"] - after

# ---- D: the label as a function of the file's evaluations ----
known = ~np.isnan(loss) & df["eval_wp"].notna()
rule = loss >= 0.15
print("\nD. ?? against the rule 'loss in win probability >= 0.15' (moves with an evaluation before and after)")
print(pd.crosstab(df.loc[known, "blunder"], rule[known], rownames=["?? mark"], colnames=["rule"]))
agree = (df.loc[known, "blunder"] == rule[known]).mean()
print(f"agreement {agree:.4%}")
for thr in (0.10, 0.14, 0.15, 0.16, 0.20, 0.30):
    r = loss[known] >= thr
    print(f"  threshold {thr:.2f}: agreement {(df.loc[known, 'blunder'] == r).mean():.4%}")
low = df["eval_wp"] < 0.15
print(f"moves with eval_wp < 0.15: {low.sum():,}, of them ??: {df.loc[low, 'blunder'].mean():.5f} "
      f"(cannot lose 0.15 from there)")

# ---- A: Lichess's evaluation against ours ----
ok = df["eval_wp"].notna() & df["sf_wp_best"].notna()
d = df[ok].copy()
d["r"] = d["eval_wp"] - d["sf_wp_best"]
print(f"\nA. r = eval_wp - sf_wp_best on {len(d):,} moves: mean {d['r'].mean():+.4f}, sd {d['r'].std():.4f}, "
      f"corr {d['eval_wp'].corr(d['sf_wp_best']):.4f}")
d["sf_band"] = pd.cut(d["sf_wp_best"], [0, .1, .3, .5, .7, .9, 1.0001], include_lowest=True)
d["r_band"] = pd.cut(d["r"], [-1.01, -.10, -.03, -.01, .01, .03, .10, 1.01])
rate = d.pivot_table(index="sf_band", columns="r_band", values="blunder", aggfunc="mean", observed=True)
n = d.pivot_table(index="sf_band", columns="r_band", values="blunder", aggfunc="size", observed=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
print("blunder rate by band of our engine's win probability (rows) and of r (columns):")
print(rate.round(4))
print("moves:")
print(n)

# how much the label depends on r beyond our engine's view, as a logistic regression on bands
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, average_precision_score
sample = d.sample(min(len(d), 3_000_000), random_state=0)
y = sample["blunder"].to_numpy()


def onehot(frame, names):
    return pd.get_dummies(frame[names].astype(str), drop_first=False).to_numpy(np.float32)


sample["sf_fine"] = pd.qcut(sample["sf_wp_best"], 50, duplicates="drop")
sample["ev_fine"] = pd.qcut(sample["eval_wp"], 50, duplicates="drop")
sample["r_fine"] = pd.qcut(sample["r"], 20, duplicates="drop")
half = np.arange(len(sample)) % 2 == 0
for name, feats in [("ours (sf_wp_best)", ["sf_fine"]), ("Lichess (eval_wp)", ["ev_fine"]),
                    ("ours + r", ["sf_fine", "r_fine"]), ("Lichess + ours", ["ev_fine", "sf_fine"])]:
    X = onehot(sample, feats)
    m = LogisticRegression(max_iter=500).fit(X[half], y[half])
    p = m.predict_proba(X[~half])[:, 1]
    print(f"  {name:20s} log loss {log_loss(y[~half], p):.5f}  PR-AUC {average_precision_score(y[~half], p):.4f}")
