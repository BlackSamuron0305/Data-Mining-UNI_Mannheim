"""Analysis of hash_order.py: does analysing a game backwards with a shared hash (as Lichess does) put information
about the played move into the evaluations a model reads before the move?

F = forward, hash cleared before every position (clean). R = backwards in chunks of 6 with a shared hash (as fishnet).
Per position: wp_fwd, wp_rev (mover's win probability before the move), eval_wp (Lichess), blunder (Lichess ??).

Usage (from the repository): uv run python -m experiments.stockfish_leakage.hash_order_analysis
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

HERE = Path("data/experiments/stockfish_leakage")
df = pd.read_parquet(HERE / "hash_order_train_300g_500000n.parquet").sort_values(["game_id", "ply"])
rng = np.random.default_rng(0)
print(f"{len(df):,} positions, {df['game_id'].nunique()} games, {int(df['blunder'].sum())} blunders")


def boot(stat, frame, n=2000):
    """95% interval of stat(frame), resampling whole games."""
    groups = [g for _, g in frame.groupby("game_id")]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(stat(pd.concat([groups[i] for i in pick])))
    return np.percentile(vals, [2.5, 97.5])


for name in ("fwd", "rev"):
    w = df[f"wp_{name}"]
    after = np.where(df["game_id"].eq(df["game_id"].shift(-1)), 1 - w.shift(-1), np.nan)
    df[f"loss_{name}"] = w - after
    df[f"opp_last_loss_{name}"] = np.where(df["game_id"].eq(df["game_id"].shift(1)), df[f"loss_{name}"].shift(1), np.nan)

# 1. How close is each variant to Lichess's evaluation?
print("\n1. agreement with Lichess's eval_wp (position before the move)")
ok = df["eval_wp"].notna()
for name in ("fwd", "rev"):
    e = (df.loc[ok, f"wp_{name}"] - df.loc[ok, "eval_wp"]).abs()
    print(f"   {name}: mean |diff| {e.mean():.4f}, median {e.median():.4f}, corr {df.loc[ok, f'wp_{name}'].corr(df.loc[ok, 'eval_wp']):.4f}")

# 2. Which variant reproduces the ?? marks (rule: loss >= 0.15)?
print("\n2. ?? marks reproduced by the rule loss >= 0.15")
for name in ("fwd", "rev"):
    k = df[f"loss_{name}"].notna()
    rule = df.loc[k, f"loss_{name}"] >= 0.15
    y = df.loc[k, "blunder"]
    print(f"   {name}: agreement {(rule == y).mean():.4%}, recall of ?? {rule[y].mean():.3f}, precision {y[rule].mean():.3f}")

# 3. The difference R - F before the move: does it depend on whether the move is a blunder?
df["d"] = df["wp_rev"] - df["wp_fwd"]
print("\n3. d = wp_rev - wp_fwd, the change the backward hash makes to the evaluation BEFORE the move")
for lab, g in df.groupby("blunder"):
    print(f"   blunder={lab}: n {len(g):,}, mean d {g['d'].mean():+.5f}, mean |d| {g['d'].abs().mean():.5f}, "
          f"share |d|>0.02 {(g['d'].abs() > 0.02).mean():.3f}")
diff = lambda f: f.loc[f["blunder"], "d"].mean() - f.loc[~f["blunder"], "d"].mean()
absdiff = lambda f: f.loc[f["blunder"], "d"].abs().mean() - f.loc[~f["blunder"], "d"].abs().mean()
print(f"   mean d, blunders minus others: {diff(df):+.5f}, 95% interval {boot(diff, df).round(5)}")
print(f"   mean |d|, blunders minus others: {absdiff(df):+.5f}, 95% interval {boot(absdiff, df).round(5)}")

# 4. Does the backward evaluation predict the label better than the forward one, with the same features?
print("\n4. logistic regression on what a model sees before the move (eval before, opponent's last loss), "
      "5-fold by game")
games = df["game_id"].unique()
fold = dict(zip(games, rng.permutation(len(games)) % 5))
df["fold"] = df["game_id"].map(fold)


def feats(frame, name):
    x = np.column_stack([frame[f"wp_{name}"], frame[f"wp_{name}"] ** 2,
                         frame[f"opp_last_loss_{name}"].fillna(0), frame[f"opp_last_loss_{name}"].isna()])
    return x


use = df[df["wp_fwd"].notna() & df["wp_rev"].notna()]
res = {}
for name in ("fwd", "rev"):
    p = np.zeros(len(use))
    for k in range(5):
        tr, te = use["fold"] != k, use["fold"] == k
        m = LogisticRegression(max_iter=1000).fit(feats(use[tr], name), use.loc[tr, "blunder"])
        p[te.to_numpy()] = m.predict_proba(feats(use[te], name))[:, 1]
    res[name] = p
    print(f"   {name}: log loss {log_loss(use['blunder'], p):.5f}")
ll = lambda y, p: -(y * np.log(p) + (1 - y) * np.log(1 - p))
use = use.assign(gain=ll(use["blunder"].to_numpy(), res["fwd"]) - ll(use["blunder"].to_numpy(), res["rev"]))
mean_gain = lambda f: f["gain"].mean()
print(f"   log loss fwd - rev: {mean_gain(use):+.5f}, 95% interval {boot(mean_gain, use).round(5)} "
      "(positive: the backward evaluations predict the label better)")
