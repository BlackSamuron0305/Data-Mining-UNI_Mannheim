"""Test B: how much does the model rely on Lichess's own evaluations, which come from the same engine run as the
?? label? XGBoost with the same fixed settings, on the engine feature set, three times:

  lichess      the tables as they are: eval_wp and the history (losses, error counts) from Lichess's evals and marks
  eval_swap    eval_wp replaced by our Stockfish's sf_wp_best; history as it is
  independent  also the history recomputed from our Stockfish: the loss of an earlier move is
               sf_wp_best before it minus (1 - sf_wp_best of the next position), errors counted by the same thresholds

Nothing of the label's engine run is left in "independent": what a tool that runs its own engine during the game
would have. Train on a sample of training games, early stopping on held-out training games, score on all test games.

Usage (from the repository): uv run python -m experiments.stockfish_leakage.independent_engine [--train-games 60000]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

extract = importlib.import_module("src.02_preprocessing.moves.extract")
OUT = Path("data/experiments/stockfish_leakage")
MARKS = {"inaccuracy": 0.05, "mistake": 0.10, "blunder": 0.15}
LOSS_COLS = ["own_last_loss", "own_loss_mean", "own_loss_recent", "own_loss_max", "opp_last_loss", "opp_loss_recent"]
MARK_COLS = ["own_inaccuracies", "own_mistakes", "own_blunders", "own_moves_since_error", "opp_mistakes",
             "opp_blunders"]
DROP = ["game_id", "ply", "bucket", "group", "blunder"]


def u(gid: str) -> float:
    return int(hashlib.sha1(f"b:{gid}".encode()).hexdigest()[:8], 16) / 2**32


def load(part: str, games: int | None) -> pd.DataFrame:
    df = pd.read_parquet(f"data/02_preprocessing/stratified/{part}.parquet")
    if games:
        ids = df["game_id"].drop_duplicates()
        keep = set(ids[ids.map(u).rank(method="first") <= games])
        df = df[df["game_id"].isin(keep)]
    sf = pd.read_parquet(f"data/02_preprocessing/stockfish/{part}.parquet", columns=["game_id", "ply", "sf_wp_best"])
    df = df.merge(sf, on=["game_id", "ply"], how="left", validate="one_to_one")
    return df.sort_values(["game_id", "ply"], kind="stable").reset_index(drop=True)


def our_history(df: pd.DataFrame) -> pd.DataFrame:
    """The loss and error columns of the history, from our engine instead of Lichess's evals and marks."""
    wp = df["sf_wp_best"].to_numpy(float)
    same = df["game_id"].to_numpy()[1:] == df["game_id"].to_numpy()[:-1]
    after = np.full(len(df), np.nan)
    after[:-1] = np.where(same, 1.0 - wp[1:], np.nan)
    loss = np.clip(wp - after, 0.0, None)
    mark = np.zeros(len(df), np.int8)
    for code, name in ((1, "inaccuracy"), (2, "mistake"), (3, "blunder")):
        mark[loss >= MARKS[name]] = code
    out = {c: np.empty(len(df), np.float32) for c in LOSS_COLS + MARK_COLS}
    starts = np.flatnonzero(np.r_[True, ~same])
    ends = np.r_[starts[1:], len(df)]
    for s, e in zip(starts, ends):
        h = extract.history(np.zeros(e - s), loss[s:e], mark[s:e])
        for c in out:
            out[c][s:e] = h[c]
    res = df.copy()
    for c in out:
        res[c] = out[c]
    return res


def matrix(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    X = df.drop(columns=DROP)
    if variant == "lichess":
        X = X.drop(columns=["sf_wp_best"])
    else:
        X = X.drop(columns=["eval_wp"])
    for c in ("speed", "eco"):
        X[c] = X[c].astype("category")
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-games", type=int, default=60_000)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    train = load("train", args.train_games)
    test = load("test", None)
    print(f"train {len(train):,} moves of {train['game_id'].nunique():,} games; test {len(test):,} moves", flush=True)
    data = {"lichess": (train, test), "eval_swap": (train, test)}
    data["independent"] = (our_history(train), our_history(test))
    valid = train["game_id"].map(u).to_numpy() < 0.1
    for c in ("speed", "eco"):
        cats = sorted(set(train[c].dropna()) | set(test[c].dropna()))
        for frame in (train, test, *data["independent"]):
            frame[c] = pd.Categorical(frame[c], categories=cats)
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist", "max_depth": 8,
              "eta": 0.08, "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0,
              "nthread": args.threads, "seed": 0}
    results = {}
    y_test = test["blunder"].to_numpy()
    for variant, (tr, te) in data.items():
        X, Xt = matrix(tr, variant), matrix(te, variant)
        y = tr["blunder"].to_numpy()
        dfit = xgb.DMatrix(X[~valid], y[~valid], enable_categorical=True)
        dval = xgb.DMatrix(X[valid], y[valid], enable_categorical=True)
        model = xgb.train(params, dfit, 2000, evals=[(dval, "valid")], early_stopping_rounds=50, verbose_eval=False)
        p = model.predict(xgb.DMatrix(Xt, enable_categorical=True), iteration_range=(0, model.best_iteration + 1))
        gain = model.get_score(importance_type="total_gain")
        total = sum(gain.values())
        top = sorted(gain.items(), key=lambda kv: -kv[1])[:8]
        results[variant] = {"rounds": model.best_iteration + 1, "pr_auc": average_precision_score(y_test, p),
                            "log_loss": log_loss(y_test, p), "roc_auc": roc_auc_score(y_test, p),
                            "top_gain": {k: round(v / total, 3) for k, v in top}}
        print(variant, json.dumps(results[variant]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "independent_engine.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
