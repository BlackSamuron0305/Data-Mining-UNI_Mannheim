"""Logistic regression on the features of a move.

Usage:  uv run python -m src.training.logistic_regression.train [--features basic|context|engine] [--C 1.0]
                                                               [--max-games N] [--name NAME]

Numbers: missing values filled with the median plus an indicator column, then standardised. Speed: one column per
speed. Fitted on the training games without the held-out ones; predicts the held-out games and the test games.
No class weights, so the predictions stay probabilities (blunders are about 5% of the moves).

Writes data/runs/logistic_regression/<name>/ (name: the feature set), see src/training/common.py, plus
coefficients.csv (on the standardised features).
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import SEED
from src.training.common import FEATURE_SETS, TARGET, TRAIN, features, load, numeric_prep, save_run

METHOD = "logistic_regression"


def model(feature_set: str, c: float) -> Pipeline:
    return Pipeline([("prep", numeric_prep(feature_set)),
                     ("lr", LogisticRegression(C=c, max_iter=2000, random_state=SEED))])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="context")
    ap.add_argument("--C", type=float, default=1.0, help="inverse strength of the L2 penalty")
    ap.add_argument("--max-games", type=int, default=TRAIN["max_train_games"])
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    t0 = time.time()
    train = load("train", args.features, args.max_games)
    test = load("test", args.features)
    fit, valid = train[train["fold"] == "train"], train[train["fold"] == "valid"]
    m = model(args.features, args.C)
    m.fit(features(fit, args.features), fit[TARGET])
    preds = {part: df.assign(p=m.predict_proba(features(df, args.features))[:, 1])
             for part, df in (("valid", valid), ("test", test))}
    name = args.name or args.features
    d = save_run(METHOD, name, {"features": args.features, "C": args.C, "max_train_games": args.max_games},
                 preds, t0, model=m, extra={"fit_rows": len(fit), "iterations": int(m["lr"].n_iter_[0])})
    coef = pd.Series(m["lr"].coef_[0], index=m["prep"].get_feature_names_out(), name="coefficient")
    coef.sort_values(key=abs, ascending=False).to_csv(d / "coefficients.csv")
    print(f"{d}: fitted on {len(fit):,} moves, valid {len(valid):,}, test {len(test):,}")


if __name__ == "__main__":
    main()
