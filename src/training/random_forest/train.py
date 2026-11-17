"""Random forest: many deep trees, each on a random sample of the moves and of the features, averaged.

Usage:  uv run python -m src.training.random_forest.train [--features basic|context|engine] [--n-estimators 200]
            [--min-samples-leaf 200] [--max-features sqrt] [--max-samples 0.2] [--max-games N] [--name NAME]

Bagging against the boosting of gradient_boosting: the trees are grown independently and averaged, which lowers the
variance of a single tree without fitting the errors of the trees before. Each tree sees a bootstrap sample of
--max-samples of the fitting moves (a fraction keeps the run time and memory down with millions of moves). The
leaves keep at least --min-samples-leaf moves, so that a leaf's blunder rate is a probability and not 0 or 1.
Speed enters as its rank; missing values are routed by the trees. Fitted on the training games without the
held-out ones; predicts the held-out games and the test games.

Writes data/runs/random_forest/<name>/ (name: the feature set), see src/training/common.py, plus
importance.csv     impurity decrease per feature, averaged over the trees
"""
from __future__ import annotations

import argparse
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import SEED, WORKERS
from src.training.common import FEATURE_SETS, TARGET, TRAIN, load, save_run, tree_features

METHOD = "random_forest"


def max_features(value: str):
    return value if value in ("sqrt", "log2") else float(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="context")
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--min-samples-leaf", type=int, default=200)
    ap.add_argument("--max-features", default="sqrt", help="sqrt, log2 or a share of the features such as 0.5")
    ap.add_argument("--max-samples", type=float, default=0.2, help="share of the fitting moves per tree")
    ap.add_argument("--max-games", type=int, default=TRAIN["max_train_games"])
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    t0 = time.time()
    train = load("train", args.features, args.max_games)
    test = load("test", args.features)
    fit, valid = train[train["fold"] == "train"], train[train["fold"] == "valid"]
    params = {"n_estimators": args.n_estimators, "min_samples_leaf": args.min_samples_leaf,
              "max_features": max_features(args.max_features), "max_samples": args.max_samples}
    m = RandomForestClassifier(**params, bootstrap=True, n_jobs=WORKERS, random_state=SEED)
    x = tree_features(fit, args.features)
    m.fit(x, fit[TARGET])
    preds = {part: df.assign(p=m.predict_proba(tree_features(df, args.features))[:, 1])
             for part, df in (("valid", valid), ("test", test))}
    d = save_run(METHOD, args.name or args.features,
                 {"features": args.features, **params, "max_train_games": args.max_games}, preds, t0, model=m,
                 extra={"fit_rows": len(fit)})
    pd.Series(m.feature_importances_, index=x.columns, name="importance").sort_values(ascending=False) \
        .to_csv(d / "importance.csv")
    print(f"{d}: {args.n_estimators} trees on {len(fit):,} moves, valid {len(valid):,}, test {len(test):,}")


if __name__ == "__main__":
    main()
