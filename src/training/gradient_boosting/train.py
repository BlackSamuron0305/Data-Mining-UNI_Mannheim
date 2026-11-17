"""Gradient boosting (scikit-learn's HistGradientBoostingClassifier) on the features of a move.

Usage:  uv run python -m src.training.gradient_boosting.train [--features basic|context|engine]
            [--learning-rate 0.1] [--max-leaf-nodes 63] [--min-samples-leaf 200] [--max-games N] [--name NAME]

Trees handle missing values and the speed (as a category) themselves. Fitted on the training games without the
held-out ones; the held-out games stop the boosting when their log loss has not improved for 30 rounds. Predicts the
held-out games and the test games.

Writes data/runs/gradient_boosting/<name>/ (name: the feature set), see src/training/common.py.
"""
from __future__ import annotations

import argparse
import time

from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import SEED
from src.training.common import FEATURE_SETS, TARGET, TRAIN, features, load, save_run

METHOD = "gradient_boosting"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="context")
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--max-leaf-nodes", type=int, default=63)
    ap.add_argument("--min-samples-leaf", type=int, default=200)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--max-games", type=int, default=TRAIN["max_train_games"])
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    t0 = time.time()
    train = load("train", args.features, args.max_games)
    test = load("test", args.features)
    fit, valid = train[train["fold"] == "train"], train[train["fold"] == "valid"]
    params = {"learning_rate": args.learning_rate, "max_leaf_nodes": args.max_leaf_nodes,
              "min_samples_leaf": args.min_samples_leaf, "max_iter": args.max_iter, "l2_regularization": 1.0}
    m = HistGradientBoostingClassifier(**params, early_stopping=True, n_iter_no_change=30, scoring="loss",
                                       categorical_features="from_dtype", random_state=SEED)
    m.fit(features(fit, args.features), fit[TARGET], X_val=features(valid, args.features), y_val=valid[TARGET])
    preds = {part: df.assign(p=m.predict_proba(features(df, args.features))[:, 1])
             for part, df in (("valid", valid), ("test", test))}
    d = save_run(METHOD, args.name or args.features,
                 {"features": args.features, **params, "max_train_games": args.max_games}, preds, t0, model=m,
                 extra={"fit_rows": len(fit), "iterations": int(m.n_iter_)})
    print(f"{d}: fitted on {len(fit):,} moves in {m.n_iter_} rounds, valid {len(valid):,}, test {len(test):,}")


if __name__ == "__main__":
    main()
