"""Random forest: many trees, each on a random sample of the moves and of the features, averaged.

Usage:  uv run python -m src.04_data_mining.random_forest.train [--features basic|context|engine] [--n-estimators 200]
            [--min-samples-leaf 200] [--max-features sqrt] [--max-samples 0.2] [--max-games N] [--name NAME]
"""
import argparse

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import SEED, WORKERS
from ..common import FEATURE_SETS, TRAIN, Fitted, fit_and_save, target, tree_features

METHOD = "random_forest"
DEFAULTS = {"n_estimators": 200, "min_samples_leaf": 200, "max_features": "sqrt", "max_samples": 0.2}
SPACE = {"min_samples_leaf": ("int", 5, 2000, True), "max_features": ("cat", ["sqrt", "0.3", "0.5"])}
GRID = {"min_samples_leaf": [100, 200, 500], "max_features": ["sqrt", "0.5"]}


def parse_max_features(value):
    if value in ("sqrt", "log2"):
        return value
    return float(value)


def fit(fit_df, valid_df, cols, n_estimators=200, min_samples_leaf=200, max_features="sqrt", max_samples=0.2):
    m = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                               max_features=parse_max_features(max_features), max_samples=max_samples, bootstrap=True,
                               n_jobs=WORKERS, random_state=SEED)
    m.fit(tree_features(fit_df, cols), target(fit_df))
    return Fitted(m, cols, inputs="tree")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=FEATURE_SETS, default="context")
    ap.add_argument("--n-estimators", type=int, default=DEFAULTS["n_estimators"])
    ap.add_argument("--min-samples-leaf", type=int, default=DEFAULTS["min_samples_leaf"])
    ap.add_argument("--max-features", default=DEFAULTS["max_features"], help="sqrt, log2 or a share such as 0.5")
    ap.add_argument("--max-samples", type=float, default=DEFAULTS["max_samples"], help="share of the moves per tree")
    ap.add_argument("--max-games", type=int, default=TRAIN["max_train_games"])
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    params = {"n_estimators": args.n_estimators, "min_samples_leaf": args.min_samples_leaf,
              "max_features": args.max_features, "max_samples": args.max_samples}
    d, m = fit_and_save(METHOD, fit, args.features, params, args.max_games, args.name)

    importance = pd.Series(m.model.feature_importances_, index=m.columns, name="importance")
    importance.sort_values(ascending=False).to_csv(d / "importance.csv")


if __name__ == "__main__":
    main()
