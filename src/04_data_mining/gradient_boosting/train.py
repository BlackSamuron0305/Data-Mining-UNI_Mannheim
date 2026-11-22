"""Gradient boosting (scikit-learn's HistGradientBoostingClassifier), stopped early on the held-out games.

Usage:  uv run python -m src.04_data_mining.gradient_boosting.train [--features basic|context|engine]
            [--learning-rate 0.1] [--max-leaf-nodes 63] [--min-samples-leaf 200] [--l2-regularization 1.0]
            [--max-iter 1000] [--max-games N] [--name NAME]
"""
import argparse

from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import SEED
from ..common import FEATURE_SETS, TRAIN, Fitted, features, fit_and_save, target

METHOD = "gradient_boosting"
DEFAULTS = {"learning_rate": 0.1, "max_leaf_nodes": 63, "min_samples_leaf": 200, "l2_regularization": 1.0,
            "max_iter": 1000}
SPACE = {"learning_rate": ("float", 0.02, 0.3, True), "max_leaf_nodes": ("int", 15, 255, True),
         "min_samples_leaf": ("int", 20, 5000, True), "l2_regularization": ("float", 1e-3, 30.0, True)}
GRID = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [31, 63, 127]}
PATIENCE = 30


def fit(fit_df, valid_df, cols, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200, l2_regularization=1.0,
        max_iter=1000):
    m = HistGradientBoostingClassifier(learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                                       min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
                                       max_iter=max_iter, early_stopping=True, n_iter_no_change=PATIENCE,
                                       scoring="loss", categorical_features="from_dtype", random_state=SEED)
    m.fit(features(fit_df, cols), target(fit_df), X_val=features(valid_df, cols), y_val=target(valid_df))
    return Fitted(m, cols, info={"iterations": int(m.n_iter_)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=FEATURE_SETS, default="context")
    ap.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    ap.add_argument("--max-leaf-nodes", type=int, default=DEFAULTS["max_leaf_nodes"])
    ap.add_argument("--min-samples-leaf", type=int, default=DEFAULTS["min_samples_leaf"])
    ap.add_argument("--l2-regularization", type=float, default=DEFAULTS["l2_regularization"])
    ap.add_argument("--max-iter", type=int, default=DEFAULTS["max_iter"])
    ap.add_argument("--max-games", type=int, default=TRAIN["max_train_games"])
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    params = {"learning_rate": args.learning_rate, "max_leaf_nodes": args.max_leaf_nodes,
              "min_samples_leaf": args.min_samples_leaf, "l2_regularization": args.l2_regularization,
              "max_iter": args.max_iter}
    fit_and_save(METHOD, fit, args.features, params, args.max_games, args.name)


if __name__ == "__main__":
    main()
