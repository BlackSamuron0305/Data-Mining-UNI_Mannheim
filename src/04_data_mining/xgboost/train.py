"""XGBoost: gradient-boosted trees, stopped early on the held-out games (--device cuda for an NVIDIA GPU).

Usage:  uv run python -m src.04_data_mining.xgboost.train [--features basic|context|engine] [--learning-rate 0.1]
            [--max-depth 6] [--min-child-weight 10] [--reg-lambda 1.0] [--subsample 0.8] [--colsample-bytree 0.8]
            [--max-rounds 3000] [--device cpu|cuda] [--max-games N] [--name NAME]
"""
import argparse

import pandas as pd

from src.config import SEED, WORKERS
from ..common import FEATURE_SETS, Fitted, features, fit_and_save, target

METHOD = "xgboost"
DEFAULTS = {"learning_rate": 0.1, "max_depth": 6, "min_child_weight": 10.0, "reg_lambda": 1.0, "subsample": 0.8,
            "colsample_bytree": 0.8, "max_rounds": 3000}
SPACE = {"learning_rate": ("float", 0.02, 0.3, True), "max_depth": ("int", 3, 10, False),
         "min_child_weight": ("float", 1.0, 1000.0, True), "reg_lambda": ("float", 1e-3, 30.0, True),
         "subsample": ("float", 0.5, 1.0, False), "colsample_bytree": ("float", 0.4, 1.0, False)}
GRID = {"learning_rate": [0.05, 0.1], "max_depth": [4, 6, 8], "min_child_weight": [10.0, 100.0]}
PATIENCE = 50


def fit(fit_df, valid_df, cols, learning_rate=0.1, max_depth=6, min_child_weight=10.0, reg_lambda=1.0,
        subsample=0.8, colsample_bytree=0.8, max_rounds=3000, device=None):
    from xgboost import XGBClassifier

    if not device:
        device = "cpu"
    m = XGBClassifier(n_estimators=max_rounds, learning_rate=learning_rate, max_depth=max_depth,
                      min_child_weight=min_child_weight, reg_lambda=reg_lambda, subsample=subsample,
                      colsample_bytree=colsample_bytree, tree_method="hist", max_bin=256, enable_categorical=True,
                      objective="binary:logistic", eval_metric="logloss", early_stopping_rounds=PATIENCE,
                      device=device, n_jobs=WORKERS, random_state=SEED)
    m.fit(features(fit_df, cols), target(fit_df), eval_set=[(features(valid_df, cols), target(valid_df))],
          verbose=False)
    # predict on the CPU, the moves are in main memory anyway
    m.set_params(device="cpu")
    return Fitted(m, cols, info={"iterations": int(m.best_iteration) + 1})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=FEATURE_SETS, default="context")
    ap.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    ap.add_argument("--max-depth", type=int, default=DEFAULTS["max_depth"])
    ap.add_argument("--min-child-weight", type=float, default=DEFAULTS["min_child_weight"])
    ap.add_argument("--reg-lambda", type=float, default=DEFAULTS["reg_lambda"], help="L2 penalty of the leaf values")
    ap.add_argument("--subsample", type=float, default=DEFAULTS["subsample"], help="share of the moves per tree")
    ap.add_argument("--colsample-bytree", type=float, default=DEFAULTS["colsample_bytree"],
                    help="share of the features per tree")
    ap.add_argument("--max-rounds", type=int, default=DEFAULTS["max_rounds"])
    ap.add_argument("--device", default=None, help="cpu (default) or cuda for an NVIDIA GPU")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    params = {"learning_rate": args.learning_rate, "max_depth": args.max_depth,
              "min_child_weight": args.min_child_weight, "reg_lambda": args.reg_lambda, "subsample": args.subsample,
              "colsample_bytree": args.colsample_bytree, "max_rounds": args.max_rounds, "device": args.device}
    d, m = fit_and_save(METHOD, fit, args.features, params, args.max_games, args.name)

    gain = m.model.get_booster().get_score(importance_type="total_gain")
    importance = pd.Series(gain, name="total_gain").reindex(m.columns, fill_value=0.0)
    importance.sort_values(ascending=False).to_csv(d / "importance.csv")


if __name__ == "__main__":
    main()
