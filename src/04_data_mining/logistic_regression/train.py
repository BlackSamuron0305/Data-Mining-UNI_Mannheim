"""Logistic regression on the standardised features of a move (no class weights, so the outputs stay probabilities).

Usage:  uv run python -m src.04_data_mining.logistic_regression.train [--features basic|context|engine] [--C 1.0]
            [--max-games N] [--name NAME]
"""
import argparse

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import SEED
from ..common import FEATURE_SETS, Fitted, features, fit_and_save, numeric_prep, target

METHOD = "logistic_regression"
DEFAULTS = {"C": 1.0}
SPACE = {"C": ("float", 1e-3, 1e2, True)}
GRID = {"C": [0.01, 0.1, 1.0, 10.0]}


def fit(fit_df, valid_df, cols, C=1.0):
    m = Pipeline([("prep", numeric_prep(cols)), ("lr", LogisticRegression(C=C, max_iter=2000, random_state=SEED))])
    m.fit(features(fit_df, cols), target(fit_df))
    return Fitted(m, cols, info={"iterations": int(m["lr"].n_iter_[0])})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=FEATURE_SETS, default="context")
    ap.add_argument("--C", type=float, default=DEFAULTS["C"], help="inverse strength of the L2 penalty")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    d, m = fit_and_save(METHOD, fit, args.features, {"C": args.C}, args.max_games, args.name)

    pipeline = m.model
    coef = pd.Series(pipeline["lr"].coef_[0], index=pipeline["prep"].get_feature_names_out(), name="coefficient")
    coef.sort_values(key=abs, ascending=False).to_csv(d / "coefficients.csv")


if __name__ == "__main__":
    main()
