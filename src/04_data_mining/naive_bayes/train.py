"""Gaussian naive Bayes, with every number first mapped to a normal distribution (or only standardised).

Usage:  uv run python -m src.04_data_mining.naive_bayes.train [--features basic|context|engine]
            [--transform quantile|standard] [--var-smoothing 1e-2] [--max-games N] [--name NAME]
"""
import argparse

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from ..common import FEATURE_SETS, Fitted, features, fit_and_save, numeric_prep, scaler, target

METHOD = "naive_bayes"
DEFAULTS = {"transform": "quantile", "var_smoothing": 1e-2}
SPACE = {"transform": ("cat", ["quantile", "standard"]), "var_smoothing": ("float", 1e-4, 10.0, True)}
GRID = {"transform": ["quantile", "standard"], "var_smoothing": [1e-3, 1e-2, 1e-1]}


def fit(fit_df, valid_df, cols, transform="quantile", var_smoothing=1e-2):
    m = Pipeline([("prep", numeric_prep(cols, scaler(transform))), ("nb", GaussianNB(var_smoothing=var_smoothing))])
    m.fit(features(fit_df, cols), target(fit_df))
    return Fitted(m, cols)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=FEATURE_SETS, default="context")
    ap.add_argument("--transform", choices=["quantile", "standard"], default=DEFAULTS["transform"])
    ap.add_argument("--var-smoothing", type=float, default=DEFAULTS["var_smoothing"])
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    params = {"transform": args.transform, "var_smoothing": args.var_smoothing}
    fit_and_save(METHOD, fit, args.features, params, args.max_games, args.name)


if __name__ == "__main__":
    main()
