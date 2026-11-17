"""Gaussian naive Bayes: every feature normally distributed within each class, the features independent of each
other given the class.

Usage:  uv run python -m src.training.naive_bayes.train [--features basic|context|engine]
            [--transform quantile|standard] [--max-games N] [--name NAME]

The simplest probabilistic learner: one mean and one variance per feature and class, no search. Missing values are
filled with the median plus an indicator column, speed is one column per speed (src/training/common.py). Clocks,
times and counts are strongly skewed, so by default every number is first mapped to a normal distribution by its
quantile (--transform quantile, fitted on the fitting moves), which makes the Gaussian assumption plausible for each
feature on its own. Its independence assumption is broken on purpose: clock, clock_rel and clock_diff say much the
same, and the model counts their evidence more than once. Fitted on the training games without the held-out ones.

Writes data/runs/naive_bayes/<name>/ (name: the feature set), see src/training/common.py.
"""
from __future__ import annotations

import argparse
import time

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from src.config import SEED
from src.training.common import FEATURE_SETS, TARGET, TRAIN, features, load, numeric_prep, save_run

METHOD = "naive_bayes"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="context")
    ap.add_argument("--transform", choices=["quantile", "standard"], default="quantile")
    ap.add_argument("--max-games", type=int, default=TRAIN["max_train_games"])
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    t0 = time.time()
    train = load("train", args.features, args.max_games)
    test = load("test", args.features)
    fit, valid = train[train["fold"] == "train"], train[train["fold"] == "valid"]
    scaler = (QuantileTransformer(output_distribution="normal", subsample=200_000, random_state=SEED)
              if args.transform == "quantile" else StandardScaler())
    m = Pipeline([("prep", numeric_prep(args.features, scaler)), ("nb", GaussianNB())])
    m.fit(features(fit, args.features), fit[TARGET])
    preds = {part: df.assign(p=m.predict_proba(features(df, args.features))[:, 1])
             for part, df in (("valid", valid), ("test", test))}
    d = save_run(METHOD, args.name or args.features,
                 {"features": args.features, "transform": args.transform, "max_train_games": args.max_games},
                 preds, t0, model=m, extra={"fit_rows": len(fit)})
    print(f"{d}: fitted on {len(fit):,} moves, valid {len(valid):,}, test {len(test):,}")


if __name__ == "__main__":
    main()
