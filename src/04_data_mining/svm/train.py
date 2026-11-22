"""Linear support vector machine (SGDClassifier, hinge loss), turned into probabilities by Platt scaling.

Usage:  uv run python -m src.04_data_mining.svm.train [--features basic|context|engine] [--alpha 1e-5]
            [--calibration-frac 0.2] [--max-epochs 20] [--patience 3] [--unweighted] [--max-games N] [--name NAME]
"""
import argparse
import copy

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier

from src.config import SEED
from src.common.metrics import scores
from ..common import (FEATURE_SETS, TRAIN, Fitted, features, fit_and_save, numeric_prep, split_games,
                                 target)

METHOD = "svm"
DEFAULTS = {"alpha": 1e-5, "calibration_frac": 0.2, "max_epochs": 50, "patience": 3, "balanced": True}
SPACE = {"alpha": ("float", 1e-8, 1e-1, True)}
GRID = {"alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]}


class PlattSVM:
    """Preprocessing, linear SVM and the Platt sigmoid on its distance."""

    def __init__(self, prep, svm, platt):
        self.prep = prep
        self.svm = svm
        self.platt = platt

    def predict_proba(self, x):
        d = self.svm.decision_function(self.prep.transform(x))
        return self.platt.predict_proba(d.reshape(-1, 1))


def platt(d, y):
    return LogisticRegression(C=1e4).fit(d.reshape(-1, 1), y)


def fit(fit_df, valid_df, cols, alpha=1e-5, calibration_frac=0.2, max_epochs=50, patience=3, balanced=True):
    svm_df, cal_df = split_games(fit_df, calibration_frac, "ie500-platt")
    prep = numeric_prep(cols)
    x_svm = prep.fit_transform(features(svm_df, cols))
    y_svm = target(svm_df)
    # partial_fit does not shuffle, so mix the moves once
    order = np.random.default_rng(SEED).permutation(len(y_svm))
    x_svm = x_svm[order]
    y_svm = y_svm[order]
    x_cal = prep.transform(features(cal_df, cols))
    y_cal = target(cal_df)
    x_valid = prep.transform(features(valid_df, cols))
    y_valid = target(valid_df)

    weights = None
    if balanced:
        n_other = max((y_svm == 0).sum(), 1)
        n_blunder = max((y_svm == 1).sum(), 1)
        weights = {0: len(y_svm) / (2 * n_other), 1: len(y_svm) / (2 * n_blunder)}
    svm = SGDClassifier(loss="hinge", alpha=alpha, learning_rate="optimal", class_weight=weights, random_state=SEED)

    history = []
    best_svm = None
    best_cal = None
    best_loss = None
    best_epoch = None
    since_best = 0
    for epoch in range(1, max_epochs + 1):
        svm.partial_fit(x_svm, y_svm, classes=[0, 1])
        cal = platt(svm.decision_function(x_cal), y_cal)
        p_valid = cal.predict_proba(svm.decision_function(x_valid).reshape(-1, 1))[:, 1]
        loss = scores(y_valid, p_valid)["log_loss"]
        history.append({"epoch": epoch, "log_loss_valid": loss})
        print(f"epoch {epoch:3d}: valid {loss:.5f}")
        if best_loss is None or loss < best_loss - 1e-6:
            best_svm = copy.deepcopy(svm)
            best_cal = cal
            best_loss = loss
            best_epoch = epoch
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break
    info = {"epochs": len(history), "best_epoch": best_epoch, "svm_rows": len(y_svm), "calibration_rows": len(y_cal)}
    return Fitted(PlattSVM(prep, best_svm, best_cal), cols, info=info, history=history)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=FEATURE_SETS, default="context")
    ap.add_argument("--alpha", type=float, default=DEFAULTS["alpha"], help="strength of the L2 penalty")
    ap.add_argument("--calibration-frac", type=float, default=DEFAULTS["calibration_frac"],
                    help="share of the fitting games for the Platt scaling")
    ap.add_argument("--max-epochs", type=int, default=DEFAULTS["max_epochs"])
    ap.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    ap.add_argument("--unweighted", dest="balanced", action="store_false", help="no class weights")
    ap.add_argument("--max-games", type=int, default=TRAIN["max_train_games"])
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    params = {"alpha": args.alpha, "calibration_frac": args.calibration_frac, "max_epochs": args.max_epochs,
              "patience": args.patience, "balanced": args.balanced}
    fit_and_save(METHOD, fit, args.features, params, args.max_games, args.name)


if __name__ == "__main__":
    # run main() from the module itself, so that the saved model can be loaded by other scripts
    import importlib
    importlib.import_module("src.04_data_mining.svm.train").main()
