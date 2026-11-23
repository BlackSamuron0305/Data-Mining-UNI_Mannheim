"""Neural network (scikit-learn's MLPClassifier), trained epoch by epoch with early stopping on the held-out games.

Usage:  uv run python -m src.04_data_mining.neural_network.train [--features basic|context|engine] [--hidden 64 32]
            [--alpha 1e-4] [--learning-rate 1e-3] [--batch-size 1024] [--max-epochs 50] [--patience 5]
            [--max-games N] [--name NAME]
"""
import argparse
import copy

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.config import SEED
from src.common.metrics import scores
from ..common import FEATURE_SETS, Fitted, features, fit_and_save, numeric_prep, target

METHOD = "neural_network"
DEFAULTS = {"hidden": "64-32", "alpha": 1e-4, "learning_rate": 1e-3, "batch_size": 1024, "max_epochs": 100,
            "patience": 5}
SPACE = {"hidden": ("cat", ["32", "64-32", "128-64", "256-128"]), "alpha": ("float", 1e-6, 1e-1, True),
         "learning_rate": ("float", 3e-4, 1e-2, True)}
GRID = {"hidden": ["32", "64-32", "128-64"], "alpha": [1e-5, 1e-4, 1e-3]}


def layers(hidden):
    """Hidden layer sizes from a text like "64-32"."""
    sizes = []
    for h in hidden.split("-"):
        sizes.append(int(h))
    return tuple(sizes)


def fit(fit_df, valid_df, cols, hidden="64-32", alpha=1e-4, learning_rate=1e-3, batch_size=1024, max_epochs=100,
        patience=5):
    prep = numeric_prep(cols)
    x_fit = prep.fit_transform(features(fit_df, cols)).astype(np.float32)
    x_valid = prep.transform(features(valid_df, cols)).astype(np.float32)
    y_fit = target(fit_df)
    y_valid = target(valid_df)
    net = MLPClassifier(hidden_layer_sizes=layers(hidden), activation="relu", solver="adam", alpha=alpha,
                        learning_rate_init=learning_rate, batch_size=batch_size, shuffle=True, random_state=SEED)
    history = []
    best_net = None
    best_loss = None
    best_epoch = None
    since_best = 0
    for epoch in range(1, max_epochs + 1):
        net.partial_fit(x_fit, y_fit, classes=[0, 1])
        loss = scores(y_valid, net.predict_proba(x_valid)[:, 1])["log_loss"]
        history.append({"epoch": epoch, "log_loss_fit": float(net.loss_), "log_loss_valid": loss})
        print(f"epoch {epoch:3d}: fit {net.loss_:.5f}, valid {loss:.5f}")
        if best_loss is None or loss < best_loss - 1e-6:
            best_net = copy.deepcopy(net)
            best_loss = loss
            best_epoch = epoch
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                break
    model = Pipeline([("prep", prep), ("mlp", best_net)])
    return Fitted(model, cols, info={"epochs": len(history), "best_epoch": best_epoch}, history=history)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=FEATURE_SETS, default="context")
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 32], help="units per hidden layer")
    ap.add_argument("--alpha", type=float, default=DEFAULTS["alpha"], help="strength of the L2 penalty")
    ap.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--max-epochs", type=int, default=DEFAULTS["max_epochs"])
    ap.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    params = {"hidden": "-".join(str(h) for h in args.hidden), "alpha": args.alpha,
              "learning_rate": args.learning_rate, "batch_size": args.batch_size, "max_epochs": args.max_epochs,
              "patience": args.patience}
    fit_and_save(METHOD, fit, args.features, params, args.max_games, args.name)


if __name__ == "__main__":
    main()
