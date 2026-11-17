"""Neural network: a multi-layer perceptron (scikit-learn's MLPClassifier) on the features of a move.

Usage:  uv run python -m src.training.neural_network.train [--features basic|context|engine] [--hidden 64 32]
            [--alpha 1e-4] [--learning-rate 1e-3] [--batch-size 1024] [--max-epochs 50] [--patience 5]
            [--max-games N] [--name NAME]

The non-symbolic counterpart of the trees: smooth functions of all features at once instead of axis-parallel
splits. Numbers are filled and standardised as for the logistic regression (src/training/common.py), which a network
needs. The output is a sigmoid trained on log loss, so the predictions are probabilities.

Early stopping on the held-out games: after every epoch (one pass over the fitting moves in mini-batches) the log
loss of the held-out games is measured, and the network of the best epoch is kept once --patience epochs bring no
improvement. scikit-learn's own early stopping would hold out random moves instead, so moves of one game would land
on both sides.

Writes data/runs/neural_network/<name>/ (name: the feature set), see src/training/common.py, plus
epochs.csv     log loss of the fitting moves (scikit-learn's running loss) and of the held-out moves per epoch
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.config import SEED
from src.training.common import FEATURE_SETS, TARGET, TRAIN, features, load, numeric_prep, save_run
from src.validation.metrics import scores

METHOD = "neural_network"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="context")
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 32], help="units per hidden layer")
    ap.add_argument("--alpha", type=float, default=1e-4, help="strength of the L2 penalty")
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-games", type=int, default=TRAIN["max_train_games"])
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    t0 = time.time()
    train = load("train", args.features, args.max_games)
    test = load("test", args.features)
    fit, valid = train[train["fold"] == "train"], train[train["fold"] == "valid"]
    prep = numeric_prep(args.features)
    xf = prep.fit_transform(features(fit, args.features)).astype(np.float32)
    xv = prep.transform(features(valid, args.features)).astype(np.float32)
    yf, yv = fit[TARGET].to_numpy(dtype=int), valid[TARGET].to_numpy(dtype=int)

    net = MLPClassifier(hidden_layer_sizes=tuple(args.hidden), activation="relu", solver="adam", alpha=args.alpha,
                        learning_rate_init=args.learning_rate, batch_size=args.batch_size, shuffle=True,
                        random_state=SEED)
    epochs, best, since = [], None, 0
    for epoch in range(1, args.max_epochs + 1):
        net.partial_fit(xf, yf, classes=[0, 1])
        loss = scores(yv, net.predict_proba(xv)[:, 1])["log_loss"]
        epochs.append({"epoch": epoch, "log_loss_fit": float(net.loss_), "log_loss_valid": loss})
        print(f"epoch {epoch:3d}: fit {net.loss_:.5f}, valid {loss:.5f}")
        if best is None or loss < best[1] - 1e-6:
            best, since = (copy.deepcopy(net), loss, epoch), 0
        else:
            since += 1
            if since >= args.patience:
                break
    net, _, best_epoch = best

    m = Pipeline([("prep", prep), ("mlp", net)])
    preds = {part: df.assign(p=m.predict_proba(features(df, args.features))[:, 1])
             for part, df in (("valid", valid), ("test", test))}
    settings = {"features": args.features, "hidden": args.hidden, "alpha": args.alpha,
                "learning_rate": args.learning_rate, "batch_size": args.batch_size,
                "max_train_games": args.max_games}
    d = save_run(METHOD, args.name or args.features, settings, preds, t0, model=m,
                 extra={"fit_rows": len(fit), "epochs": len(epochs), "best_epoch": best_epoch})
    pd.DataFrame(epochs).to_csv(d / "epochs.csv", index=False)
    print(f"{d}: best epoch {best_epoch} of {len(epochs)} on {len(fit):,} moves, valid {len(valid):,}, "
          f"test {len(test):,}")


if __name__ == "__main__":
    main()
