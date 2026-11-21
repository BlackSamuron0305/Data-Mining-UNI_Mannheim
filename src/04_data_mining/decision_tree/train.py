"""Decision tree (CART) whose leaf rates are pulled towards the overall rate; the depth is chosen on the held-out games.

Usage:  uv run python -m src.04_data_mining.decision_tree.train [--features basic|context|engine]
            [--depths 2 3 4 5 6 8 10 12] [--min-samples-leaf 1000] [--rules-depth 4] [--max-games N] [--name NAME]

Also writes tuning.csv, importance.csv and rules.txt (a small tree of --rules-depth for the paper).
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from src.config import SEED
from src.shared.buckets import SHRINK
from src.shared.metrics import scores
from ..common import (FEATURE_SETS, TRAIN, Fitted, columns, folds, load, predictions, run_dir, save_run,
                                 target, tree_features)

METHOD = "decision_tree"
DEFAULTS = {"max_depth": 8, "min_samples_leaf": 1000}
SPACE = {"max_depth": ("int", 2, 24, False), "min_samples_leaf": ("int", 100, 10000, True)}
GRID = {"max_depth": [2, 3, 4, 5, 6, 8, 10, 12], "min_samples_leaf": [1000]}


class ShrunkTree:
    """A fitted tree whose leaf probabilities are pulled towards the overall rate."""

    def __init__(self, tree, prior, shrink=SHRINK):
        self.tree = tree
        self.prior = prior
        n = tree.tree_.n_node_samples.astype(float)
        p = tree.tree_.value[:, 0, 1]
        self.node_p = (p * n + shrink * prior) / (n + shrink)

    def predict_proba(self, x):
        p = self.node_p[self.tree.apply(x)]
        return np.column_stack([1 - p, p])


def fit(fit_df, valid_df, cols, max_depth=8, min_samples_leaf=1000):
    y = target(fit_df)
    t = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=SEED)
    t.fit(tree_features(fit_df, cols), y)
    return Fitted(ShrunkTree(t, float(np.mean(y))), cols, inputs="tree", info={"leaves": int(t.get_n_leaves())})


def rules(m, names):
    """The tree as indented rules, every leaf with its blunder probability and moves."""
    text = export_text(m.tree, feature_names=names, show_weights=False, decimals=2)
    leaves = list(np.flatnonzero(m.tree.tree_.children_left == -1))
    lines = []
    i = 0
    for line in text.splitlines():
        if "class:" in line:
            leaf = leaves[i]
            i += 1
            line = (line.split("class:")[0] + f"blunder probability {100 * m.node_p[leaf]:.2f}% "
                    f"({m.tree.tree_.n_node_samples[leaf]:,} moves)")
        lines.append(line)
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="context")
    ap.add_argument("--depths", type=int, nargs="+", default=GRID["max_depth"])
    ap.add_argument("--min-samples-leaf", type=int, default=DEFAULTS["min_samples_leaf"])
    ap.add_argument("--rules-depth", type=int, default=4)
    ap.add_argument("--max-games", type=int, default=TRAIN["max_train_games"])
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    t0 = time.time()
    cols = columns(args.features)
    fit_df, valid = folds(load("train", args.features, args.max_games))
    y_fit = target(fit_df)
    y_valid = target(valid)

    tuning = []
    best_model = None
    best_row = None
    for depth in sorted(set(args.depths)):
        m = fit(fit_df, None, cols, depth, args.min_samples_leaf)
        row = {"depth": depth, "leaves": m.info["leaves"],
               "log_loss_fit": scores(y_fit, m.predict(fit_df))["log_loss"],
               "log_loss_valid": scores(y_valid, m.predict(valid))["log_loss"]}
        tuning.append(row)
        print(f"depth {depth:2d}: {row['leaves']:5d} leaves, log loss fit {row['log_loss_fit']:.5f}, "
              f"valid {row['log_loss_valid']:.5f}")
        if best_row is None or row["log_loss_valid"] < best_row["log_loss_valid"]:
            best_model = m
            best_row = row
    m = best_model

    test = load("test", args.features)
    name = args.name or args.features
    params = {"max_depth": best_row["depth"], "min_samples_leaf": args.min_samples_leaf}
    settings = {"features": args.features, "depth": best_row["depth"], "min_samples_leaf": args.min_samples_leaf,
                "params": params, "shrink": SHRINK, "max_train_games": args.max_games}
    d = save_run(METHOD, name, settings, predictions(m, {"valid": valid, "test": test}), t0, model=m,
                 extra={"fit_rows": len(fit_df), "leaves": best_row["leaves"]})
    pd.DataFrame(tuning).to_csv(d / "tuning.csv", index=False)

    names = list(m.x(fit_df.iloc[:1]).columns)
    importance = pd.Series(m.model.tree.feature_importances_, index=names, name="importance")
    importance.sort_values(ascending=False).to_csv(d / "importance.csv")
    readable = fit(fit_df, None, cols, args.rules_depth, args.min_samples_leaf).model
    (run_dir(METHOD, name) / "rules.txt").write_text(rules(readable, names), encoding="utf-8")
    print(f"{d}: depth {best_row['depth']} ({best_row['leaves']} leaves) chosen on {len(valid):,} held-out moves")


if __name__ == "__main__":
    # run main() from the module itself, so that the saved model can be loaded by other scripts
    import importlib
    importlib.import_module("src.04_data_mining.decision_tree.train").main()
