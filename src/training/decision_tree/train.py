"""Decision tree (CART): the symbolic model, whose splits can be read as rules.

Usage:  uv run python -m src.training.decision_tree.train [--features basic|context|engine]
            [--depths 2 3 4 5 6 8 10 12] [--min-samples-leaf 1000] [--rules-depth 4] [--max-games N] [--name NAME]

One tree per depth in --depths is fitted on the training games without the held-out ones; the depth with the lowest
log loss on the held-out games is kept (tuning.csv). The probability of a leaf is its blunder rate pulled towards the
rate of all training moves by SHRINK moves (as the bucket rates, src/preprocessing/bucket_rates/rates.py), so a
leaf without a blunder does not predict 0. Speed enters as its rank (bullet < blitz < rapid < classical); missing
values are routed by the tree itself.

The chosen tree is usually too deep to read, so a second tree of --rules-depth is fitted for the rules in the paper.

Writes data/runs/decision_tree/<name>/ (name: the feature set), see src/training/common.py, plus
tuning.csv                 depth, leaves, log loss on the fitting and the held-out moves
rules.txt                  the rules of the tree of --rules-depth (blunder probability per leaf, moves per leaf)
importance.csv             impurity decrease per feature of the chosen tree
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from src.config import SEED
from src.preprocessing.bucket_rates.rates import SHRINK
from src.training.common import FEATURE_SETS, TARGET, TRAIN, load, run_dir, save_run, tree_features
from src.validation.metrics import scores

METHOD = "decision_tree"


class ShrunkTree:
    """A fitted tree whose leaf probabilities are pulled towards the overall rate (predict_proba as scikit-learn)."""

    def __init__(self, tree: DecisionTreeClassifier, prior: float, shrink: float = SHRINK):
        self.tree, self.prior = tree, prior
        n = tree.tree_.n_node_samples.astype(float)
        p = tree.tree_.value[:, 0, 1]                      # share of blunders per node (scikit-learn >= 1.4)
        self.node_p = (p * n + shrink * prior) / (n + shrink)

    def predict_proba(self, x) -> np.ndarray:
        p = self.node_p[self.tree.apply(x)]
        return np.column_stack([1 - p, p])


def fit(x, y, depth: int, min_leaf: int) -> ShrunkTree:
    t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf, random_state=SEED).fit(x, y)
    return ShrunkTree(t, float(np.mean(y)))


def rules(m: ShrunkTree, names: list[str]) -> str:
    """The tree as indented rules, every leaf with its (shrunk) blunder probability and moves."""
    text = export_text(m.tree, feature_names=names, show_weights=False, decimals=2)
    leaves = iter(np.flatnonzero(m.tree.tree_.children_left == -1))
    out = []
    for line in text.splitlines():                                  # leaves are printed in node order
        if "class:" in line:
            leaf = next(leaves)
            line = line.split("class:")[0] + (f"blunder probability {100 * m.node_p[leaf]:.2f}% "
                                              f"({m.tree.tree_.n_node_samples[leaf]:,} moves)")
        out.append(line)
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="context")
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8, 10, 12])
    ap.add_argument("--min-samples-leaf", type=int, default=1000)
    ap.add_argument("--rules-depth", type=int, default=4)
    ap.add_argument("--max-games", type=int, default=TRAIN["max_train_games"])
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    t0 = time.time()
    train = load("train", args.features, args.max_games)
    test = load("test", args.features)
    fit_df, valid = train[train["fold"] == "train"], train[train["fold"] == "valid"]
    xf, xv = tree_features(fit_df, args.features), tree_features(valid, args.features)
    yf, yv = fit_df[TARGET].to_numpy(), valid[TARGET].to_numpy()

    tuning, best = [], None
    for depth in sorted(set(args.depths)):
        m = fit(xf, yf, depth, args.min_samples_leaf)
        row = {"depth": depth, "leaves": int(m.tree.get_n_leaves()),
               "log_loss_fit": scores(yf, m.predict_proba(xf)[:, 1])["log_loss"],
               "log_loss_valid": scores(yv, m.predict_proba(xv)[:, 1])["log_loss"]}
        tuning.append(row)
        print(f"depth {depth:2d}: {row['leaves']:5d} leaves, log loss fit {row['log_loss_fit']:.5f}, "
              f"valid {row['log_loss_valid']:.5f}")
        if best is None or row["log_loss_valid"] < best[1]["log_loss_valid"]:
            best = (m, row)
    m, chosen = best

    preds = {part: df.assign(p=m.predict_proba(tree_features(df, args.features))[:, 1])
             for part, df in (("valid", valid), ("test", test))}
    name = args.name or args.features
    d = save_run(METHOD, name, {"features": args.features, "depth": chosen["depth"],
                                "min_samples_leaf": args.min_samples_leaf, "shrink": SHRINK,
                                "max_train_games": args.max_games}, preds, t0,
                 model={"tree": m.tree, "leaf_p": m.node_p},          # p of a move: leaf_p[tree.apply(x)]
                 extra={"fit_rows": len(fit_df), "leaves": chosen["leaves"]})
    pd.DataFrame(tuning).to_csv(d / "tuning.csv", index=False)
    names = list(xf.columns)
    pd.Series(m.tree.feature_importances_, index=names, name="importance").sort_values(ascending=False) \
        .to_csv(d / "importance.csv")
    readable = fit(xf, yf, args.rules_depth, args.min_samples_leaf)
    (run_dir(METHOD, name) / "rules.txt").write_text(rules(readable, names), encoding="utf-8")
    print(f"{d}: depth {chosen['depth']} ({chosen['leaves']} leaves) chosen on {len(valid):,} held-out moves")


if __name__ == "__main__":
    main()
