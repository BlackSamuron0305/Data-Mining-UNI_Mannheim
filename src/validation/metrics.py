"""Scores of predicted blunder probabilities against what happened.

Blunders are about 5% of the moves, so accuracy says little (always "no blunder" gets 95%). The scores here judge
the probabilities themselves:

log_loss            mean -log of the probability given to what happened (lower is better); the main score
brier               mean squared difference between probability and outcome (lower is better)
brier_skill         1 - brier / brier of always predicting the observed rate (0 = no better than that, 1 = perfect)
roc_auc             chance that a blunder gets a higher probability than a non-blunder (0.5 = guessing)
pr_auc              average precision: area under precision against recall, the base rate is guessing
ece                 expected calibration error: mean gap between predicted and observed rate over 20 bins of equal
                    size, weighted by the bin size (0 = calibrated)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-7


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 20) -> pd.DataFrame:
    """Moves, mean prediction and observed rate in `bins` bins of equal size by prediction."""
    order = np.argsort(p, kind="stable")
    parts = np.array_split(order, bins)
    return pd.DataFrame({"bin": np.arange(len(parts)), "moves": [len(i) for i in parts],
                         "mean_p": [p[i].mean() for i in parts], "rate": [y[i].mean() for i in parts]})


def scores(y, p) -> dict:
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    base = y.mean()
    brier = float(np.mean((p - y) ** 2))
    both = 0 < base < 1
    cal = calibration(y, p)
    return {"moves": len(y), "blunders": int(y.sum()), "rate": float(base), "mean_p": float(p.mean()),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": brier, "brier_skill": 1 - brier / (base * (1 - base)) if both else np.nan,
            "roc_auc": float(roc_auc_score(y, p)) if both else np.nan,
            "pr_auc": float(average_precision_score(y, p)) if both else np.nan,
            "ece": float(np.sum(cal["moves"] * (cal["mean_p"] - cal["rate"]).abs()) / len(y))}
