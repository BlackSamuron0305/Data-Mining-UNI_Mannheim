"""Scores of predicted blunder probabilities against what happened.

Log loss, Brier score, ROC-AUC, PR-AUC and calibration error, plus the scores at a cut-off (best F1 on valid)
and among the 5% of the moves with the highest prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-7


def calibration(y: np.ndarray, p: np.ndarray, bins: int = 20) -> pd.DataFrame:
    """Moves, mean prediction and observed rate in bins of equal size by prediction."""
    order = np.argsort(p, kind="stable")
    parts = np.array_split(order, bins)
    moves, mean_p, rate = [], [], []
    for i in parts:
        moves.append(len(i))
        mean_p.append(p[i].mean())
        rate.append(y[i].mean())
    return pd.DataFrame({"bin": np.arange(len(parts)), "moves": moves, "mean_p": mean_p, "rate": rate})


def move_loss(y, p) -> np.ndarray:
    """Log loss of every move."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def scores(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    base = y.mean()
    brier = float(np.mean((p - y) ** 2))
    both = 0 < base < 1
    cal = calibration(y, p)
    out = {"moves": len(y), "blunders": int(y.sum()), "rate": float(base), "mean_p": float(p.mean()),
           "log_loss": float(move_loss(y, p).mean()), "brier": brier}
    if both:
        out["brier_skill"] = 1 - brier / (base * (1 - base))
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["brier_skill"] = np.nan
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    out["ece"] = float(np.sum(cal["moves"] * (cal["mean_p"] - cal["rate"]).abs()) / len(y))
    return out


def f32(p) -> np.ndarray:
    """Predictions as float32, so that a cut-off flags the same moves in every file."""
    return np.asarray(p, dtype=np.float32)


def ranking(y, p) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Blunders and other moves flagged at every distinct prediction, highest first, and that prediction."""
    y = np.asarray(y, dtype=float)
    p = f32(p)
    order = np.argsort(-p, kind="stable")
    ps = p[order]
    ys = y[order]
    last = np.r_[ps[1:] != ps[:-1], True]
    return np.cumsum(ys)[last], np.cumsum(1 - ys)[last], ps[last]


def best_cutoff(y, p) -> float:
    """The cut-off with the highest F1."""
    tp, fp, cut = ranking(y, p)
    f1 = 2 * tp / np.maximum(tp + fp + tp[-1], 1)
    return float(cut[np.argmax(f1)])


def at_cutoff(y, p, cut: float) -> dict:
    y = np.asarray(y, dtype=bool)
    flag = f32(p) >= np.float32(cut)
    tp = float((flag & y).sum())
    fp = float((flag & ~y).sum())
    fn = float(y.sum()) - tp
    tn = float((~y).sum()) - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    rec = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {"cutoff": float(cut), "flagged": float(flag.mean()), "precision": prec, "recall": rec, "f1": f1,
            "mcc": mcc, "balanced_accuracy": (rec + spec) / 2}


def top_name(k: float) -> str:
    return f"top{100 * k:g}"


def top_share(y, p, k: float) -> dict:
    """Precision, recall and lift among the share k of the moves with the highest prediction."""
    y = np.asarray(y, dtype=float)
    p = f32(p)
    m = max(1, int(round(k * len(p))))
    edge = -np.partition(-p, m - 1)[m - 1]
    above = p > edge
    tie = p == edge
    # moves tied at the border count with the share of them that fits in
    caught = y[above].sum() + (m - above.sum()) * y[tie].mean()
    rate = y.mean()
    t = top_name(k)
    recall = caught / y.sum() if y.sum() else np.nan
    lift = caught / m / rate if rate else np.nan
    return {f"precision_{t}": caught / m, f"recall_{t}": recall, f"lift_{t}": lift}


def decisions(y, p, cut: float, k: float) -> dict:
    """The scores at the cut-off and among the top share k."""
    out = at_cutoff(y, p, cut)
    out.update(top_share(y, p, k))
    return out
