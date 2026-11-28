# Scores of the predicted blunder probabilities, used by training and evaluation
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-7


# Predicted against observed blunder rate, in bins
def calibration(y: np.ndarray, p: np.ndarray, bins: int = 20) -> pd.DataFrame:
    order = np.argsort(p, kind="stable")
    parts = np.array_split(order, bins)
    moves, mean_p, rate = [], [], []
    for i in parts:
        moves.append(len(i))
        mean_p.append(p[i].mean())
        rate.append(y[i].mean())
    return pd.DataFrame({"bin": np.arange(len(parts)), "moves": moves, "mean_p": mean_p, "rate": rate})


# Log loss of every move
def move_loss(y, p) -> np.ndarray:
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


# Predictions as float32, so that a cut-off flags the same moves in every file
def f32(p) -> np.ndarray:
    return np.asarray(p, dtype=np.float32)


# Blunders and other moves flagged at every prediction value, highest first
def ranking(y, p) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    p = f32(p)
    order = np.argsort(-p, kind="stable")
    ps = p[order]
    ys = y[order]
    last = np.r_[ps[1:] != ps[:-1], True]
    return np.cumsum(ys)[last], np.cumsum(1 - ys)[last], ps[last]


# The cut-off optimising a validation objective
def best_cutoff(y, p, objective: str = "f1", target_precision: float | None = None) -> float:
    tp, fp, cut = ranking(y, p)
    total_pos = tp[-1]
    if objective == "f1":
        f1 = 2 * tp / np.maximum(tp + fp + total_pos, 1)
        return float(cut[np.argmax(f1)])
    if objective in ("f0.5", "f05"):
        f05 = 1.25 * tp / np.maximum(tp + fp + 0.25 * total_pos, 1)
        return float(cut[np.argmax(f05)])
    if objective == "precision_target":
        if target_precision is None or not (0.0 < target_precision <= 1.0):
            raise ValueError("target_precision must be in (0, 1] when objective='precision_target'")
        prec = tp / np.maximum(tp + fp, 1)
        eligible = prec >= target_precision
        if np.any(eligible):
            # Maximize recall (tp) among eligible thresholds; np.argmax picks first index (highest cutoff) on ties
            score = np.where(eligible, tp, -1.0)
            return float(cut[int(np.argmax(score))])
        return float(cut[int(np.argmax(prec))])
    raise ValueError(f"unknown objective {objective!r}; expected 'f1', 'f0.5', 'f05', or 'precision_target'")


def _confusion_metrics(tp: float, fp: float, fn: float, tn: float) -> dict:
    total = tp + fp + fn + tn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    acc = (tp + tn) / total if total else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    f05_den = 1.25 * tp + 0.25 * fn + fp
    f05 = 1.25 * tp / f05_den if f05_den else np.nan
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {
        "accuracy": float(acc) if not np.isnan(acc) else np.nan,
        "precision": float(prec),
        "recall": float(rec) if not np.isnan(rec) else np.nan,
        "f1": float(f1) if not np.isnan(f1) else np.nan,
        "f05": float(f05) if not np.isnan(f05) else np.nan,
        "mcc": float(mcc),
        "balanced_accuracy": float((rec + spec) / 2) if not (np.isnan(rec) or np.isnan(spec)) else np.nan,
    }


def at_cutoff(y, p, cut: float) -> dict:
    y = np.asarray(y, dtype=bool)
    flag = f32(p) >= np.float32(cut)
    tp = float((flag & y).sum())
    fp = float((flag & ~y).sum())
    fn = float(y.sum()) - tp
    tn = float((~y).sum()) - fp
    out = {"cutoff": float(cut), "flagged": float(flag.mean())}
    out.update(_confusion_metrics(tp, fp, fn, tn))
    return out


def top_name(k: float) -> str:
    return f"top{100 * k:g}"


# Precision, recall and lift among the share k of the riskiest moves
def top_share(y, p, k: float) -> dict:
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


# Scores at the cut-off and among the riskiest moves
def decisions(y, p, cut: float, k: float) -> dict:
    out = at_cutoff(y, p, cut)
    out.update(top_share(y, p, k))
    return out


def operating_points(
    y_valid,
    p_valid,
    y_test,
    p_test,
    top_shares: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05, 0.10),
    precision_targets: tuple[float, ...] = (0.35, 0.50),
) -> pd.DataFrame:
    """Evaluate operating policies on test moves using cutoffs fitted only on valid moves."""
    yt = np.asarray(y_test, dtype=float)
    pt = f32(p_test)
    n = len(yt)
    rate = float(yt.mean()) if n else np.nan
    total_pos = float(yt.sum())
    total_neg = float(n) - total_pos

    cols = [
        "policy", "cutoff", "flagged", "accuracy", "balanced_accuracy",
        "precision", "recall", "f1", "f05", "mcc", "lift",
    ]
    rows = [{
        "policy": "majority_never_blunder",
        "cutoff": 1.0,
        "flagged": 0.0,
        "accuracy": float(1.0 - rate) if n else np.nan,
        "balanced_accuracy": 0.5 if 0 < rate < 1 else np.nan,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "f05": 0.0,
        "mcc": 0.0,
        "lift": np.nan,
    }]

    policies = [
        ("valid_max_f1", best_cutoff(y_valid, p_valid, "f1")),
        ("valid_max_f05", best_cutoff(y_valid, p_valid, "f0.5")),
    ]
    for target_p in precision_targets:
        policies.append((
            f"valid_prec_{int(round(100 * target_p))}",
            best_cutoff(y_valid, p_valid, "precision_target", target_precision=target_p),
        ))

    for name, cut_val in policies:
        m = at_cutoff(yt, pt, cut_val)
        m["policy"] = name
        m["lift"] = float(m["precision"] / rate) if rate else np.nan
        rows.append({c: m[c] for c in cols})

    for k in top_shares:
        t = top_name(k)
        ts = top_share(yt, pt, k)
        m_cnt = max(1, int(round(k * n)))
        edge = float(-np.partition(-pt, m_cnt - 1)[m_cnt - 1])
        prec = float(ts[f"precision_{t}"])
        tp = prec * m_cnt
        fp = float(m_cnt) - tp
        fn = total_pos - tp
        tn = total_neg - fp
        row = {
            "policy": f"top_{100 * k:g}pct",
            "cutoff": edge,
            "flagged": float(m_cnt / n) if n else np.nan,
            "lift": float(ts[f"lift_{t}"]),
        }
        row.update(_confusion_metrics(tp, fp, fn, tn))
        rows.append({c: row[c] for c in cols})

    return pd.DataFrame(rows, columns=cols)
