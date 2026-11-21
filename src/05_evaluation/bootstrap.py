"""95% intervals of log loss, log-loss gain and PR-AUC, from resampling whole games (used by validate.py).

Every run sees the same resamples, so the intervals of two runs can be compared.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score

from src.config import SEED
from src.shared.metrics import f32, move_loss

CHUNK = 100


def draws(n_games: int, n: int, seed: int = SEED):
    """Blocks of resamples: how often every game is drawn (resamples x games)."""
    rng = np.random.default_rng(seed)
    for start in range(0, n, CHUNK):
        c = min(CHUNK, n - start)
        rows = []
        for _ in range(c):
            rows.append(np.bincount(rng.integers(0, n_games, n_games), minlength=n_games))
        yield np.stack(rows).astype(np.float64)


def steps(p: np.ndarray, bins: int) -> np.ndarray:
    """Step of every prediction: its distinct value, or one of `bins` quantile steps if there are more."""
    u = np.unique(p)
    if len(u) <= bins:
        return np.searchsorted(u, p)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.searchsorted(edges, p, side="right")


def binned_ap(pos: np.ndarray, moves: np.ndarray) -> np.ndarray:
    """Average precision from blunders and moves per step (steps x resamples, lowest step first)."""
    tp = np.cumsum(pos[::-1], axis=0)
    flagged = np.cumsum(moves[::-1], axis=0)
    prec = np.divide(tp, flagged, out=np.zeros_like(tp), where=flagged > 0)
    return (pos[::-1] * prec).sum(axis=0) / tp[-1]


def per_game(codes: np.ndarray, values: np.ndarray, n_games: int) -> np.ndarray:
    return np.bincount(codes, weights=values, minlength=n_games)


def intervals(y: np.ndarray, games: np.ndarray, preds: dict[str, np.ndarray], base: str | None, n: int,
              bins: int) -> pd.DataFrame:
    """One row per run and score: the estimate on all moves and its 95% interval."""
    codes, uniq = pd.factorize(games)
    g = len(uniq)
    y = np.asarray(y, dtype=float)
    moves_g = per_game(codes, np.ones(len(y)), g)
    loss_g = {}
    for r, p in preds.items():
        loss_g[r] = per_game(codes, move_loss(y, p), g)

    rows = []
    for run, p in preds.items():
        s = steps(f32(p), bins)
        k = int(s.max()) + 1
        pos_m = sparse.csr_matrix((y, (s, codes)), shape=(k, g))
        all_m = sparse.csr_matrix((np.ones(len(y)), (s, codes)), shape=(k, g))
        exact = float(average_precision_score(y, p))
        # shift the binned PR-AUC so that it matches the exact one on all moves
        shift = exact - binned_ap(np.asarray(pos_m.sum(axis=1)), np.asarray(all_m.sum(axis=1)))[0]

        got = {"log_loss": [], "pr_auc": [], "log_loss_gain": []}
        for w in draws(g, n):
            nm = w @ moves_g
            got["log_loss"].append(w @ loss_g[run] / nm)
            if base is not None:
                got["log_loss_gain"].append(w @ (loss_g[base] - loss_g[run]) / nm)
            wt = np.ascontiguousarray(w.T)
            got["pr_auc"].append(binned_ap(pos_m @ wt, all_m @ wt) + shift)

        est = {"log_loss": loss_g[run].sum() / len(y), "pr_auc": exact}
        if base is not None:
            est["log_loss_gain"] = (loss_g[base].sum() - loss_g[run].sum()) / len(y)
        else:
            est["log_loss_gain"] = np.nan
        for score, vals in got.items():
            if vals:
                lo, hi = np.nanpercentile(np.concatenate(vals), [2.5, 97.5])
                rows.append({"run": run, "score": score, "estimate": est[score], "lo": lo, "hi": hi})
    return pd.DataFrame(rows)
