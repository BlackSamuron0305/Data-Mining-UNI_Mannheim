"""Validation figures of the baseline and the best run of every method on the common test moves (used by validate.py).

Writes roc_pr, calibration, gains, subgroups_gain and subgroups_pr_auc as .pdf and .png.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.common.metrics import calibration, f32, ranking
from .groups import DIMENSIONS

WIDTH = 14.5 / 2.54
BASE_C = "#3d3d3a"
RUN_C = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7"]
INK, MUTED, GRID = "#1f1f1e", "#6b6b66", "#e4e4e0"
POINTS = 1500

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .5, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": .02, "pdf.fonttype": 42,
})
logging.getLogger("fontTools").setLevel(logging.WARNING)


def styles(runs: list[str], base: str | None) -> dict[str, dict]:
    out, i = {}, 0
    for r in runs:
        if r == base:
            out[r] = {"color": BASE_C, "ls": "--", "lw": 1.1}
        else:
            out[r] = {"color": RUN_C[i % len(RUN_C)], "ls": "-", "lw": 1.3}
            i += 1
    return out


def save(fig, folder: Path, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def legend(fig, st: dict, extra: list | None = None) -> None:
    h = []
    for r, s in st.items():
        h.append(Line2D([], [], label=r, **s))
    if extra:
        h += extra
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(.5, .99), ncol=min(3, len(h)))


def thin(n: int) -> np.ndarray:
    """Indices of about 2 x POINTS points of a curve of n points, dense at its start."""
    dense_start = np.geomspace(1, n, POINTS).astype(int) - 1
    even = np.linspace(0, n - 1, POINTS).astype(int)
    return np.unique(np.r_[dense_start, even])


def curves(y: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    tp, fp, cut = ranking(y, p)
    i = thin(len(tp))
    tp = tp[i]
    fp = fp[i]
    cut = cut[i]
    return {"recall": tp / tp[-1], "fpr": fp / fp[-1], "precision": tp / (tp + fp),
            "flagged": (tp + fp) / len(y), "cut": cut}


def roc_pr(y, preds: dict, st: dict, cuts: dict, folder: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.9))
    for r, p in preds.items():
        c = curves(y, p)
        a.plot(np.r_[0, c["fpr"]], np.r_[0, c["recall"]], **st[r])
        b.plot(c["recall"], c["precision"], **st[r])
        flag = f32(p) >= np.float32(cuts[r])
        rec = (flag & (y == 1)).sum() / (y == 1).sum()
        fpr = (flag & (y == 0)).sum() / (y == 0).sum()
        prec = (flag & (y == 1)).sum() / max(flag.sum(), 1)
        a.plot(fpr, rec, "o", ms=3, color=st[r]["color"], mec="white", mew=.4)
        b.plot(rec, prec, "o", ms=3, color=st[r]["color"], mec="white", mew=.4)
    a.plot([0, 1], [0, 1], color=MUTED, lw=.7, ls=":")
    b.axhline(y.mean(), color=MUTED, lw=.7, ls=":")
    a.set(xlabel="other moves flagged (share)", ylabel="blunders flagged (recall)", xlim=(0, 1), ylim=(0, 1))
    b.set(xlabel="blunders flagged (recall)", ylabel="flagged moves that are blunders (precision)", xlim=(0, 1),
          ylim=(0, None))
    a.set_title("ROC")
    b.set_title("precision and recall")
    legend(fig, st, [Line2D([], [], marker="o", ms=3, ls="", color=MUTED, label="cut-off from valid"),
                     Line2D([], [], color=MUTED, lw=.7, ls=":", label="guessing")])
    fig.tight_layout(w_pad=1.5)
    save(fig, folder, "roc_pr")


def calibration_fig(y, preds: dict, st: dict, folder: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.9))
    lo = 1e-4
    edges = np.geomspace(lo, 1, 61)
    for r, p in preds.items():
        c = calibration(y, np.asarray(p, dtype=float))
        a.plot(np.clip(c["mean_p"], lo, 1), np.clip(c["rate"], lo, 1), marker="o", ms=2.2, **st[r])
        share = np.histogram(np.clip(p, lo, 1), edges)[0] / len(p)
        b.step(edges[:-1], 100 * share, where="post", **st[r])
    a.plot([lo, 1], [lo, 1], color=MUTED, lw=.7, ls=":")
    a.set(xscale="log", yscale="log", xlim=(lo, 1), ylim=(lo, 1), xlabel="mean predicted probability",
          ylabel="observed blunder rate")
    a.set_title("calibration (20 bins of equal size; at the edge: 0)")
    b.set(xscale="log", xlim=(lo, 1), xlabel="predicted probability", ylabel="moves (%)")
    b.set_title("spread of the predictions")
    legend(fig, st, [Line2D([], [], color=MUTED, lw=.7, ls=":", label="perfect calibration")])
    fig.tight_layout(w_pad=1.5)
    save(fig, folder, "calibration")


def gains(y, preds: dict, st: dict, k: float, folder: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.9))
    for r, p in preds.items():
        c = curves(y, p)
        a.plot(np.r_[0, c["flagged"]], np.r_[0, c["recall"]], **st[r])
        keep = c["flagged"] >= 1e-3
        b.plot(c["flagged"][keep], c["precision"][keep] / y.mean(), **st[r])
    rate = y.mean()
    a.plot([0, 1], [0, 1], color=MUTED, lw=.7, ls=":")
    a.plot([0, rate, 1], [0, 1, 1], color=MUTED, lw=.7, ls="-.")
    b.axhline(1, color=MUTED, lw=.7, ls=":")
    for ax in (a, b):
        ax.axvline(k, color=GRID, lw=1.2, zorder=0)
    a.set(xlabel="moves flagged, riskiest first (share)", ylabel="blunders caught (share)", xlim=(0, 1), ylim=(0, 1))
    b.set(xscale="log", xlim=(1e-3, 1), ylim=(0, None), xlabel="moves flagged, riskiest first (share)",
          ylabel="lift (blunder rate among them / overall)")
    a.set_title("cumulative gains")
    b.set_title("lift")
    legend(fig, st, [Line2D([], [], color=MUTED, lw=.7, ls=":", label="guessing"),
                     Line2D([], [], color=MUTED, lw=.7, ls="-.", label="perfect")])
    fig.tight_layout(w_pad=1.5)
    save(fig, folder, "gains")


def subgroups(sub: pd.DataFrame, value: str, ylabel: str, st: dict, folder: Path, name: str,
              zero_line: float | None = None) -> None:
    dims = [d for d in DIMENSIONS if d in set(sub["dimension"])]
    if not dims:
        return
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.6))
    axes = axes.ravel()
    for ax, dim in zip(axes, dims):
        d = sub[sub["dimension"] == dim]
        levels = list(d["group"].unique())
        for r, s in st.items():
            v = d[d["run"] == r].set_index("group")[value].reindex(levels)
            if v.notna().any():
                ax.plot(range(len(levels)), v.to_numpy(), marker="o", ms=2.5, **s)
        if zero_line is not None:
            ax.axhline(zero_line, color=MUTED, lw=.7, ls=":")
        ax.set_xticks(range(len(levels)), levels, rotation=35, ha="right")
        ax.set_title(DIMENSIONS[dim])
        ax.grid(axis="x", visible=False)
    rows_used = (len(dims) + 2) // 3
    for row in range(rows_used):
        axes[3 * row].set_ylabel(ylabel)
    for ax in axes[len(dims):]:
        ax.axis("off")
    legend(fig, st)
    fig.tight_layout(h_pad=.8, w_pad=1)
    save(fig, folder, name)


def draw(y: np.ndarray, preds: dict, base: str | None, cuts: dict, sub: pd.DataFrame | None, k: float,
         folder: Path) -> list[str]:
    """Every figure; returns their names."""
    folder.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=float)
    st = styles(list(preds), base)
    roc_pr(y, preds, st, cuts, folder)
    calibration_fig(y, preds, st, folder)
    gains(y, preds, st, k, folder)
    names = ["roc_pr", "calibration", "gains"]
    if sub is not None and len(sub):
        if base is not None:
            gst = {}
            for r, s in st.items():
                if r != base:
                    gst[r] = s
            subgroups(sub, "gain_pct", "log-loss gain over the baseline (%)", gst, folder, "subgroups_gain", 0)
            names.append("subgroups_gain")
        subgroups(sub, "pr_auc_over_rate", "PR-AUC / blunder rate", st, folder, "subgroups_pr_auc", 1)
        names.append("subgroups_pr_auc")
    return names
