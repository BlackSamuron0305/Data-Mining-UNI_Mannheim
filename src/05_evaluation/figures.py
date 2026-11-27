"""Validation figures of the baseline and the best run of every method on the common test moves (used by validate.py).

Writes roc_pr, calibration, gains, subgroups_gain and subgroups_pr_auc as .pdf and .png.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.tree import plot_tree

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


def operating_points_metrics(
    y_test,
    p_test,
    cut_f1: float,
    cut_f05: float,
    op_df: pd.DataFrame,
    folder: Path,
) -> None:
    """Threshold sweep (Accuracy Paradox vs Precision/Recall/F1/F0.5) and Top-K% budget comparison."""
    folder.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y_test, dtype=float)
    p = f32(p_test)
    tp, fp, cut = ranking(y, p)
    idx = thin(len(tp))
    tp = tp[idx][::-1]
    fp = fp[idx][::-1]
    cut = cut[idx][::-1]

    total_pos = max(float(y.sum()), 1.0)
    total_neg = float(len(y)) - float(y.sum())
    hi = min(0.6, max(float(np.max(cut)), float(cut_f1) * 1.15, float(cut_f05) * 1.15, 0.25))
    keep = cut <= hi
    c_x = cut[keep]
    tp_k, fp_k = tp[keep], fp[keep]
    prec = tp_k / np.maximum(tp_k + fp_k, 1.0)
    rec = tp_k / total_pos
    acc = (tp_k + (total_neg - fp_k)) / max(len(y), 1)
    f1 = 2.0 * tp_k / np.maximum(2.0 * tp_k + (total_pos - tp_k) + fp_k, 1.0)
    f05 = 1.25 * tp_k / np.maximum(1.25 * tp_k + 0.25 * (total_pos - tp_k) + fp_k, 1.0)

    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH * 1.15, 3.0))
    a.plot(c_x, acc, color=BASE_C, ls="--", lw=1.1, label="accuracy (paradox)")
    a.plot(c_x, prec, color="#0072B2", lw=1.3, label="precision")
    a.plot(c_x, rec, color="#009E73", lw=1.3, label="recall")
    a.plot(c_x, f1, color="#D55E00", lw=1.3, label="F1")
    a.plot(c_x, f05, color="#CC79A7", lw=1.3, label="F0.5")
    a.axvline(cut_f1, color="#D55E00", ls=":", lw=0.9, label=f"max F1 ({cut_f1:.3f})")
    a.axvline(cut_f05, color="#CC79A7", ls="-.", lw=0.9, label=f"max F0.5 ({cut_f05:.3f})")
    a.set(xlabel="probability cut-off", ylabel="score", xlim=(0.0, hi), ylim=(0.0, 1.02))
    a.set_title("operating threshold sweep", pad=18)
    a.legend(loc="center right", fontsize=6.5)

    top_rows = op_df[op_df["policy"].astype(str).str.startswith("top_")].sort_values("flagged")
    if len(top_rows):
        x = np.arange(len(top_rows))
        w = 0.36
        p_vals = top_rows["precision"].to_numpy(dtype=float)
        r_vals = top_rows["recall"].to_numpy(dtype=float)
        lifts = top_rows["lift"].to_numpy(dtype=float)
        labels = [f"{100 * f:.0f}%" for f in top_rows["flagged"]]
        b.bar(x - w / 2, p_vals, width=w, color="#0072B2", label="precision")
        b.bar(x + w / 2, r_vals, width=w, color="#009E73", label="recall")
        for xi, pv, rv, lv in zip(x, p_vals, r_vals, lifts):
            if np.isfinite(lv):
                b.text(xi, max(pv, rv) + 0.02, f"{lv:.1f}x", ha="center", va="bottom", fontsize=6.5, color=INK)
        b.set_xticks(x, labels)
        b.set(
            xlabel="riskiest moves flagged (top-K%)",
            ylabel="precision / recall",
            ylim=(0.0, min(1.0, max(float(np.nanmax(np.r_[p_vals, r_vals])) * 1.32, 0.35))),
        )
        b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=6.5)
    b.set_title("top-K% budget trade-off (lift shown)", pad=18)
    fig.tight_layout(w_pad=1.5)
    save(fig, folder, "operating_points_metrics")


def stratified_vs_balanced(
    res: pd.DataFrame,
    iv: pd.DataFrame | None,
    folder: Path,
    other_label: str = "stratified_balanced",
) -> bool:
    """Compare paired stratified vs balanced test runs on PR-AUC and Log-Loss with 95% CIs."""
    test = res[(res["part"] == "test") & (res["moves_on"] == "common")]
    run_set = set(test["run"])
    prefix = f"{other_label}:"
    paired = []
    for r in test["run"]:
        if r.startswith(prefix):
            base_run = r[len(prefix):]
            if base_run in run_set:
                paired.append(base_run)
    if not paired:
        return False

    t_idx = test.set_index("run")
    paired = sorted(dict.fromkeys(paired), key=lambda r: float(t_idx.loc[r, "log_loss"]), reverse=True)
    iv_map = iv.set_index(["run", "score"]) if (iv is not None and len(iv)) else None

    folder.mkdir(parents=True, exist_ok=True)
    h = max(2.6, 0.42 * len(paired) + 0.9)
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH * 1.15, h), sharey=True)
    y_pos = np.arange(len(paired))

    def _err(run_name: str, metric: str, val: float) -> np.ndarray | None:
        if iv_map is None or (run_name, metric) not in iv_map.index:
            return None
        row = iv_map.loc[(run_name, metric)]
        return np.array([[max(0.0, val - float(row["lo"]))], [max(0.0, float(row["hi"]) - val)]])

    all_ll = [float(t_idx.loc[x, "log_loss"]) for r in paired for x in (r, f"{prefix}{r}")]
    ll_med = float(np.median(all_ll))
    ll_thresh = max(2.5 * ll_med, ll_med + 0.25)
    in_scale: list[float] = []

    for i, r in enumerate(paired):
        rb = f"{prefix}{r}"
        v_s = float(t_idx.loc[r, "pr_auc"])
        v_b = float(t_idx.loc[rb, "pr_auc"])
        a.errorbar(v_s, i + 0.13, xerr=_err(r, "pr_auc", v_s), fmt="o", ms=4, color="#0072B2", capsize=2, lw=1.0)
        a.errorbar(v_b, i - 0.13, xerr=_err(rb, "pr_auc", v_b), fmt="s", ms=4, color="#D55E00", capsize=2, lw=1.0)

        ll_s = float(t_idx.loc[r, "log_loss"])
        ll_b = float(t_idx.loc[rb, "log_loss"])
        if max(ll_s, ll_b) > ll_thresh and any(v <= ll_thresh for v in all_ll):
            b.text(
                0.98,
                i,
                f"off-scale: {ll_s:.3f} / {ll_b:.3f} \u2192",
                transform=b.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=6.2,
                style="italic",
                color=INK,
            )
        else:
            b.errorbar(ll_s, i + 0.13, xerr=_err(r, "log_loss", ll_s), fmt="o", ms=4, color="#0072B2", capsize=2, lw=1.0)
            b.errorbar(ll_b, i - 0.13, xerr=_err(rb, "log_loss", ll_b), fmt="s", ms=4, color="#D55E00", capsize=2, lw=1.0)
            in_scale.extend([ll_s, ll_b])

    if in_scale:
        lo, hi = min(in_scale), max(in_scale)
        pad = max((hi - lo) * 0.14, 0.004)
        b.set_xlim(lo - pad, hi + pad)

    a.set_yticks(y_pos, paired)
    a.set_xlabel("PR-AUC (higher is better)")
    b.set_xlabel("Log loss (lower is better)")
    a.set_title("ranking on shared test games")
    b.set_title("probability calibration & loss")
    handles = [
        Line2D([], [], marker="o", ms=4, ls="", color="#0072B2", label="stratified"),
        Line2D([], [], marker="s", ms=4, ls="", color="#D55E00", label=other_label.replace("_", " ")),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=2)
    fig.tight_layout(w_pad=1.4)
    save(fig, folder, "stratified_vs_balanced")
    return True


def tree_depth4(model_path: Path, folder: Path, max_depth: int = 3) -> bool:
    """Render a saved DecisionTree model up to max_depth as .pdf and .png."""
    if not model_path or not Path(model_path).exists():
        return False
    m = joblib.load(model_path)
    names = getattr(m, "columns", getattr(m, "cols", getattr(m, "features", None)))
    inner = getattr(m, "model", m)
    clf = getattr(inner, "tree", inner)
    if not hasattr(clf, "tree_"):
        return False
    if names is not None and len(names) != clf.n_features_in_:
        names = None

    folder.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(WIDTH * 1.9, 4.6))
    plot_tree(
        clf,
        max_depth=max_depth,
        feature_names=names,
        class_names=["ok", "blunder"],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        label="root",
        fontsize=7,
        ax=ax,
    )
    ax.set_title(
        f"decision tree top {min(max_depth, clf.get_depth())} levels ({clf.get_n_leaves()} leaves in depth-{clf.get_depth()} tree)"
    )
    fig.tight_layout()
    save(fig, folder, "tree_depth4")
    return True


def _compact_count(n: float) -> str:
    n_int = int(round(n))
    if n_int >= 1_000_000:
        return f"{n_int / 1e6:.2f}M"
    if n_int >= 10_000:
        return f"{n_int / 1e3:.0f}k"
    return f"{n_int:,}"


def error_profile(sub_csv: Path, cells_csv: Path, folder: Path) -> bool:
    """Plot subgroup recall vs false-alarm rate and normalized feature profiles across the 4 cells."""
    has_sub = sub_csv.exists()
    has_cells = cells_csv.exists()
    if not has_sub and not has_cells:
        return False

    folder.mkdir(parents=True, exist_ok=True)
    if has_sub and has_cells:
        fig, (ax_sub, ax_cel) = plt.subplots(1, 2, figsize=(WIDTH * 1.40, 3.85), gridspec_kw={"width_ratios": [1.05, 1.15]})
    elif has_sub:
        fig, ax_sub = plt.subplots(figsize=(WIDTH, 3.85))
        ax_cel = None
    else:
        fig, ax_cel = plt.subplots(figsize=(WIDTH, 3.85))
        ax_sub = None

    if ax_sub is not None:
        sub = pd.read_csv(sub_csv)
        filt = sub[sub["blunders"] >= 20] if "blunders" in sub and (sub["blunders"] >= 20).any() else sub
        lo = filt.nsmallest(5, "recall")
        hi = filt.nlargest(5, "recall")
        picked = pd.concat([lo, hi], ignore_index=True).drop_duplicates(subset=["dimension", "group"]).sort_values("recall")
        labels = [f"{DIMENSIONS.get(d, d)}: {g}" for d, g in zip(picked["dimension"], picked["group"])]
        y_pos = np.arange(len(picked))
        h_bar = 0.36
        ax_sub.barh(y_pos + h_bar / 2, 100 * picked["recall"], height=h_bar, color="#0072B2", label="recall (%)")
        ax_sub.barh(y_pos - h_bar / 2, 100 * picked["false_alarm_rate"], height=h_bar, color="#D55E00", label="false-alarm rate (%)")
        ax_sub.set_yticks(y_pos, labels, fontsize=6.5)
        ax_sub.set_xlabel("rate (%)")
        ax_sub.set_title("lowest vs highest recall subgroups", pad=18)
        ax_sub.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=6.5)

    if ax_cel is not None:
        cel = pd.read_csv(cells_csv, index_col=0)
        counts = cel.loc["moves"].to_dict() if "moves" in cel.index else {}
        if "moves" in cel.index:
            cel = cel.drop(index="moves")
        cols = [c for c in ("caught", "missed", "false alarm", "quiet") if c in cel.columns]
        cel = cel[cols].apply(pd.to_numeric, errors="coerce")
        vals = cel.to_numpy(dtype=float)
        row_mean = np.nanmean(vals, axis=1, keepdims=True)
        row_std = np.nanstd(vals, axis=1, keepdims=True)
        z = np.divide(vals - row_mean, np.where(row_std > 1e-12, row_std, 1.0))
        z = np.where(np.isfinite(z), z, 0.0)
        ax_cel.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-1.8, vmax=1.8)
        col_labels = [f"{c}\n({_compact_count(counts[c])})" if c in counts and np.isfinite(counts[c]) else c for c in cols]
        ax_cel.set_xticks(np.arange(len(cols)), col_labels, fontsize=6.5)
        ax_cel.set_yticks(np.arange(len(cel.index)), cel.index, fontsize=6.5)
        ax_cel.grid(visible=False)
        for ri in range(vals.shape[0]):
            for ci in range(vals.shape[1]):
                v = vals[ri, ci]
                if not np.isfinite(v):
                    txt = "-"
                elif abs(v) >= 100:
                    txt = f"{v:.0f}"
                elif abs(v) >= 10:
                    txt = f"{v:.1f}"
                else:
                    txt = f"{v:.2f}"
                ax_cel.text(ci, ri, txt, ha="center", va="center", fontsize=5.8, color="white" if abs(z[ri, ci]) > 1.05 else INK)
        ax_cel.set_title("mean features by error cell (row-standardized)", pad=18)

    fig.tight_layout(w_pad=1.4)
    save(fig, folder, "error_profile")
    return True
