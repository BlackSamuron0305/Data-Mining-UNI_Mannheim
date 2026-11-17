"""Bucket rates step 3 - figures and a table of the results of steps 1 and 2.

Usage:  uv run python -m src.preprocessing.bucket_rates.plots       (after rates.py; selection_check.py adds the
                                                                    training and test part)

Without the outputs of selection_check.py only the month is drawn.

Outputs (data/bucket_rates/figures/; every figure as .pdf for the paper and as .png, 14.5 cm wide = the text width)
rate_by_time_left    blunder probability by time left, one panel per speed (all Elo bands): the month with its 95%
                     interval, the training and test part with theirs
rate_heatmap         blunder probability of every bucket of the month: Elo of the mover x time left, one panel per
                     speed; a dot marks a bucket with fewer than `min_games` games
selection_z          z of every bucket in the training and test part against the standard normal, and chi2 per bucket
                     of the selection against the random stratified draws
results_table        the main numbers as a table (also results_table.csv, and printed to the log)
logs/plots.log
"""
from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from src.config import BUCKET_RATES_DIR, CFG, LOGS_SUB, MONTH, setup_log
from src.preprocessing.bucket_rates.rates import time_order
from src.preprocessing.selection.census import SPEEDS

log = logging.getLogger("bucket_rate_plots")
FIG_DIR = BUCKET_RATES_DIR / "figures"
PARTS = ("train", "test")
PART_NAME = {"train": "training", "test": "test"}
WIDTH = 14.5 / 2.54            # inches: the text width of the paper
STEP_MIN = CFG["selection"]["window"] / 60
MIN_GAMES = CFG["selection"]["min_games"]

# The month is the reference and stays neutral; the two parts carry the only hues (Okabe-Ito blue and vermilion,
# checked for colour blindness). Text stays in ink colours, never in a series colour.
MONTH_C, PART_C = "#3d3d3a", {"train": "#0072B2", "test": "#D55E00"}
INK, MUTED, GRID, EMPTY = "#1f1f1e", "#6b6b66", "#e4e4e0", "#f0f0ec"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .5, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": .02, "pdf.fonttype": 42,
})


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    log.info("figure %s", FIG_DIR / f"{name}.pdf")


def minutes(labels: pd.Series) -> np.ndarray:
    """Time left at the start of every step, in minutes."""
    return time_order(labels).to_numpy() / 60


def interval(d: pd.DataFrame, suffix: str = "") -> tuple[np.ndarray, list[np.ndarray]]:
    """Rate and its 95% interval as error bar lengths, in %."""
    r, lo, hi = (100 * d[f"{c}{suffix}"].to_numpy() for c in ("rate", "rate_lo", "rate_hi"))
    return r, [r - lo, hi - r]


# ---------------------------------------------------------------------------------------------------- figures
def rate_by_time_left(tl: pd.DataFrame, parts: tuple[str, ...]) -> None:
    fig, axes = plt.subplots(1, len(SPEEDS), figsize=(WIDTH, 2.15), sharey=True)
    for ax, sp in zip(axes, SPEEDS):
        d = tl[(tl["speed"] == sp) & (tl["moves"] > 0)]
        x = minutes(d["time_left"])
        ax.fill_between(x, 100 * d["rate_lo"], 100 * d["rate_hi"], color=MONTH_C, alpha=.15, lw=0)
        ax.plot(x, 100 * d["rate"], color=MONTH_C, lw=1.5)
        for i, q in enumerate(parts):
            r, err = interval(d, f"_{q}")
            ax.errorbar(x + (i - .5) * .3 * STEP_MIN, r, yerr=err, fmt="o", ms=2.6, lw=.7, capsize=0,
                        color=PART_C[q], mec="white", mew=.4)
        ax.set_title(sp)
        ax.set_xlabel("time left (min)")
        ax.set_xlim(left=0)
    axes[0].set_ylabel("blunder probability (%)")
    axes[0].set_ylim(bottom=0)
    handles = [Line2D([], [], color=MONTH_C, lw=1.5, label=f"month {MONTH} (95% interval)")]
    handles += [Line2D([], [], color=PART_C[q], marker="o", ms=3.5, ls="", label=f"{PART_NAME[q]} part")
                for q in parts]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .99), ncol=len(handles))
    fig.tight_layout(w_pad=.6)
    save(fig, "rate_by_time_left")


def rate_heatmap(rates: pd.DataFrame) -> None:
    elos = list(dict.fromkeys(rates["mover_elo"]))
    vmax = float(np.nanpercentile(100 * rates["rate"], 99))
    norm, cmap = Normalize(0, vmax), plt.get_cmap("Blues").copy()
    cmap.set_bad(EMPTY)
    fig, axes = plt.subplots(1, len(SPEEDS), figsize=(WIDTH, 2.3), sharey=True)
    for ax, sp in zip(axes, SPEEDS):
        d = rates[rates["speed"] == sp]
        steps = list(dict.fromkeys(d["time_left"]))
        grid = d.pivot(index="mover_elo", columns="time_left", values="rate").reindex(index=elos, columns=steps)
        few = d.pivot(index="mover_elo", columns="time_left", values="few_games").reindex(index=elos, columns=steps)
        im = ax.imshow(100 * grid.to_numpy(dtype=float), aspect="auto", origin="lower", cmap=cmap, norm=norm,
                       interpolation="nearest", extent=(0, len(steps) * STEP_MIN, -.5, len(elos) - .5))
        yy, xx = np.nonzero(few.fillna(False).to_numpy(dtype=bool))
        ax.scatter((xx + .5) * STEP_MIN, yy, s=1.5, color=MUTED, lw=0)
        ax.set_title(sp)
        ax.set_xlabel("time left (min)")
        ax.grid(False)
        ax.spines[["left", "bottom"]].set_visible(False)
    axes[0].set_yticks(range(len(elos)), elos)
    axes[0].set_ylabel("Elo of the mover")
    cb = fig.colorbar(im, ax=axes, fraction=.025, pad=.015, extend="max")
    cb.set_label("blunder probability (%)")
    cb.outline.set_visible(False)
    key = [Line2D([], [], marker="o", ms=2, color=MUTED, ls="", label=f"fewer than {MIN_GAMES} games"),
           Line2D([], [], marker="s", ms=6, color=EMPTY, mec=GRID, ls="", label="no move")]
    fig.legend(handles=key, loc="lower right", bbox_to_anchor=(.9, .99), ncol=2, handletextpad=.3)
    save(fig, "rate_heatmap")


def selection_z(sc: pd.DataFrame, check: dict) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.2), gridspec_kw={"width_ratios": [1.7, 1]})
    edges = np.linspace(-5, 5, 41)
    for q in PARTS:
        z = np.clip(sc[f"z_{q}"].dropna().to_numpy(), -4.99, 4.99)
        a.hist(z, edges, density=True, histtype="step", lw=1.5, color=PART_C[q], label=f"{PART_NAME[q]} part")
    xs = np.linspace(-5, 5, 300)
    a.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color=MONTH_C, lw=1, ls="--", label="standard normal")
    a.set_xlabel("z of a bucket (values beyond $\\pm$5 at the edge)")
    a.set_ylabel("density")
    a.legend(loc="upper left")

    draws = check.get("random_draws", {})
    for i, q in enumerate(PARTS):
        vals = np.array([d["chi2_per_df"] for d in draws.get(q, {}).get("all", [])])
        if len(vals):
            jitter = np.random.default_rng(i).uniform(-.12, .12, len(vals))
            b.scatter(i + jitter, vals, s=9, color=MUTED, alpha=.55, lw=0,
                      label=f"{len(vals)} random draws" if i == 0 else None)
        b.scatter(i, check["selection"][q]["chi2_per_df"], marker="D", s=32, color=PART_C[q], ec="white", lw=.6,
                  zorder=3, label="the selection" if i == 0 else None)
    b.axhline(1, color=MUTED, lw=.8, ls=":")
    b.set_xticks(range(len(PARTS)), [PART_NAME[q] for q in PARTS])
    b.set_xlim(-.6, len(PARTS) - .4)
    b.set_ylabel("$\\chi^2$ per bucket")
    b.grid(axis="x", visible=False)
    b.legend(loc="best", handletextpad=.2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "selection_z")


# ------------------------------------------------------------------------------------------------------ table
def results(rates: pd.DataFrame, sc: pd.DataFrame | None, check: dict | None) -> pd.DataFrame:
    """The main numbers: one row per figure, columns month, training and test part ('' where it does not apply)."""
    parts = PARTS if sc is not None else ()
    cols = ["month"] + [PART_NAME[q] for q in parts]
    moves = {"month": rates["moves"].sum(), **{PART_NAME[q]: sc[f"moves_{q}"].sum() for q in parts}}
    blunders = {"month": rates["blunders"].sum(), **{PART_NAME[q]: sc[f"blunders_{q}"].sum() for q in parts}}
    rows = {"moves": {c: f"{int(moves[c]):,}" for c in cols},
            "blunder probability (%)": {c: f"{100 * blunders[c] / moves[c]:.3f}" for c in cols}}
    src = sc if sc is not None else rates
    for sp in SPEEDS:
        d = src[src["speed"] == sp]
        rows[f"  {sp} (%)"] = {"month": f"{100 * d['blunders'].sum() / d['moves'].sum():.3f}",
                               **{PART_NAME[q]: f"{100 * d[f'blunders_{q}'].sum() / d[f'moves_{q}'].sum():.3f}"
                                  for q in parts}}
    rows["buckets with a move"] = {"month": f"{int((rates['moves'] > 0).sum())} of {len(rates)}"}
    rows["buckets with few games"] = {"month": f"{int(rates['few_games'].sum())}"}
    if check is not None:
        s, rd = check["selection"], check.get("random_draws", {})
        for q in parts:
            c = PART_NAME[q]
            rows["buckets with a move"][c] = f"{s[q]['df']}"
            rows.setdefault("balanced", {})[c] = "yes" if check["selection_balanced"][q] else "no"
            rows.setdefault("chi2 per bucket", {})[c] = f"{s[q]['chi2_per_df']:.2f}"
            rows.setdefault("buckets with |z| > 2 (%)", {})[c] = f"{100 * s[q]['share_abs_z_over_2']:.1f}"
            rows.setdefault("mean difference to the month (points)", {})[c] = f"{s[q]['mean_abs_diff_pp']:.3f}"
            if q in rd:
                n = check["draws"]
                lo, hi = np.quantile([d["chi2_per_df"] for d in rd[q]["all"]], [.05, .95])
                rows.setdefault(f"chi2 per bucket, {n} random draws (5-95%)", {})[c] = f"{lo:.2f}-{hi:.2f}"
                rows.setdefault("random draws with a lower chi2 (%)", {})[c] = \
                    f"{100 * rd[q]['share_draws_with_lower_chi2']:.0f}"
                rows.setdefault("mean difference, random draws (points)", {})[c] = \
                    f"{rd[q]['mean_abs_diff_pp_mean']:.3f}"
    return pd.DataFrame.from_dict(rows, orient="index", columns=cols).fillna("")


def results_table(table: pd.DataFrame) -> None:
    n = len(table)
    fig, ax = plt.subplots(figsize=(WIDTH, .2 * (n + 1) + .1))
    ax.axis("off")
    labels = [r.replace("chi2", "$\\chi^2$").replace("|z|", "$|z|$") for r in table.index]
    widths = [.52] + [(1 - .52) / len(table.columns)] * len(table.columns)
    tbl = ax.table(cellText=table.to_numpy(), rowLabels=None, colLabels=list(table.columns), cellLoc="right",
                   colLoc="right", colWidths=widths[1:], bbox=(widths[0], 0, 1 - widths[0], 1))
    lab = ax.table(cellText=[[r] for r in labels], colLabels=[""], cellLoc="left", colLoc="left",
                   colWidths=[1], bbox=(0, 0, widths[0], 1))
    for t in (tbl, lab):
        t.auto_set_font_size(False)
        t.set_fontsize(8)
        for (r, _), cell in t.get_celld().items():
            cell.set_edgecolor(INK)
            cell.set_linewidth(.8)
            cell.visible_edges = "B" if r in (0, n) else ""
            if r == 0:
                cell.set_text_props(weight="bold")
            elif labels[r - 1].startswith("  "):
                cell.set_text_props(color=MUTED)
    ax.axhline(1, color=INK, lw=.8)            # top rule
    save(fig, "results_table")


# -------------------------------------------------------------------------------------------------------- run
def run() -> None:
    rates_path = BUCKET_RATES_DIR / "rates.csv"
    if not rates_path.exists():
        raise FileNotFoundError(f"{rates_path} is missing: run uv run python -m src.preprocessing.bucket_rates.rates "
                                "first")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (BUCKET_RATES_DIR / LOGS_SUB).mkdir(parents=True, exist_ok=True)
    setup_log(BUCKET_RATES_DIR / LOGS_SUB / "plots.log")
    logging.getLogger("fontTools").setLevel(logging.WARNING)      # it logs every font it embeds in a PDF
    rates = pd.read_csv(rates_path, index_col="bucket")

    sel_paths = [BUCKET_RATES_DIR / f for f in ("selection_check.csv", "selection_by_time_left.csv",
                                                "selection_check.json")]
    if all(p.exists() for p in sel_paths):
        sc = pd.read_csv(sel_paths[0], index_col="bucket")
        tl = pd.read_csv(sel_paths[1])
        check = json.loads(sel_paths[2].read_text(encoding="utf-8"))
        parts = PARTS
    else:
        log.info("no check of the selection yet (selection_check.py): the figures show the month only")
        sc, check, parts = None, None, ()
        tl = pd.read_csv(BUCKET_RATES_DIR / "by_time_left.csv")

    rate_by_time_left(tl, parts)
    rate_heatmap(rates)
    if check is not None:
        selection_z(sc, check)
    table = results(rates, sc, check)
    table.to_csv(BUCKET_RATES_DIR / "figures" / "results_table.csv", index_label="")
    results_table(table)
    log.info("results\n%s", table.to_string())


if __name__ == "__main__":
    run()
