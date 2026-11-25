"""Is the blunder probability the same in every month? September 2026, the month we use, against August 2026.

A model learns the blunder probability of its month. If another month has another one, the model's probabilities
are off by that much there: a monthly bias.

Usage:  uv run python -m src.01_selection.months   (after rates.py of both months)
Output: data/01_selection/comparison/months.csv, months.md; figures from rates_plots.ipynb
"""
import sys

import pandas as pd

from src.config import CFG, COMPARISON_DIR, MONTH_NAMES, SELECTION_ROOT
from src.common.buckets import SPEEDS
from .intervals import add_rates

OUT_DIR = COMPARISON_DIR
MONTHS = [CFG["month"], CFG["compare_month"]]


def label(month: str) -> str:
    """'2026-09' as 'September 2026'."""
    return pd.Timestamp(f"{month}-01").strftime("%B %Y")


def month_table(month: str) -> pd.DataFrame:
    """Moves, blunders and blunder probability of a month, per speed and for all speeds."""
    path = SELECTION_ROOT / MONTH_NAMES[month] / "rates.csv"
    if not path.exists():
        sys.exit(f"{path} is missing: run the cleanup and the blunder rates of {month} first (README.md)")
    rates = pd.read_csv(path)
    t = rates.groupby("speed")[["moves", "blunders"]].sum().reindex(SPEEDS)
    t.loc["all"] = t.sum()
    t = t.rename_axis("speed").reset_index()
    t.insert(0, "month", month)
    return add_rates(t)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.concat([month_table(m) for m in MONTHS], ignore_index=True)
    table.to_csv(OUT_DIR / "months.csv", index=False)

    ours, other = MONTHS
    wide = table.pivot(index="speed", columns="month", values="rate").reindex(SPEEDS + ["all"])
    lo = table.pivot(index="speed", columns="month", values="rate_lo").reindex(wide.index)
    hi = table.pivot(index="speed", columns="month", values="rate_hi").reindex(wide.index)
    lines = [f"# Blunder probability: {label(ours)} against {label(other)}", "",
             f"| speed | {label(ours)} (%) | 95% interval | {label(other)} (%) | 95% interval | difference (points) |",
             "|:--|--:|--:|--:|--:|--:|"]
    for speed, r in wide.iterrows():
        cells = [speed]
        for m in MONTHS:
            cells += [f"{100 * r[m]:.2f}", f"{100 * lo.loc[speed, m]:.2f}-{100 * hi.loc[speed, m]:.2f}"]
        cells.append(f"{100 * (r[other] - r[ours]):+.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "The intervals treat every move as independent, so they are narrower than they should be; the "
              "difference between the months is what matters."]
    (OUT_DIR / "months.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
