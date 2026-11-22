"""Error analysis of one run on its test moves at its cut-off (used by validate.py).

Every move is caught, missed, a false alarm or quiet; writes by_subgroup.csv, cells.csv and summary.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import TARGET
from src.common.metrics import f32
from .groups import DIMENSIONS

CELLS = ["caught", "missed", "false alarm", "quiet"]
FEATURES = ["mover_elo", "elo_diff", "move_no", "clock", "clock_rel", "eval_wp", "own_last_loss", "own_loss_recent",
            "own_mistakes", "opp_last_loss", "legal_moves", "in_check", "material_total"]


def cells(y: np.ndarray, flag: np.ndarray) -> np.ndarray:
    return np.select([flag & y, ~flag & y, flag & ~y], CELLS[:3], CELLS[3])


def ratio(a, b):
    """a / b, or NaN if b is 0."""
    if b:
        return a / b
    return np.nan


def by_subgroup(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    rows = []
    for dim in dims:
        t = pd.crosstab(df[dim], df["cell"]).reindex(columns=CELLS, fill_value=0)
        t = t[t.sum(axis=1) > 0]
        for group, c in t.iterrows():
            b = c["caught"] + c["missed"]
            alarms = c["false alarm"]
            row = {"dimension": dim, "group": str(group), "moves": int(c.sum()), "blunders": int(b)}
            for k in CELLS:
                row[k.replace(" ", "_")] = int(c[k])
            row["recall"] = ratio(c["caught"], b)
            row["precision"] = ratio(c["caught"], c["caught"] + alarms)
            row["false_alarm_rate"] = ratio(alarms, alarms + c["quiet"])
            row["share_of_missed"] = c["missed"] / max(t["missed"].sum(), 1)
            row["share_of_false_alarms"] = alarms / max(t["false alarm"].sum(), 1)
            rows.append(row)
    return pd.DataFrame(rows)


def means(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of the features in each cell; the last row is the number of moves."""
    cols = ["p"] + [c for c in FEATURES if c in df]
    t = df.groupby("cell", observed=False)[cols].mean().reindex(CELLS).T
    t.loc["moves"] = df["cell"].value_counts().reindex(CELLS).fillna(0).to_numpy()
    return t


def md_table(t: pd.DataFrame, index: str, fmt: str = ".3f") -> list[str]:
    cols = list(t.columns)
    whole = [t[c].dtype.kind in "iu" for c in cols]
    out = ["| " + index + " | " + " | ".join(map(str, cols)) + " |", "|:--|" + "--:|" * len(cols)]
    for i, r in zip(t.index, t.itertuples(index=False)):
        cells_text = []
        for v, w in zip(r, whole):
            if pd.isna(v):
                cells_text.append("")
            elif w:
                cells_text.append(f"{v:,}")
            else:
                cells_text.append(f"{v:{fmt}}")
        out.append(f"| {i} | " + " | ".join(cells_text) + " |")
    return out


def analyse(run: str, df: pd.DataFrame, cut: float, folder: Path) -> list[str]:
    """Writes the error analysis of a run's test moves and returns the lines of the summary."""
    folder.mkdir(parents=True, exist_ok=True)
    y = df[TARGET].to_numpy(dtype=bool)
    flag = f32(df["p"]) >= np.float32(cut)
    df = df.assign(cell=pd.Categorical(cells(y, flag), CELLS))
    n = df["cell"].value_counts().reindex(CELLS).fillna(0).astype(int)
    rec = n["caught"] / max(n["caught"] + n["missed"], 1)
    prec = n["caught"] / max(n["caught"] + n["false alarm"], 1)
    lines = [f"# Error analysis of {run}", "",
             f"Test moves of the run: {len(df):,}. Cut-off {cut:.4f} (highest F1 on its valid moves): "
             f"{flag.mean():.1%} of the moves flagged, {rec:.1%} of the blunders caught, {prec:.1%} of the flagged "
             "moves are blunders.", "",
             f"Caught {n['caught']:,}, missed {n['missed']:,}, false alarms {n['false alarm']:,}, quiet {n['quiet']:,}.",
             ""]

    dims = [d for d in DIMENSIONS if d in df]
    if dims:
        sub = by_subgroup(df, dims)
        sub.to_csv(folder / "by_subgroup.csv", index=False)
        lines += ["## Where it misses blunders", "",
                  "Recall per subgroup (share of its blunders caught) and false-alarm rate (share of its other "
                  "moves flagged); lowest recall first, subgroups with at least 20 blunders.", ""]
        t = sub[sub["blunders"] >= 20].sort_values("recall").head(8)
        t = t.assign(subgroup=t["dimension"].map(DIMENSIONS) + ": " + t["group"]).set_index("subgroup")
        lines += md_table(t[["moves", "blunders", "recall", "precision", "false_alarm_rate"]], "subgroup") + [""]

    m = means(df)
    m.to_csv(folder / "cells.csv")
    lines += ["## The four cells", "", "Mean of the main features in each cell (p = the prediction).", ""]
    names = {}
    for c in CELLS:
        names[c] = f"{c} ({int(m.loc['moves', c]):,})"
    shown = m.drop(index="moves").rename(columns=names)
    lines += md_table(shown, "feature (moves)") + [""]
    if dims:
        lines += ["All subgroups: by_subgroup.csv.", ""]
    (folder / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines
