"""Which columns the reduction removes, and the final move tables without them.

The decision follows fixed rules on the inspection of the training games (settings: reduction in src/settings.py):
  1. domain    columns that follow from others or that a kept column already covers (a written list per table)
  2. redundant of two columns with |Spearman| >= `redundant`, the one that tells less about blunders
  3. weak      a column that tells less than `weak_share` of what the best column tells about blunders
Columns in `keep` stay whatever the numbers say, each with its written reason; the rules only remove `candidates`.
The balanced run takes the decisions of the stratified run. With reduction.apply = False only the false predictors
are gone (step 2.3/2.4 preprocess) and the final tables are the clean ones.
"""
import sys

import pandas as pd

from src.config import CFG, PREPROCESSING_ROOT, SELECTION
from .paths import PARTS, PRE_DIR
from .tables import write_without

RULES = CFG["reduction"]
COLUMNS = ["column", "rule", "basis", "reason"]


def evidence(col: str, info: pd.Series, corr: pd.DataFrame | None, exclude) -> str:
    """What the inspection measured for a column: its strongest correlation with a column that stays, and how much
    it tells about blunders."""
    parts = []
    if corr is not None and col in corr.columns:
        others = corr[col].drop([c for c in corr.index if c == col or c in exclude], errors="ignore").dropna()
        if len(others):
            partner = others.abs().idxmax()
            parts.append(f"Spearman {others[partner]:.2f} with {partner}")
    parts.append(f"{info[col]:.2f} millibits about blunders")
    return "measured: " + ", ".join(parts)


def decide(inspection_dir, domain: dict, keep: dict, candidates=None, notes: dict | None = None) -> pd.DataFrame:
    """The columns to remove, one row each: the rule, its basis (domain knowledge or inspection) and the reason with
    the measured numbers. candidates: the columns the rules may remove (None: all but those in keep); notes: a few
    words of domain knowledge added to the reason of a column the rules remove."""
    columns = pd.read_csv(inspection_dir / "columns.csv").set_index("column")
    pairs = pd.read_csv(inspection_dir / "pairs.csv")
    corr_path = inspection_dir / "correlation.csv"
    corr = pd.read_csv(corr_path, index_col=0) if corr_path.exists() else None
    info = columns["mi_millibits"]
    notes = notes or {}
    if candidates is None:
        candidates = [c for c in columns.index if c not in keep]
    removed = {}
    for col, why in domain.items():
        if col in columns.index:
            removed[col] = ("domain", "domain knowledge", f"{why}; {evidence(col, info, corr, domain)}")

    # 2. redundant: the pairs from the strongest correlation down
    for _, p in pairs.reindex(pairs["corr"].abs().sort_values(ascending=False).index).iterrows():
        a, b = p["column_a"], p["column_b"]
        if a in removed or b in removed:
            continue
        loser = [c for c in (a, b) if c in candidates]
        if not loser:
            continue
        if len(loser) == 2:
            loser = [a if info[a] < info[b] else b]
        other = b if loser[0] == a else a
        why = "which stays" if other not in candidates else "which tells more about blunders"
        removed[loser[0]] = ("redundant", "inspection",
                             f"Spearman {p['corr']:.2f} with {other}, {why} ({info[loser[0]]:.2f} against "
                             f"{info[other]:.2f} millibits)")

    # 3. weak
    best = info.max()
    for col in candidates:
        if col not in removed and info[col] < RULES["weak_share"] * best:
            removed[col] = ("weak", "inspection",
                            f"tells {info[col]:.2f} millibits about blunders, less than {100 * RULES['weak_share']:g}% "
                            f"of the best column ({info.idxmax()}: {best:.2f})")
    rows = []
    for col, (rule, basis, why) in removed.items():
        if rule != "domain" and col in notes:
            why = f"{why}; {notes[col]}"
        rows.append({"column": col, "rule": rule, "basis": basis, "reason": why})
    return pd.DataFrame(rows, columns=COLUMNS)


def stratified_run_decisions(table: str) -> pd.DataFrame:
    path = PREPROCESSING_ROOT / "stratified" / table / "final" / "removed.csv"
    if not path.exists():
        sys.exit(f"{path} is missing: the balanced run takes the decisions of the stratified run, so reduce that first")
    return pd.read_csv(path)


def reduce_tables(table: str, decisions) -> None:
    """Write data/02_preprocessing/<selection>/<table>/final/ from clean/: the tables without the removed columns,
    removed.csv and kept.csv. decisions: a function that returns (removed, keep) for the stratified run."""
    clean_dir = PRE_DIR / table / "clean"
    final_dir = PRE_DIR / table / "final"
    if not RULES["apply"]:
        removed, keep = pd.DataFrame(columns=COLUMNS), {}
    elif SELECTION != "stratified":
        removed, keep = stratified_run_decisions(table), {}
    else:
        removed, keep = decisions()
    final_dir.mkdir(parents=True, exist_ok=True)
    for part in PARTS:
        n = write_without(clean_dir / f"{part}.parquet", final_dir / f"{part}.parquet", list(removed["column"]))
        print(f"{part}: {n:,} moves, {len(removed)} columns removed")
    removed.to_csv(final_dir / "removed.csv", index=False)
    pd.DataFrame([{"column": c, "reason": r} for c, r in keep.items()], columns=["column", "reason"]) \
        .to_csv(final_dir / "kept.csv", index=False)
    for _, r in removed.iterrows():
        print(f"  - {r['column']:22s} {r['rule']:9s} {r['reason']}")
