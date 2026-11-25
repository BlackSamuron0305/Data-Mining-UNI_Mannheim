"""Step 2.5: which columns we remove, and the final move tables without them.

The decision follows fixed rules on the inspection of the training games (settings: reduction in src/settings.py):
  1. domain    columns that follow from others or that a kept column already covers (the written lists below)
  2. redundant of two columns with |Spearman| >= `redundant`, the one that tells less about blunders
  3. weak      a column that tells less than `weak_share` of what the best column tells about blunders
Columns in KEEP stay whatever the numbers say, each with its written reason.

The rules run twice. First on the columns without Stockfish alone, as if Stockfish did not exist; then on the
Stockfish columns, which may only remove Stockfish columns. So the table without its Stockfish columns (sf_*) is
exactly the table a model without Stockfish gets, and the models can show what Stockfish adds.

The balanced run takes the decisions of the stratified run. With reduction.apply = False only the false predictors
are gone (preprocess.py).

Usage:  uv run python -m src.02_preprocessing.reduce   (after inspection.py; the balanced run: after the stratified run)
Output: data/02_preprocessing/<selection>/train.parquet, test.parquet, removed.csv, kept.csv
"""
import sys

import pandas as pd
import pyarrow.parquet as pq

from src.config import CFG, PREPROCESSING_DIR, PREPROCESSING_ROOT, PREPROCESSING_WORK, SELECTION, STOCKFISH_PREFIX

RULES = CFG["reduction"]
COLUMNS = ["column", "rule", "basis", "reason"]
PARTS = ("train", "test")
INSPECTION = PREPROCESSING_ROOT / "stratified" / "inspection"

# Removed whatever the numbers say: they follow from columns that stay, or a column that stays covers them
DOMAIN = {
    "eval_cp": "the same evaluation as eval_wp in other units; eval_wp stays because it lies between 0 and 1",
    "material_own": "follows from material_total and material_diff",
    "material_opp": "follows from material_total and material_diff",
    "opp_elo": "follows from mover_elo and elo_diff",
    "opp_clock": "follows from clock and clock_diff",
    "base_time": "speed follows from base_time and increment, and clock_rel already uses both; speed stays",
    "increment": "speed follows from base_time and increment, and clock_rel already uses both; speed stays",
}
DOMAIN_STOCKFISH = {
    "sf_best_move": "the best move as text (e.g. e2e4): a name, not a number a model can use",
    "sf_n_scored": "the number of legal moves Stockfish scored: the same as legal_moves",
}

# Kept whatever the numbers say
KEEP = {
    "speed": "the buckets and the basic feature set are built on speed, mover_elo and clock",
    "mover_elo": "the buckets and the basic feature set are built on speed, mover_elo and clock",
    "clock": "the buckets and the basic feature set are built on speed, mover_elo and clock",
    "move_no": "repeats material_total in part (Spearman about -0.9), but counts the time in the game, "
               "material_total how simplified the position is",
    "material_total": "repeats move_no in part (Spearman about -0.9), but tells how simplified the position is, "
                      "move_no the time in the game",
}

# Why a column that the rules remove tells nothing, in words (added to its reason in removed.csv)
NOTES = {
    "hour": "the hour is in UTC, not the player's own time, so it cannot show tiredness",
    "white": "White and Black blunder at the same rate",
    "event_type": "lobby, arena and swiss games have almost the same blunder rate (inspection/categories.csv)",
    "elo_diff": "the rating gap says little about one move once the mover's own rating is known",
    "in_check": "most moves are not in check, and legal_moves already counts how few moves a check leaves",
}


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


def decide(inspection_dir, domain: dict, keep: dict, candidates=None, notes: dict | None = None,
           only=None) -> pd.DataFrame:
    """The columns to remove, one row each: the rule, its basis (domain knowledge or inspection) and the reason with
    the measured numbers. candidates: the columns the rules may remove (None: all but those in keep); notes: a few
    words of domain knowledge added to the reason of a column the rules remove; only: look at these columns alone
    (None: all)."""
    columns = pd.read_csv(inspection_dir / "columns.csv").set_index("column")
    pairs = pd.read_csv(inspection_dir / "pairs.csv")
    corr_path = inspection_dir / "correlation.csv"
    corr = pd.read_csv(corr_path, index_col=0) if corr_path.exists() else None
    if only is not None:
        columns = columns.loc[[c for c in columns.index if c in only]]
        pairs = pairs[pairs["column_a"].isin(only) & pairs["column_b"].isin(only)]
        if corr is not None:
            inside = [c for c in corr.columns if c in only]
            corr = corr.loc[inside, inside]
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


def decisions() -> pd.DataFrame:
    """The removed columns of the stratified run: first the rules on the columns without Stockfish, then on the
    Stockfish columns, with the columns removed in the first round counted as removed from the start."""
    names = pd.read_csv(INSPECTION / "columns.csv")["column"]
    stockfish = [c for c in names if c.startswith(STOCKFISH_PREFIX)]
    plain = [c for c in names if c not in stockfish]
    without = decide(INSPECTION, DOMAIN, KEEP, notes=NOTES, only=plain).set_index("column")
    domain = {c: f"as without Stockfish: {r}" for c, r in without["reason"].items()} | DOMAIN_STOCKFISH
    removed = decide(INSPECTION, domain, KEEP, candidates=stockfish)
    same = removed["column"].isin(without.index)
    for field in ("rule", "basis"):
        removed.loc[same, field] = removed.loc[same, "column"].map(without[field])
    removed.loc[same, "reason"] = removed.loc[same, "column"].map(without["reason"])
    return removed


def stratified_run_decisions() -> pd.DataFrame:
    path = PREPROCESSING_ROOT / "stratified" / "removed.csv"
    if not path.exists():
        sys.exit(f"{path} is missing: the balanced run takes the decisions of the stratified run, so reduce that first")
    return pd.read_csv(path)


def write_without(src, dst, drop) -> int:
    """Copy the table src to dst without the columns in drop. Returns the number of rows."""
    names = pq.read_schema(src).names
    table = pq.read_table(src, columns=[c for c in names if c not in drop])
    pq.write_table(table, dst, compression="zstd")
    return table.num_rows


def run() -> None:
    if not RULES["apply"]:
        removed, keep = pd.DataFrame(columns=COLUMNS), {}
    elif SELECTION != "stratified":
        removed, keep = stratified_run_decisions(), KEEP
    else:
        removed, keep = decisions(), KEEP
    PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    for part in PARTS:
        n = write_without(PREPROCESSING_WORK / f"clean_{part}.parquet", PREPROCESSING_DIR / f"{part}.parquet",
                          list(removed["column"]))
        print(f"{part}: {n:,} moves, {len(removed)} columns removed")
    removed.to_csv(PREPROCESSING_DIR / "removed.csv", index=False)
    pd.DataFrame([{"column": c, "reason": r} for c, r in keep.items()], columns=["column", "reason"]) \
        .to_csv(PREPROCESSING_DIR / "kept.csv", index=False)
    for _, r in removed.iterrows():
        print(f"  - {r['column']:22s} {r['rule']:9s} {r['reason']}")


if __name__ == "__main__":
    run()
