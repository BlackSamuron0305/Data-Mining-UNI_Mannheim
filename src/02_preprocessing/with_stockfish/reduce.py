"""Remove the columns we do not need, by the rules of common/reduction.py on the inspection (see report.ipynb).

The columns without Stockfish are removed exactly as in without_stockfish/reduce.py, and the rules only remove
Stockfish columns: so the table with Stockfish is the one without plus Stockfish columns, and the models can show
what Stockfish adds.

Usage:  uv run python -m src.02_preprocessing.with_stockfish.reduce
        (after inspection.py and without_stockfish/reduce.py; the balanced run: after the stratified run)
Output: data/02_preprocessing/<selection>/with_stockfish/final/train.parquet, test.parquet, removed.csv, kept.csv
"""
import pandas as pd

from ..common.paths import WITH_DIR, WITHOUT_DIR
from ..common.reduction import decide, reduce_tables
from ..without_stockfish.reduce import KEEP as KEEP_WITHOUT

DOMAIN = {
    "sf_best_move": "the best move as text (e.g. e2e4): a name, not a number a model can use",
    "sf_n_scored": "the number of legal moves Stockfish scored: the same as legal_moves",
}


def decisions():
    # the columns removed without Stockfish count as removed from the start, with their rule and reason
    without = pd.read_csv(WITHOUT_DIR / "final" / "removed.csv").set_index("column")
    domain = {c: f"as without Stockfish: {r}" for c, r in without["reason"].items()} | DOMAIN
    columns = pd.read_csv(WITH_DIR / "inspection" / "columns.csv")["column"]
    stockfish = [c for c in columns if c.startswith("sf_")]
    removed = decide(WITH_DIR / "inspection", domain, KEEP_WITHOUT, candidates=stockfish)
    same = removed["column"].isin(without.index)
    for field in ("rule", "basis"):
        removed.loc[same, field] = removed.loc[same, "column"].map(without[field])
    return removed, KEEP_WITHOUT


if __name__ == "__main__":
    reduce_tables("with_stockfish", decisions)
