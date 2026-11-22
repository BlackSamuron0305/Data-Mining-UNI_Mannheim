"""Remove the columns we do not need, each with its reason (see report.ipynb). The columns without Stockfish are
removed for the same reasons as in without_stockfish/reduce.py.

Usage:  uv run python -m src.02_preprocessing.with_stockfish.reduce   (after inspection.py)
Output: data/02_preprocessing/<selection>/with_stockfish/final/train.parquet, test.parquet, removed.csv
"""
from ..common.paths import WITH_DIR
from ..common.reduction import reduce_tables
from ..without_stockfish.reduce import REMOVE as REMOVE_WITHOUT

REMOVE = {
    **REMOVE_WITHOUT,
    "sf_best_move": ("domain knowledge", "the best move as text (e.g. e2e4): a name, not a number a model can use"),
    "sf_n_scored": ("inspection", "the number of legal moves Stockfish scored: the same as legal_moves "
                                  "(correlation 1.00)"),
    "sf_wp_best": ("inspection", "the best move's win probability: almost the same as eval_wp (correlation 0.99)"),
    "sf_gap_3": ("inspection", "correlation 0.95 with sf_gap_4 and 0.90 with sf_gap_5; sf_gap_2 and sf_gap_5 stay"),
    "sf_gap_4": ("inspection", "correlation 0.96 with sf_gap_5"),
    "sf_worst_2": ("inspection", "correlation 0.95 with sf_worst_1 and 0.97 with sf_worst_3; sf_worst_1 and "
                                 "sf_worst_5 stay"),
    "sf_worst_3": ("inspection", "correlation 0.97 with sf_worst_2 and sf_worst_4"),
    "sf_worst_4": ("inspection", "correlation 0.97 with sf_worst_5"),
    "sf_share_mistake": ("inspection", "correlation 0.95 with sf_share_blunder and 0.92 with sf_share_inaccuracy"),
}

if __name__ == "__main__":
    reduce_tables(WITH_DIR / "clean", WITH_DIR / "final", REMOVE)
