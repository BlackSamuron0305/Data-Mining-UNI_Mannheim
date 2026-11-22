"""Remove the columns we do not need, each with its reason (see report.ipynb).

Usage:  uv run python -m src.02_preprocessing.without_stockfish.reduce   (after inspection.py)
Output: data/02_preprocessing/<selection>/without_stockfish/final/train.parquet, test.parquet, removed.csv
"""
from ..common.paths import WITHOUT_DIR
from ..common.reduction import reduce_tables

REMOVE = {
    "eval_cp": ("domain knowledge", "the same evaluation as eval_wp in other units (correlation 1.00); eval_wp "
                                    "stays because it lies between 0 and 1"),
    "material_own": ("inspection", "correlation 0.99 with material_total; material_total and material_diff "
                                   "give it exactly"),
    "material_opp": ("inspection", "correlation 0.99 with material_total; material_total and material_diff "
                                   "give it exactly"),
    "opp_elo": ("inspection", "correlation 0.94 with mover_elo; mover_elo and elo_diff give it exactly"),
    "opp_clock": ("inspection", "correlation 0.92 with clock; clock and clock_diff give it exactly"),
    "event_type": ("inspection", "90% of the moves are lobby games, and the blunder rate is the same in all "
                                 "three kinds (5.4-5.6%)"),
    "white": ("inspection", "no relation to blunders (correlation 0.00)"),
    "hour": ("domain knowledge", "the hour is in UTC, not the player's own time, so it cannot show tiredness; "
                                 "no relation to blunders either (correlation 0.00)"),
}

if __name__ == "__main__":
    reduce_tables(WITHOUT_DIR / "clean", WITHOUT_DIR / "final", REMOVE)
