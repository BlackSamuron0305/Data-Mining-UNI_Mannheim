"""Remove the columns we do not need, by the rules of common/reduction.py on the inspection (see report.ipynb).

Usage:  uv run python -m src.02_preprocessing.without_stockfish.reduce
        (after inspection.py; the balanced run: after the stratified run)
Output: data/02_preprocessing/<selection>/without_stockfish/final/train.parquet, test.parquet, removed.csv, kept.csv
"""
from ..common.paths import WITHOUT_DIR
from ..common.reduction import decide, reduce_tables

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


def decisions():
    return decide(WITHOUT_DIR / "inspection", DOMAIN, KEEP, notes=NOTES), KEEP


if __name__ == "__main__":
    reduce_tables("without_stockfish", decisions)
