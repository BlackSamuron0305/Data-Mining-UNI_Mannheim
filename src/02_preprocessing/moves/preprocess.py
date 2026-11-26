"""Step 2.3: remove the false predictors from the extracted moves and add the Stockfish columns.

Usage:  uv run python -m src.02_preprocessing.moves.preprocess   (after extract.py, and engine/stockfish.py or its download)
Output: data/02_preprocessing/work/<selection>/clean_train.parquet, clean_test.parquet;
        data/02_preprocessing/<selection>/false_predictors.csv

The Stockfish columns come from data/02_preprocessing/stockfish/: downloaded from our server (data.stockfish:
download in config.yaml) or computed by engine/stockfish.py (data.stockfish: compute).
"""
import logging
import sys

import pandas as pd
import pyarrow.parquet as pq

from src.config import (KEYS, PREPROCESSING_DIR, PREPROCESSING_ROOT, PREPROCESSING_WORK, STOCKFISH, STOCKFISH_DIR,
                        log_path, setup_log)
from src.common.server import download

log = logging.getLogger("preprocess")
PARTS = ("train", "test")

# The columns we remove because they are only known after the move or after the game, or say who played
FALSE_PREDICTORS = {
    "mark": ("after the move", "Lichess's mark of this move (?!, ?, ??): the ?? mark is the target itself"),
    "loss": ("after the move", "how much this move lost; a blunder is defined by it"),
    "eval_after": ("after the move", "the evaluation after this move shows whether it was a blunder"),
    "clock_after": ("after the move", "the clock after this move"),
    "spent": ("after the move", "the time spent on this move"),
    "result": ("after the game", "who won; a blunder often decides the game"),
    "rating_change": ("after the game", "rating points won or lost, follows from the result"),
    "termination": ("after the game", "how the game ended (normal or on time)"),
    "game_plies": ("after the game", "the length of the game, tells how many moves are still to come"),
    "opening": ("after the opening", "the opening's name, which names moves that are not played yet"),
    "eco_full": ("after the opening", "the opening letter at every move; we keep it only from move 12 on (eco)"),
    "mover_name": ("identifier", "the player's name: says who played, not how the move is played"),
    "opp_name": ("identifier", "the opponent's name"),
}


def drop_false_predictors(df):
    """The table without the false predictors."""
    return df.drop(columns=[c for c in FALSE_PREDICTORS if c in df.columns])


def decision_table(decisions: dict) -> pd.DataFrame:
    """One row per removed column: its reason and on what the decision rests."""
    rows = []
    for col, (basis, reason) in decisions.items():
        rows.append({"column": col, "basis": basis, "reason": reason})
    return pd.DataFrame(rows, columns=["column", "basis", "reason"])


def stockfish_table(part: str):
    """The Stockfish columns of the part, downloaded first if needed."""
    path = STOCKFISH_DIR / f"{part}.parquet"
    if not path.exists():
        if STOCKFISH == "compute":
            sys.exit(f"{path} is missing: run uv run python -m src.02_preprocessing.engine.stockfish first")
        print("no Stockfish columns on this computer: downloading them from our server")
        download([STOCKFISH_DIR.relative_to(PREPROCESSING_ROOT.parent).as_posix()])
    return pq.read_table(path)


def run() -> None:
    setup_log(log_path(PREPROCESSING_ROOT, f"preprocess_{PREPROCESSING_DIR.name}"))
    PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    for part in PARTS:
        src = PREPROCESSING_WORK / f"extracted_{part}.parquet"
        if not src.exists():
            sys.exit(f"{src} is missing: run uv run python -m src.02_preprocessing.moves.extract first")
        names = [c for c in pq.read_schema(src).names if c not in FALSE_PREDICTORS]
        moves = pq.read_table(src, columns=names)
        table = moves.join(stockfish_table(part), keys=KEYS, join_type="left outer")
        table = table.sort_by([("game_id", "ascending"), ("ply", "ascending")])
        missing = table["sf_wp_best"].null_count
        if missing:
            games = len(set(table.filter(table["sf_wp_best"].is_null())["game_id"].to_pylist()))
            sys.exit(f"{part}: {missing:,} moves of {games:,} games have no Stockfish columns (results of an older "
                     "selection?): run uv run python -m src.02_preprocessing.engine.stockfish, which analyses only the "
                     "missing games")
        pq.write_table(table, PREPROCESSING_WORK / f"clean_{part}.parquet", compression="zstd")
        log.info("%s: %s moves, %d false predictors removed, Stockfish columns added", part,
                 f"{table.num_rows:,}", len(FALSE_PREDICTORS))
    decision_table(FALSE_PREDICTORS).to_csv(PREPROCESSING_DIR / "false_predictors.csv", index=False)


if __name__ == "__main__":
    run()
