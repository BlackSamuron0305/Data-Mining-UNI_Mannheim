"""Add the Stockfish columns to the moves without false predictors.

Usage:  uv run python -m src.02_preprocessing.with_stockfish.preprocess   (after without_stockfish/preprocess.py)
Output: data/preprocessing/<selection>/with_stockfish/clean/train.parquet, test.parquet

If the Stockfish files are not on this computer, they are downloaded from the team's server (needs .env),
unless IE500_SOURCE=scratch.
"""
import logging

import pyarrow as pa
import pyarrow.parquet as pq

from src.config import setup_log
from src.common.server import fetch
from ..common.paths import KEYS, PARTS, STOCKFISH_DIR, WITH_DIR, WITHOUT_DIR

log = logging.getLogger("with_stockfish")


def stockfish_files(part: str) -> list:
    return sorted((STOCKFISH_DIR / part).glob("*.parquet"))


def stockfish_table(part: str) -> pa.Table:
    """The Stockfish rows of a part, each move once."""
    tables = [pq.read_table(f) for f in stockfish_files(part)]
    table = pa.concat_tables(tables, promote_options="default")
    if len(tables) > 1:
        df = table.to_pandas().drop_duplicates(KEYS)
        table = pa.Table.from_pandas(df, preserve_index=False)
    return table


def run() -> None:
    WITH_DIR.mkdir(parents=True, exist_ok=True)
    setup_log(WITH_DIR / "preprocess.log")
    if not all(stockfish_files(part) for part in PARTS):
        fetch("stockfish", "src/02_preprocessing/with_stockfish/engine.py")
    for part in PARTS:
        src = WITHOUT_DIR / "clean" / f"{part}.parquet"
        if not src.exists():
            raise FileNotFoundError(f"{src} is missing: run without_stockfish/preprocess.py first")
        moves = pq.read_table(src)
        table = moves.join(stockfish_table(part), keys=KEYS, join_type="left outer")
        table = table.sort_by([("game_id", "ascending"), ("ply", "ascending")])
        dst = WITH_DIR / "clean" / f"{part}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst, compression="zstd")
        covered = table.num_rows - table["sf_wp_best"].null_count
        log.info("%s: %s moves, %s with Stockfish (%.1f%%)", part, f"{table.num_rows:,}", f"{covered:,}",
                 100 * covered / table.num_rows)


if __name__ == "__main__":
    run()
