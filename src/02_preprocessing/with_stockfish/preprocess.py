"""Add the Stockfish columns to the moves without false predictors.

Usage:  uv run python -m src.02_preprocessing.with_stockfish.preprocess   (after without_stockfish/preprocess.py)
Output: data/02_preprocessing/<selection>/with_stockfish/clean/train.parquet, test.parquet

If the Stockfish files are not on this computer, they are downloaded from our server (data.stockfish: download in
config.yaml), or engine.py has to compute them first (data.stockfish: compute).
"""
import logging
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from src.config import KEYS, PREPROCESSING_ROOT, SELECTIONS, STOCKFISH, setup_log
from src.common.server import download
from ..common.paths import PARTS, PRE_DIR, STOCKFISH_DIR, WITH_DIR, WITHOUT_DIR

log = logging.getLogger("with_stockfish")


# Stockfish files of the part from every selection; the join below keeps only the moves of this one
def stockfish_files(part: str) -> list:
    files = []
    for selection in SELECTIONS:
        folder = PREPROCESSING_ROOT / selection / STOCKFISH_DIR.relative_to(PRE_DIR) / part
        files += sorted(folder.glob("*.parquet"))
    return files


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
        if STOCKFISH == "compute":
            sys.exit("no Stockfish columns yet: run uv run python -m src.02_preprocessing.with_stockfish.engine first")
        print("no Stockfish columns on this computer: downloading them from our server")
        download(["stockfish"])
    for part in PARTS:
        src = WITHOUT_DIR / "clean" / f"{part}.parquet"
        if not src.exists():
            raise FileNotFoundError(f"{src} is missing: run without_stockfish/preprocess.py first")
        moves = pq.read_table(src)
        table = moves.join(stockfish_table(part), keys=KEYS, join_type="left outer")
        table = table.sort_by([("game_id", "ascending"), ("ply", "ascending")])
        missing = table["sf_wp_best"].null_count
        if missing:
            games = len(set(table.filter(table["sf_wp_best"].is_null())["game_id"].to_pylist()))
            sys.exit(f"{part}: {missing:,} moves of {games:,} games have no Stockfish columns (results of an older "
                     "selection?): run uv run python -m src.02_preprocessing.with_stockfish.engine, which analyses "
                     "only the missing games")
        dst = WITH_DIR / "clean" / f"{part}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, dst, compression="zstd")
        log.info("%s: %s moves, all with Stockfish", part, f"{table.num_rows:,}")


if __name__ == "__main__":
    run()
