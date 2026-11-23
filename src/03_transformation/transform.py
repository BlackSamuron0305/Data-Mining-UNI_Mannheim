"""Step 3: the tables the models read.

Takes the final tables of step 2 and writes them with what every model needs and nothing learned from the games:
the text columns as categories with a fixed list of values, true/false as 1/0, decimals as 32-bit floats, and for
the training games the column `fold` (the held-out games that choose between runs). What a model learns from the
games (gaps filled, scaling, one 0/1 column per category) stays in its own pipeline (encoding/numeric.py,
encoding/trees.py) and is fitted on the training games only.

Usage:  uv run python -m src.03_transformation.transform
Output: data/03_transformation/<selection>/<without|with>_stockfish/train.parquet, test.parquet, transform.json
"""
import json
import logging

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import BOOKKEEPING, FINAL_DIRS, LOGS_SUB, TRANSFORMATION_DIR, TRANSFORMED_DIRS, setup_log
from src.common.server import ensure
from .encoding.categories import add_categories
from .encoding.folds import VALID_FRAC, fold_of

log = logging.getLogger("transformation")
PARTS = ("train", "test")
BATCH = 1_000_000


def transform(df, part):
    """One batch of moves, ready for the models."""
    for c in df.columns:
        if c in BOOKKEEPING:
            continue
        if df[c].dtype == bool or df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    df = add_categories(df, [c for c in df.columns if c not in BOOKKEEPING])
    if part == "train":
        df["fold"] = fold_of(df["game_id"])
    return df


def run_part(table: str, part: str) -> dict:
    src = ensure(FINAL_DIRS[table] / f"{part}.parquet", "step 2 (src/02_preprocessing/, README.md)")
    out_dir = TRANSFORMED_DIRS[table]
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{part}.parquet.part"
    writer = None
    rows = 0
    games = set()
    valid_games = set()
    try:
        for batch in pq.ParquetFile(src).iter_batches(BATCH):
            df = transform(batch.to_pandas(), part)
            out = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                schema = out.schema
                writer = pq.ParquetWriter(tmp, schema, compression="zstd")
            writer.write_table(out.cast(schema))
            rows += len(df)
            games.update(df["game_id"].unique())
            if part == "train":
                valid_games.update(df.loc[df["fold"] == "valid", "game_id"].unique())
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(out_dir / f"{part}.parquet")
    info = {"moves": rows, "games": len(games), "columns": len(schema.names)}
    if part == "train":
        info["held_out_games"] = len(valid_games)
    log.info("%s %s: %s moves, %s games%s", table, part, f"{rows:,}", f"{len(games):,}",
             f", {len(valid_games):,} held out" if part == "train" else "")
    return info


def run() -> None:
    (TRANSFORMATION_DIR / LOGS_SUB).mkdir(parents=True, exist_ok=True)
    setup_log(TRANSFORMATION_DIR / LOGS_SUB / "transform.log")
    meta = {"valid_frac": VALID_FRAC}
    for table in TRANSFORMED_DIRS:
        meta[table] = {part: run_part(table, part) for part in PARTS}
    (TRANSFORMATION_DIR / "transform.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()
