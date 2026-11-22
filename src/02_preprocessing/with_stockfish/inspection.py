"""Inspect the columns of the moves with Stockfish (training games only).

Usage:  uv run python -m src.02_preprocessing.with_stockfish.inspection   (after preprocess.py)
Output: data/preprocessing/<selection>/with_stockfish/inspection/; look at it in report.ipynb
"""
from ..common.inspection import inspect_table
from ..common.paths import WITH_DIR

if __name__ == "__main__":
    inspect_table(WITH_DIR / "clean" / "train.parquet", WITH_DIR / "inspection")
