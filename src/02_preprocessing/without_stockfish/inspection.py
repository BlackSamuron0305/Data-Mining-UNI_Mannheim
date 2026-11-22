"""Inspect the columns of the moves without Stockfish (training games only).

Usage:  uv run python -m src.02_preprocessing.without_stockfish.inspection   (after preprocess.py)
Output: data/preprocessing/without_stockfish/inspection/; look at it in report.ipynb
"""
from ..common.inspection import inspect_table
from ..common.paths import WITHOUT_DIR

if __name__ == "__main__":
    inspect_table(WITHOUT_DIR / "clean" / "train.parquet", WITHOUT_DIR / "inspection")
