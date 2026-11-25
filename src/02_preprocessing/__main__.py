"""Step 2 in one command, for the stratified and the balanced run: one row per move, the leakage check, Stockfish
(only with data.stockfish: compute; else its columns are downloaded), the false predictors out, the inspection and
the reduction. A part whose result is there and up to date is skipped; the order and the checks are those of
src/pipeline.py.

Usage:  uv run python -m src.02_preprocessing [--dry-run]
"""
from src.pipeline import step_main

if __name__ == "__main__":
    step_main("02_preprocessing")
