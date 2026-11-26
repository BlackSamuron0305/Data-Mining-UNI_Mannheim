"""Step 1 in one command: the cleanup of the month and its blunder rates, the three selections and their comparison,
and the counts of August to compare the months. A part whose result is there and up to date is skipped; the order and
the checks are those of src/pipeline.py.

Usage:  uv run python -m src.01_selection [--dry-run]
"""
from src.pipeline import step_main

if __name__ == "__main__":
    step_main("01_selection")
