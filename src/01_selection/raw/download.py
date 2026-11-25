"""Step 1.1: download the raw Lichess file of a month (about 30 GB), from Lichess or our server (data.raw_from).
August is only needed to compare its blunder probability with September's (comparison/months.py).

Usage:  uv run python -m src.01_selection.raw.download [2026-09|2026-08]   (default: the month of src/settings.py)
Output: data/00_raw/<month>/ (or LICHESS_LOCAL_DIR of .env), checked against Lichess's sha256
"""
import logging
import sys

from src.config import MONTH
from .storage import download

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    download(sys.argv[1] if len(sys.argv) > 1 else MONTH)
