"""Download the raw Lichess file of August 2026 (about 30 GB), only to compare its blunder probability with
September's (preparation/blunder_rates/months.py).

Usage:  uv run python -m src.01_selection.download.august
Output: data/01_selection/raw/august/ (or LICHESS_LOCAL_DIR of .env), checked against Lichess's sha256
"""
import logging

from .storage import download

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    download("2026-08")
