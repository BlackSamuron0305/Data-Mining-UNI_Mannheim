"""Download the raw Lichess file of September 2026 (about 30 GB), from Lichess or our server (data.raw_from).

Usage:  uv run python -m src.01_selection.download.september
Output: data/01_selection/raw/september/ (or LICHESS_LOCAL_DIR of .env), checked against Lichess's sha256
"""
import logging

from .storage import download

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    download("2026-09")
