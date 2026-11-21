"""Download the raw Lichess file of September 2026 from the team's S3 storage into the local folder.

Usage:  uv run python -m src.01_selection.download.september

The folder is LICHESS_LOCAL_DIR (.env), else source.local_dir of config.yaml, else cache/raw/. The cleanup then reads
the file from there instead of from S3.
"""
import logging

from src.lichess.storage import download

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    download("2026-09")
