# Download the raw Lichess file of August 2026
import logging

from .storage import download

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    download("2026-08")
