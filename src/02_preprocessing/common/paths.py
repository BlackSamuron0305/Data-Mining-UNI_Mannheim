"""Folders of the preprocessing."""
from src.config import PREPROCESSING_DIR

PRE_DIR = PREPROCESSING_DIR
EXTRACTED_DIR = PRE_DIR / "extracted"
WITHOUT_DIR = PRE_DIR / "without_stockfish"
WITH_DIR = PRE_DIR / "with_stockfish"
STOCKFISH_DIR = WITH_DIR / "stockfish"
PARTS = ("train", "test")
