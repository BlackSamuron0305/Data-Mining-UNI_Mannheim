"""Folders of the preprocessing."""
from src.config import DATA

PRE_DIR = DATA / "preprocessing"
EXTRACTED_DIR = PRE_DIR / "extracted"
WITHOUT_DIR = PRE_DIR / "without_stockfish"
WITH_DIR = PRE_DIR / "with_stockfish"
STOCKFISH_DIR = WITH_DIR / "stockfish"
PARTS = ("train", "test")
KEYS = ["game_id", "ply"]
TARGET = "blunder"
BOOKKEEPING = ["game_id", "ply", "bucket", "group", "blunder"]
