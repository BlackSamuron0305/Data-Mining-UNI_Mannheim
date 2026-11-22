# Load the pool of games that the cleanup kept
import json

import numpy as np
import pandas as pd

from src.config import CLEANUP_DIR
from src.common.server import fetch
from src.common.buckets import uniform
from .entries import Entries


class Pool:

    def __init__(self):
        meta_path = CLEANUP_DIR / "cleanup.json"
        if not meta_path.exists():
            fetch("cleanup", "the cleanup (src/01_selection/preparation/cleanup/cleanup.py)")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.buckets = pd.read_csv(CLEANUP_DIR / "buckets.csv", index_col="bucket")
        self.groups = pd.read_csv(CLEANUP_DIR / "groups.csv", index_col="group")
        self.games = pd.read_parquet(CLEANUP_DIR / "pool.parquet",
                                     columns=["game_id", "speed", "white_elo", "black_elo", "group", "plies", "u"])
        z = pd.read_parquet(CLEANUP_DIR / "pool_buckets.parquet")
        self.entries = Entries(len(self.games), z["row"].to_numpy(), z["bucket"].to_numpy(), z["moves"].to_numpy(),
                               z["blunders"].to_numpy())
        self.u = self.games["u"].to_numpy()
        self.v = np.array([uniform(g, "ie500-strata-split") for g in self.games["game_id"]])
        self.group = self.games["group"].to_numpy()
