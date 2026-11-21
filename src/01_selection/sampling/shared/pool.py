"""Load the pool of games that the cleanup kept."""
import json

import numpy as np
import pandas as pd

from src.config import CLEANUP_DIR
from src.lichess.share import download
from src.shared.buckets import uniform
from .entries import Entries


class Pool:
    """The pool games, their buckets and the counts of the month."""

    def __init__(self):
        meta_path = CLEANUP_DIR / "cleanup.json"
        if not meta_path.exists():
            print("no cleanup on this computer: downloading it from the team's server")
            download(["cleanup"])
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
