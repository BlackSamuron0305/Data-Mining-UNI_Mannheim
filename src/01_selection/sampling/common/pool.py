"""The pool of games that the cleanup kept, with what the selections need of it."""
import json
import sys

import numpy as np
import pandas as pd

from src.config import CLEANUP_DIR
from src.common.server import ensure
from src.common.buckets import uniform
from ...preparation.cleanup.filters import same_filters
from .entries import Entries

CLEANUP_STEP = "the cleanup (python -m src.01_selection.preparation.cleanup.cleanup)"


class Pool:

    def __init__(self):
        meta_path = ensure(CLEANUP_DIR / "cleanup.json", CLEANUP_STEP)
        for name in ("buckets.csv", "groups.csv", "pool.parquet", "pool_buckets.parquet"):
            ensure(CLEANUP_DIR / name, CLEANUP_STEP)
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not same_filters(self.meta):
            sys.exit(f"{CLEANUP_DIR} was made with other filters than src/settings.py: run {CLEANUP_STEP} again")
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
