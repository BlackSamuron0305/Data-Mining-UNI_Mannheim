"""What the selections share: the pool of games that the cleanup kept, the draw and the split into training and test
games, and the chance (weight) of every selected game."""
import json
import sys

import numpy as np
import pandas as pd

from src.config import CFG, MONTH_DIR, POOL_DIR
from src.common.server import ensure
from src.common.buckets import N_B, uniform
from ..month.filters import same_filters

TEST_FRAC = CFG["selection"]["test_frac"]
CLEANUP_STEP = "the cleanup (python -m src.01_selection.month.cleanup)"


class Entries:
    """Moves and blunders per bucket of every pool game, to add them up quickly for any set of games.

    They are kept as "entries": one row per game and bucket that the game has a move in, ordered by game. So the
    entries of game g are the rows start[g] to start[g + 1] - 1, and a selection's totals per bucket are sums over the
    rows of its games."""

    def __init__(self, n_games: int, rows, buckets, moves, blunders):
        """rows: the game (0 to n_games - 1) of every entry, in order; buckets, moves, blunders: one value per entry."""
        if np.any(np.diff(rows) < 0):
            raise ValueError("bucket entries are not ordered by game")
        # start[g] is the first entry of game g: where g would be inserted into the sorted rows
        self.start = np.searchsorted(rows, np.arange(n_games + 1)).astype(np.int64)
        self.buckets = buckets.astype(np.int64)
        self.moves = moves.astype(float)
        self.blunders = blunders.astype(float)

    def find(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For the games in ids, every entry: the position of its game in ids, and its row."""
        first = self.start[ids]                      # the first entry of every game
        lengths = self.start[ids + 1] - first        # and how many it has
        game_pos = np.repeat(np.arange(len(ids)), lengths)
        # the rows first, first + 1, ... of every game, without a loop: number all entries 0, 1, 2, ... and shift
        # every game's run of numbers so that it starts at the game's first row
        shift = np.repeat(first - (np.cumsum(lengths) - lengths), lengths)
        return game_pos, shift + np.arange(lengths.sum())

    def per_game(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Moves and blunders per bucket, one row per game in ids."""
        game_pos, idx = self.find(ids)
        cell = game_pos * N_B + self.buckets[idx]    # (game, bucket) as one number, to count with bincount
        size = len(ids) * N_B
        moves = np.bincount(cell, weights=self.moves[idx], minlength=size).reshape(len(ids), N_B)
        blunders = np.bincount(cell, weights=self.blunders[idx], minlength=size).reshape(len(ids), N_B)
        return moves, blunders

    def totals(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Moves, blunders and games per bucket over all games in ids."""
        _, idx = self.find(ids)
        bucket = self.buckets[idx]
        moves = np.bincount(bucket, weights=self.moves[idx], minlength=N_B)
        blunders = np.bincount(bucket, weights=self.blunders[idx], minlength=N_B)
        return moves, blunders, np.bincount(bucket, minlength=N_B)


class Pool:
    """The pool of games that the cleanup kept, with what the selections need of it."""

    def __init__(self):
        meta_path = ensure(MONTH_DIR / "cleanup.json", CLEANUP_STEP)
        for name in ("buckets.csv", "groups.csv"):
            ensure(MONTH_DIR / name, CLEANUP_STEP)
        for name in ("pool.parquet", "pool_buckets.parquet"):
            ensure(POOL_DIR / name, CLEANUP_STEP)
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not same_filters(self.meta):
            sys.exit(f"{MONTH_DIR} was made with other filters than src/settings.py: run {CLEANUP_STEP} again")
        self.buckets = pd.read_csv(MONTH_DIR / "buckets.csv", index_col="bucket")
        self.groups = pd.read_csv(MONTH_DIR / "groups.csv", index_col="group")
        self.games = pd.read_parquet(POOL_DIR / "pool.parquet",
                                     columns=["game_id", "speed", "white_elo", "black_elo", "group", "plies", "u"])
        z = pd.read_parquet(POOL_DIR / "pool_buckets.parquet")
        self.entries = Entries(len(self.games), z["row"].to_numpy(), z["bucket"].to_numpy(), z["moves"].to_numpy(),
                               z["blunders"].to_numpy())
        self.u = self.games["u"].to_numpy()
        self.v = np.array([uniform(g, "ie500-strata-split") for g in self.games["game_id"]])
        self.group = self.games["group"].to_numpy()


# Split n games over the groups by their share of the month
def allocate(n: int, shares: np.ndarray) -> np.ndarray:
    raw = n * shares / shares.sum()
    counts = np.floor(raw).astype(np.int64)
    missing = n - counts.sum()
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:missing]] += 1
    return counts


# Pick n games at random
def draw_random(u: np.ndarray, n: int) -> np.ndarray:
    return np.argsort(u, kind="stable")[:n]


# Pick n games so that every group gets its share
def draw_stratified(u: np.ndarray, groups: np.ndarray, group_games: np.ndarray, n: int) -> np.ndarray:
    need = allocate(n, group_games)
    chosen = []
    for group in np.flatnonzero(need):
        rows = np.flatnonzero(groups == group)
        if len(rows) < need[group]:
            raise ValueError(f"group {group} needs {need[group]:,} games but the pool has {len(rows):,}: "
                             "raise selection.pool_rate in src/settings.py and run the cleanup again")
        rows = rows[np.argsort(u[rows], kind="stable")]
        chosen.append(rows[:need[group]])
    return np.concatenate(chosen)


# Mark the test games: 20% of every group
def test_part(groups: np.ndarray, v: np.ndarray) -> np.ndarray:
    test = np.zeros(len(groups), dtype=bool)
    for group in np.unique(groups):
        pos = np.flatnonzero(groups == group)
        pos = pos[np.argsort(v[pos], kind="stable")]
        n_test = int(round(TEST_FRAC * len(pos)))
        test[pos[:n_test]] = True
    return test


# Every game has the same chance (random selection)
def same_chance(games: pd.DataFrame, month_games: int) -> pd.DataFrame:
    games["prob"] = len(games) / month_games
    for part in ("train", "test"):
        on = games["split"] == part
        games.loc[on, "part_prob"] = on.sum() / month_games
    return add_weights(games)


# Chance of a game = selected games of its group / games of its group in the month
def chance_by_group(games: pd.DataFrame, group_games: pd.Series) -> pd.DataFrame:
    selected = games.groupby("group").size()
    games["prob"] = games["group"].map(selected / group_games)
    for part in ("train", "test"):
        on = games["split"] == part
        selected = games[on].groupby("group").size()
        games.loc[on, "part_prob"] = games.loc[on, "group"].map(selected / group_games)
    return add_weights(games)


# Weight of a game = 1 / its chance, to estimate the month's numbers from a selection
def add_weights(games: pd.DataFrame) -> pd.DataFrame:
    games["weight"] = 1 / games["prob"]
    games["part_weight"] = 1 / games["part_prob"]
    return games
