# Moves and blunders per bucket of every pool game
import numpy as np

from src.common.buckets import N_B


class Entries:

    def __init__(self, n_games: int, rows, buckets, moves, blunders):
        if np.any(np.diff(rows) < 0):
            raise ValueError("bucket entries are not ordered by game")
        self.start = np.searchsorted(rows, np.arange(n_games + 1)).astype(np.int64)
        self.buckets = buckets.astype(np.int64)
        self.moves = moves.astype(float)
        self.blunders = blunders.astype(float)

    # Find the entries that belong to the games in ids
    def find(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        first = self.start[ids]
        lengths = self.start[ids + 1] - first
        game_pos = np.repeat(np.arange(len(ids)), lengths)
        offset = np.repeat(first - (np.cumsum(lengths) - lengths), lengths)
        return game_pos, offset + np.arange(lengths.sum())

    # Moves and blunders per bucket, one row per game
    def per_game(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        game_pos, idx = self.find(ids)
        key = game_pos * N_B + self.buckets[idx]
        size = len(ids) * N_B
        moves = np.bincount(key, weights=self.moves[idx], minlength=size).reshape(len(ids), N_B)
        blunders = np.bincount(key, weights=self.blunders[idx], minlength=size).reshape(len(ids), N_B)
        return moves, blunders

    # Moves, blunders and games per bucket over the games in ids
    def totals(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, idx = self.find(ids)
        b = self.buckets[idx]
        moves = np.bincount(b, weights=self.moves[idx], minlength=N_B)
        blunders = np.bincount(b, weights=self.blunders[idx], minlength=N_B)
        return moves, blunders, np.bincount(b, minlength=N_B)
