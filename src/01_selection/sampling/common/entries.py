"""Moves and blunders per bucket of every pool game, to add them up quickly for any set of games.

They are kept as "entries": one row per game and bucket that the game has a move in, ordered by game. So the
entries of game g are the rows start[g] to start[g + 1] - 1, and a selection's totals per bucket are sums over the
rows of its games.
"""
import numpy as np

from src.common.buckets import N_B


class Entries:

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
