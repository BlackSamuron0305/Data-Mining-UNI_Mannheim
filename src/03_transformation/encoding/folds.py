"""The training games held out to choose between runs: about transformation.valid_frac of them, chosen by game,
the same for every run."""
import numpy as np
import pandas as pd

from src.config import CFG
from src.common.buckets import uniform

VALID_FRAC = CFG["transformation"]["valid_frac"]
FOLDS = ["train", "valid"]


def is_valid_game(game_ids) -> np.ndarray:
    """True for the held-out games."""
    return np.array([uniform(g, "ie500-valid") < VALID_FRAC for g in game_ids], dtype=bool)


def fold_of(game_ids: pd.Series) -> pd.Categorical:
    """'valid' for the moves of the held-out games, else 'train'."""
    ids = pd.unique(game_ids)
    valid = dict(zip(ids, is_valid_game(ids)))
    held_out = game_ids.map(valid).to_numpy(dtype=bool)
    return pd.Categorical(np.where(held_out, "valid", "train"), categories=FOLDS)
