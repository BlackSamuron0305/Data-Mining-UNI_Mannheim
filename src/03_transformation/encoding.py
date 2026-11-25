"""How the moves are coded for the models.

Fixed for every model and set in step 3 (transform.py): the text columns as categories with a fixed list of values,
and the training games held out to choose between runs. Learned by a model from the training games only, in its own
pipeline in step 4: true/false as 1/0, gaps filled and numbers scaled (numeric_prep), categories as numbers for the
trees (tree_features).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, StandardScaler

from src.config import CFG, SEED
from src.common.buckets import SPEEDS, uniform

# The text columns and their values, so that the training and the test games get the same categories
CATEGORIES = {"speed": SPEEDS, "eco": list("ABCDE")}
CATEGORICAL = list(CATEGORIES)
# The training games held out to choose between runs: about transformation.valid_frac of them, chosen by game, the
# same for every run
VALID_FRAC = CFG["transformation"]["valid_frac"]
FOLDS = ["train", "valid"]


def add_categories(df, cols):
    """The text columns of df as pandas categories with a fixed list of values."""
    for c in cols:
        if c in CATEGORIES:
            df[c] = pd.Categorical(df[c], categories=CATEGORIES[c])
    return df


def is_valid_game(game_ids) -> np.ndarray:
    """True for the held-out games."""
    return np.array([uniform(g, "ie500-valid") < VALID_FRAC for g in game_ids], dtype=bool)


def fold_of(game_ids: pd.Series) -> pd.Categorical:
    """'valid' for the moves of the held-out games, else 'train'."""
    ids = pd.unique(game_ids)
    valid = dict(zip(ids, is_valid_game(ids)))
    held_out = game_ids.map(valid).to_numpy(dtype=bool)
    return pd.Categorical(np.where(held_out, "valid", "train"), categories=FOLDS)


def features(df, cols):
    """The feature columns, booleans as numbers, categories kept as categories."""
    x = df[list(cols)].copy()
    for c in x.columns:
        if x[c].dtype == bool:
            x[c] = x[c].astype(np.float32)
    return x


def tree_features(df, cols):
    """Features for scikit-learn trees: a category becomes its rank, missing values stay NaN."""
    x = features(df, cols)
    for c in x.columns:
        if isinstance(x[c].dtype, pd.CategoricalDtype):
            codes = x[c].cat.codes.to_numpy()
            x[c] = np.where(codes < 0, np.nan, codes).astype(np.float32)
    return x


def scaler(name="standard"):
    """standard: mean 0 and spread 1; quantile: every column reshaped into a bell curve."""
    if name == "quantile":
        return QuantileTransformer(output_distribution="normal", subsample=200_000, random_state=SEED)
    return StandardScaler()


def numeric_prep(cols, scale=None):
    """For the models that need numbers on one scale (logistic regression, SVM, Naive Bayes, neural network): median
    fill plus missing indicators, then scaling; categorical features one-hot encoded."""
    if scale is None:
        scale = scaler()
    cat = [c for c in cols if c in CATEGORICAL]
    num = [c for c in cols if c not in CATEGORICAL]
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), scale), num),
        ("missing", MissingIndicator(features="missing-only", error_on_new=False), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
