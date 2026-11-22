"""The text columns and their values, so that the training and the test games get the same categories."""
import pandas as pd

from src.common.buckets import SPEEDS

CATEGORIES = {"speed": SPEEDS, "eco": list("ABCDE")}
CATEGORICAL = list(CATEGORIES)


def add_categories(df, cols):
    """The text columns of df as pandas categories with a fixed list of values."""
    for c in cols:
        if c in CATEGORIES:
            df[c] = pd.Categorical(df[c], categories=CATEGORIES[c])
    return df
