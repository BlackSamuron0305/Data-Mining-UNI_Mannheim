"""For the scikit-learn trees: a category becomes its number, missing values stay missing."""
import numpy as np
import pandas as pd

from .booleans import features


def tree_features(df, cols):
    """Features for scikit-learn trees: a category becomes its rank, missing values stay NaN."""
    x = features(df, cols)
    for c in x.columns:
        if isinstance(x[c].dtype, pd.CategoricalDtype):
            codes = x[c].cat.codes.to_numpy()
            x[c] = np.where(codes < 0, np.nan, codes).astype(np.float32)
    return x
