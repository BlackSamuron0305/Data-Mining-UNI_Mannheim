"""True/false columns as 1/0."""
import numpy as np


def features(df, cols):
    """The feature columns, booleans as numbers, categories kept as categories."""
    x = df[list(cols)].copy()
    for c in x.columns:
        if x[c].dtype == bool:
            x[c] = x[c].astype(np.float32)
    return x
