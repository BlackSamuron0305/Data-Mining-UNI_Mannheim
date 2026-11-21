"""The 95% interval of a blunder probability."""
import numpy as np
import pandas as pd

Z95 = 1.959964


def wilson(k, n, z: float = Z95) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson interval of a probability with k hits in n trials (NaN for n = 0)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        mid = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        lo = np.minimum(np.maximum(mid - half, 0.0), p)
        hi = np.maximum(np.minimum(mid + half, 1.0), p)
        return lo, hi


def add_rates(df: pd.DataFrame, moves: str = "moves", blunders: str = "blunders", suffix: str = "") -> pd.DataFrame:
    """Add the rate and its 95% interval to `df`."""
    m = df[moves].to_numpy(dtype=float)
    k = df[blunders].to_numpy(dtype=float)
    safe_m = np.where(m > 0, m, 1)
    df[f"rate{suffix}"] = np.where(m > 0, k / safe_m, np.nan)
    lo, hi = wilson(k, m)
    df[f"rate_lo{suffix}"] = lo
    df[f"rate_hi{suffix}"] = hi
    return df
