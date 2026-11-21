"""Draw games from the pool and split them into training and test games."""
import numpy as np

from src.config import CFG

TEST_FRAC = CFG["selection"]["test_frac"]


def allocate(n: int, shares: np.ndarray) -> np.ndarray:
    """Split n over the groups by their shares, rounding by the largest remainders."""
    raw = n * shares / shares.sum()
    counts = np.floor(raw).astype(np.int64)
    missing = n - counts.sum()
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:missing]] += 1
    return counts


def draw_random(u: np.ndarray, n: int) -> np.ndarray:
    """Row numbers of the n games with the smallest u."""
    return np.argsort(u, kind="stable")[:n]


def draw_stratified(u: np.ndarray, groups: np.ndarray, group_games: np.ndarray, n: int) -> np.ndarray:
    """Row numbers of n games: every group gets its share, taking the games with the smallest u."""
    need = allocate(n, group_games)
    chosen = []
    for group in np.flatnonzero(need):
        rows = np.flatnonzero(groups == group)
        if len(rows) < need[group]:
            raise ValueError(f"group {group} needs {need[group]:,} games but the pool has {len(rows):,}: "
                             "raise selection.pool_rate in config.yaml and run the cleanup again")
        rows = rows[np.argsort(u[rows], kind="stable")]
        chosen.append(rows[:need[group]])
    return np.concatenate(chosen)


def test_part(groups: np.ndarray, v: np.ndarray) -> np.ndarray:
    """True for the test games: in every group the TEST_FRAC share with the smallest v."""
    test = np.zeros(len(groups), dtype=bool)
    for group in np.unique(groups):
        pos = np.flatnonzero(groups == group)
        pos = pos[np.argsort(v[pos], kind="stable")]
        n_test = int(round(TEST_FRAC * len(pos)))
        test[pos[:n_test]] = True
    return test
