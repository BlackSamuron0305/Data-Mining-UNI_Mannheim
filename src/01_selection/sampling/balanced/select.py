"""Balanced selection: the stratified selection, then training games swapped for pool games of the same group
until every bucket has the month's blunder probability. The test games stay those of the stratified selection.

Usage:  uv run python -m src.01_selection.sampling.balanced.select

Writes to data/selection/balanced/. Swaps stay inside a group, so the games per group and the weights are those of
the stratified selection.
"""
import logging
import time

import numpy as np

from src.config import CFG, LOGS_SUB, setup_log
from ..shared.pool import Pool
from ..shared.split import TEST_FRAC, draw_stratified, test_part
from ..shared.weights import chance_by_group
from ..shared.write import folder, make_games, write_selection
from .swap import balance

NAME = "balanced"
N = CFG["selection"]["n_games"]
log = logging.getLogger(NAME)


def run() -> None:
    out = folder(NAME)
    setup_log(out / LOGS_SUB / "selection.log")
    t0 = time.time()
    pool = Pool()
    sel = draw_stratified(pool.u, pool.group, pool.groups["games"].to_numpy(), N)
    test = test_part(pool.group[sel], pool.v[sel])
    free = np.setdiff1d(np.arange(len(pool.games)), sel)
    spare = free[pool.v[free] >= TEST_FRAC]
    rate = pool.buckets["rate"].to_numpy()
    share = pool.buckets["share"].to_numpy()
    train, stats = balance(sel[~test], spare, pool.group, pool.entries, rate, share, 100)
    test = sel[test]
    log.info("training part: %s games, %s swapped, %.1f min", f"{len(train):,}", f"{stats['games_replaced']:,}",
             (time.time() - t0) / 60)
    games = make_games(pool, train, test)
    games["replacement"] = ~np.isin(np.concatenate([train, test]), sel)
    games = chance_by_group(games, pool.groups["games"])
    write_selection(NAME, pool, games, {"train": train, "test": test}, {"train": stats}, time.time() - t0)
    log.info("%s: %s training and %s test games, %.1f min", NAME, f"{len(train):,}", f"{len(test):,}",
             (time.time() - t0) / 60)


if __name__ == "__main__":
    run()
