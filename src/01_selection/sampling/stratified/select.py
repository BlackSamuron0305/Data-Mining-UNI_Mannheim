"""Step 1.4, stratified selection, for the stratified run: every group (speed x mean Elo) gets its share of the
month, and the same share of every group are test games. The training games stay as drawn; the test games are
balanced to the month's blunder rate in every bucket (common/swap.py), the same test games as in the balanced run.

Usage:  uv run python -m src.01_selection.sampling.stratified.select   (after the cleanup)
Output: data/01_selection/sampling/stratified/ (see sampling/common/write.py)
"""
import logging
import time

import numpy as np

from src.config import CFG, LOGS_SUB, setup_log
from ..common.pool import Pool
from ..common.split import draw_stratified, test_part
from ..common.swap import balance_part
from ..common.weights import chance_by_group
from ..common.write import folder, make_games, write_selection

NAME = "stratified"
N = CFG["selection"]["n_games"]
log = logging.getLogger(NAME)


def run() -> None:
    out = folder(NAME)
    setup_log(out / LOGS_SUB / "selection.log")
    t0 = time.time()
    pool = Pool()
    sel = draw_stratified(pool.u, pool.group, pool.groups["games"].to_numpy(), N)
    is_test = test_part(pool.group[sel], pool.v[sel])
    train = sel[~is_test]
    test, test_stats = balance_part(pool, sel, sel[is_test], "test")
    log.info("test part: %s games, %s swapped, %.1f min", f"{len(test):,}", f"{test_stats['games_replaced']:,}",
             (time.time() - t0) / 60)
    games = make_games(pool, train, test)
    games["replacement"] = ~np.isin(np.concatenate([train, test]), sel)
    games = chance_by_group(games, pool.groups["games"])
    write_selection(NAME, pool, games, {"train": train, "test": test}, {"test": test_stats}, time.time() - t0)
    log.info("%s: %s training and %s test games, %.1f min", NAME, f"{len(train):,}", f"{len(test):,}",
             (time.time() - t0) / 60)


if __name__ == "__main__":
    run()
