"""Step 1.4, balanced selection, for the balanced run: the stratified draw and split, with the training games and
the test games each balanced to the month's blunder rate in every bucket (common/swap.py). The test games are the
same as in the stratified run, so both runs are scored on the same games.

Usage:  uv run python -m src.01_selection.sampling.stratified_balanced.select   (after the cleanup)
Output: data/01_selection/sampling/stratified_balanced/ (see sampling/common/write.py)
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

NAME = "stratified_balanced"
N = CFG["selection"]["n_games"]
log = logging.getLogger(NAME)


def run() -> None:
    out = folder(NAME)
    setup_log(out / LOGS_SUB / "selection.log")
    t0 = time.time()
    pool = Pool()
    sel = draw_stratified(pool.u, pool.group, pool.groups["games"].to_numpy(), N)
    is_test = test_part(pool.group[sel], pool.v[sel])
    stats = {}
    test, stats["test"] = balance_part(pool, sel, sel[is_test], "test")
    train, stats["train"] = balance_part(pool, sel, sel[~is_test], "train")
    for part, ids in (("test", test), ("train", train)):
        log.info("%s part: %s games, %s swapped, %.1f min", part, f"{len(ids):,}",
                 f"{stats[part]['games_replaced']:,}", (time.time() - t0) / 60)
    games = make_games(pool, train, test)
    games["replacement"] = ~np.isin(np.concatenate([train, test]), sel)
    games = chance_by_group(games, pool.groups["games"])
    write_selection(NAME, pool, games, {"train": train, "test": test}, stats, time.time() - t0)
    log.info("%s: %s training and %s test games, %.1f min", NAME, f"{len(train):,}", f"{len(test):,}",
             (time.time() - t0) / 60)


if __name__ == "__main__":
    run()
