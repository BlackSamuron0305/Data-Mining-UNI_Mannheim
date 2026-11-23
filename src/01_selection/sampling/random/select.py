"""Step 1.4, random selection: games from the pool, every game with the same chance; a share of them are test games.

Usage:  uv run python -m src.01_selection.sampling.random.select   (after the cleanup)
Output: data/01_selection/sampling/random/ (see sampling/common/write.py)
"""
import logging
import time

import numpy as np

from src.config import CFG, LOGS_SUB, setup_log
from src.common.buckets import uniform
from ..common.pool import Pool
from ..common.split import draw_random, test_part
from ..common.weights import same_chance
from ..common.write import folder, make_games, write_selection

NAME = "random"
N = CFG["selection"]["n_games"]
log = logging.getLogger(NAME)


def run() -> None:
    out = folder(NAME)
    setup_log(out / LOGS_SUB / "selection.log")
    t0 = time.time()
    pool = Pool()
    u = np.array([uniform(g, "ie500-random") for g in pool.games["game_id"]])
    sel = draw_random(u, N)
    test = test_part(np.zeros(len(sel)), pool.v[sel])
    train, test = sel[~test], sel[test]
    games = make_games(pool, train, test)
    games = same_chance(games, pool.meta["n_complete_games"])
    write_selection(NAME, pool, games, {"train": train, "test": test}, None, time.time() - t0)
    log.info("%s: %s training and %s test games, %.1f min", NAME, f"{len(train):,}", f"{len(test):,}",
             (time.time() - t0) / 60)


if __name__ == "__main__":
    run()
