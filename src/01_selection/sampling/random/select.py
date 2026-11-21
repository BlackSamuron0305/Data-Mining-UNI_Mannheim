"""Random selection: games drawn from the pool at random, every game with the same chance. It uses its own
random numbers, so it is a draw of its own and not a copy of the stratified one.

Usage:  uv run python -m src.01_selection.sampling.random.select

Writes to data/selection/random/.
"""
import logging
import time

import numpy as np

from src.config import CFG, LOGS_SUB, setup_log
from src.shared.buckets import uniform
from ..shared.pool import Pool
from ..shared.split import draw_random, test_part
from ..shared.weights import same_chance
from ..shared.write import folder, make_games, write_selection

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
