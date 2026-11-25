"""Step 1.4: draw the games of a selection from the pool and split them into training and test games.

  random               every game has the same chance; a share of them are test games
  stratified           every group (speed x mean Elo) gets its share of the month, and the same share of every group
                       are test games; the test games are balanced to the month's blunder rate in every bucket
                       (swap.py), the training games stay as drawn
  stratified_balanced  the same draw and split, with the training games balanced as well; its test games are the
                       same as in the stratified run, so both runs are scored on the same games

Usage:  uv run python -m src.01_selection.select --run random|stratified|stratified_balanced   (after the cleanup)
Output: data/01_selection/<run>/ (see write.py)
"""
import argparse
import logging
import time

import numpy as np

from src.config import CFG, SELECTION_ROOT, SELECTIONS, log_path, setup_log
from src.common.buckets import uniform
from .sampling import Pool, chance_by_group, draw_random, draw_stratified, same_chance, test_part
from .swap import balance_part
from .write import make_games, write_selection

N = CFG["selection"]["n_games"]


def random_draw(pool):
    u = np.array([uniform(g, "ie500-random") for g in pool.games["game_id"]])
    sel = draw_random(u, N)
    test = test_part(np.zeros(len(sel)), pool.v[sel])
    train, test = sel[~test], sel[test]
    games = same_chance(make_games(pool, train, test), pool.meta["n_complete_games"])
    return games, {"train": train, "test": test}, None


def stratified_draw(pool, balance_train: bool, log):
    sel = draw_stratified(pool.u, pool.group, pool.groups["games"].to_numpy(), N)
    is_test = test_part(pool.group[sel], pool.v[sel])
    stats = {}
    test, stats["test"] = balance_part(pool, sel, sel[is_test], "test")
    train = sel[~is_test]
    if balance_train:
        train, stats["train"] = balance_part(pool, sel, train, "train")
    for part in stats:
        log.info("%s part: %s games swapped", part, f"{stats[part]['games_replaced']:,}")
    games = make_games(pool, train, test)
    games["replacement"] = ~np.isin(np.concatenate([train, test]), sel)
    games = chance_by_group(games, pool.groups["games"])
    return games, {"train": train, "test": test}, stats


def run(name: str) -> None:
    setup_log(log_path(SELECTION_ROOT, f"select_{name}"))
    log = logging.getLogger(name)
    t0 = time.time()
    pool = Pool()
    if name == "random":
        games, parts, stats = random_draw(pool)
    else:
        games, parts, stats = stratified_draw(pool, name == "stratified_balanced", log)
    write_selection(name, pool, games, parts, stats, time.time() - t0)
    log.info("%s: %s training and %s test games, %.1f min", name, f"{len(parts['train']):,}",
             f"{len(parts['test']):,}", (time.time() - t0) / 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", choices=SELECTIONS, required=True)
    run(ap.parse_args().run)
