# Stratified selection: every group gets its share of the month
import logging
import time

from src.config import CFG, LOGS_SUB, setup_log
from ..common.pool import Pool
from ..common.split import draw_stratified, test_part
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
    test = test_part(pool.group[sel], pool.v[sel])
    train, test = sel[~test], sel[test]
    games = make_games(pool, train, test)
    games = chance_by_group(games, pool.groups["games"])
    write_selection(NAME, pool, games, {"train": train, "test": test}, None, time.time() - t0)
    log.info("%s: %s training and %s test games, %.1f min", NAME, f"{len(train):,}", f"{len(test):,}",
             (time.time() - t0) / 60)


if __name__ == "__main__":
    run()
