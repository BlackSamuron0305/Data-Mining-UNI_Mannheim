"""Balancing: swap games of a part (training or test) until every bucket has the month's blunder rate.

The games of the part are changed one at a time. A random game of the part is compared with TRIES random spare
games of the same group (so every group keeps its size) and replaced by the one that brings the part closest to the
month, if any of them does. "Close" is measured by a loss summed over the buckets:

    (blunders - month rate x moves)^2 / binomial variance        the bucket's blunder rate is the month's
  + MOVE_WEIGHT x (moves - month share x all moves)^2 / target    the bucket keeps its share of the moves

The search stops after PATIENCE proposals in a row without a better swap, or after MAX_PROPOSALS proposals.
The test part is balanced in both runs (the same games); the training part only in the balanced run.
"""
import numpy as np

from src.config import SEED
from .split import TEST_FRAC

TRIES = 32                   # spare games compared per proposal
PATIENCE = 20_000            # proposals in a row without a better swap before the search stops
MAX_PROPOSALS = 1_500_000
MOVE_WEIGHT = 0.1            # how much a changed share of moves counts against a changed blunder rate
MIN_GAIN = 1e-9              # a swap must lower the loss by more than this
SEEDS = {"train": 100, "test": 200}   # which random sequence of proposals, per part


def spare(pool, sel: np.ndarray, part: str) -> np.ndarray:
    """The pool games that may replace a selected game of the part: not selected, and on the part's side of the
    split value (below TEST_FRAC for the test part), so the training and the test part never share a game."""
    free = np.setdiff1d(np.arange(len(pool.games)), sel)
    on_test_side = pool.v[free] < TEST_FRAC
    return free[on_test_side] if part == "test" else free[~on_test_side]


def balance_part(pool, sel: np.ndarray, ids: np.ndarray, part: str) -> tuple[np.ndarray, dict]:
    """The games `ids` of the part ('train' or 'test') of the selection `sel`, balanced to the month."""
    return balance(ids, spare(pool, sel, part), pool.group, pool.entries, pool.buckets["rate"].to_numpy(),
                   pool.buckets["share"].to_numpy(), SEEDS[part])


def balance(sel, pool, groups, ent, rate, share, seed: int) -> tuple[np.ndarray, dict]:
    """The games `sel` of a part with swaps from the spare games `pool`, and statistics of the search.

    groups: the group of every pool game; ent: their moves and blunders per bucket (Entries); rate and share: the
    month's blunder rate and share of the moves per bucket; seed: which random sequence of proposals."""
    rng = np.random.default_rng([SEED, 7, seed])
    sel = sel.copy()

    # the gaps between the training games and the month, per bucket
    moves, blunders, _ = ent.totals(sel)
    month_rate = np.nan_to_num(rate)
    target_moves = share * moves.sum()
    variance = month_rate * (1 - month_rate) * np.maximum(moves, 1) + 0.25    # + 0.25: never divide by zero
    move_weight = MOVE_WEIGHT / np.maximum(target_moves, 1)
    blunder_gap = blunders - month_rate * moves
    move_gap = moves - target_moves
    start = {"loss_blunders": float((blunder_gap ** 2 / variance).sum()),
             "loss_moves": float((move_gap ** 2 * move_weight).sum())}
    loss = start["loss_blunders"] + start["loss_moves"]

    # where the training games of each group sit in `sel`, and the spare games of each group
    slots_by_group = {g: np.flatnonzero(groups[sel] == g) for g in np.unique(groups[sel])}
    spare_by_group = {g: pool[groups[pool] == g] for g in slots_by_group}
    group_list = np.array([g for g in slots_by_group if len(spare_by_group[g]) > 0])
    group_prob = np.array([len(slots_by_group[g]) for g in group_list], dtype=float)   # groups by their size
    group_prob /= group_prob.sum()

    swaps, since_last, proposals = 0, 0, 0
    swapped = np.zeros(len(sel), dtype=bool)
    while since_last < PATIENCE and proposals < MAX_PROPOSALS:
        proposals += 1
        group = group_list[rng.choice(len(group_list), p=group_prob)]
        slot = slots_by_group[group][rng.integers(len(slots_by_group[group]))]
        spare = spare_by_group[group]
        candidates = rng.integers(len(spare), size=min(TRIES, len(spare)))

        # the gaps if the game in `slot` were replaced by each candidate (one row per candidate)
        old_moves, old_blunders = ent.per_game(sel[slot:slot + 1])
        new_moves, new_blunders = ent.per_game(spare[candidates])
        new_blunder_gap = blunder_gap - (old_blunders[0] - month_rate * old_moves[0]) + \
            (new_blunders - month_rate * new_moves)
        new_move_gap = move_gap - old_moves[0] + new_moves
        new_loss = (new_blunder_gap ** 2 / variance).sum(1) + (new_move_gap ** 2 * move_weight).sum(1)

        best = int(np.argmin(new_loss))
        if new_loss[best] < loss - MIN_GAIN:
            loss, blunder_gap, move_gap = float(new_loss[best]), new_blunder_gap[best], new_move_gap[best]
            # the old game goes back to the spare games, so it may be chosen again later
            sel[slot], spare[candidates[best]] = spare[candidates[best]], sel[slot]
            swapped[slot] = True
            swaps += 1
            since_last = 0
        else:
            since_last += 1

    stats = {**start, "end_loss_blunders": float((blunder_gap ** 2 / variance).sum()),
             "end_loss_moves": float((move_gap ** 2 * move_weight).sum()), "proposals": proposals, "swaps": swaps,
             "games_replaced": int(swapped.sum()), "max_abs_blunder_gap": float(np.abs(blunder_gap).max()),
             "moves_change": float((move_gap + target_moves).sum() / target_moves.sum() - 1)}
    return sel, stats
