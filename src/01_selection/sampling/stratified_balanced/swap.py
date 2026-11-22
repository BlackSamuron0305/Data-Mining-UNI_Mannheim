# Swap training games until every bucket has the month's blunder rate
import numpy as np

from src.config import SEED


def balance(sel, pool, groups, ent, rate, share, seed: int) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng([SEED, 7, seed])
    sel = sel.copy()
    tries, patience, max_proposals = 32, 20_000, 1_500_000
    move_weight = 0.1

    moves, blunders, _ = ent.totals(sel)
    r = np.nan_to_num(rate)
    target = share * moves.sum()
    var = r * (1 - r) * np.maximum(moves, 1) + 0.25
    w = move_weight / np.maximum(target, 1)
    d = blunders - r * moves
    e = moves - target
    start = {"loss_blunders": float((d ** 2 / var).sum()), "loss_moves": float((e ** 2 * w).sum())}
    loss = start["loss_blunders"] + start["loss_moves"]

    sel_by_group = {g: np.flatnonzero(groups[sel] == g) for g in np.unique(groups[sel])}
    pool_by_group = {g: pool[groups[pool] == g] for g in sel_by_group}
    group_list = np.array([g for g in sel_by_group if len(pool_by_group[g]) > 0])
    group_prob = np.array([len(sel_by_group[g]) for g in group_list], dtype=float)
    group_prob /= group_prob.sum()

    swaps, since_last, proposals = 0, 0, 0
    swapped = np.zeros(len(sel), dtype=bool)
    while since_last < patience and proposals < max_proposals:
        proposals += 1
        g = group_list[rng.choice(len(group_list), p=group_prob)]
        pos = sel_by_group[g][rng.integers(len(sel_by_group[g]))]
        group_pool = pool_by_group[g]
        j = rng.integers(len(group_pool), size=min(tries, len(group_pool)))
        out_m, out_k = ent.per_game(sel[pos:pos + 1])
        in_m, in_k = ent.per_game(group_pool[j])
        new_d = d - (out_k[0] - r * out_m[0]) + (in_k - r * in_m)
        new_e = e - out_m[0] + in_m
        new_loss = (new_d ** 2 / var).sum(1) + (new_e ** 2 * w).sum(1)
        best = int(np.argmin(new_loss))
        if new_loss[best] < loss - 1e-9:
            loss, d, e = float(new_loss[best]), new_d[best], new_e[best]
            sel[pos], group_pool[j[best]] = group_pool[j[best]], sel[pos]
            swapped[pos] = True
            swaps += 1
            since_last = 0
        else:
            since_last += 1

    stats = {**start, "end_loss_blunders": float((d ** 2 / var).sum()), "end_loss_moves": float((e ** 2 * w).sum()),
             "proposals": proposals, "swaps": swaps, "games_replaced": int(swapped.sum()),
             "max_abs_blunder_gap": float(np.abs(d).max()),
             "moves_change": float((e + target).sum() / target.sum() - 1)}
    return sel, stats
