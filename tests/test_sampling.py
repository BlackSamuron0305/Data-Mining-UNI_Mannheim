"""Step 1, sampling: allocation to the groups, the train/test split, the per-bucket totals and the balancing."""
import importlib

import numpy as np
import pandas as pd
import pytest

from src.common.buckets import N_B

sampling = importlib.import_module("src.01_selection.sampling")
split = entries = weights = sampling
swap = importlib.import_module("src.01_selection.swap")


def test_allocate_is_proportional_and_exact():
    counts = split.allocate(10, np.array([50, 30, 20]))
    assert counts.tolist() == [5, 3, 2]
    counts = split.allocate(7, np.array([1, 1, 1]))           # largest remainder: 7 = 3 + 2 + 2
    assert counts.sum() == 7 and sorted(counts.tolist()) == [2, 2, 3]


def test_draw_stratified_takes_the_lowest_values_per_group():
    rng = np.random.default_rng(0)
    groups = np.repeat([0, 1, 2], [500, 300, 200])
    u = rng.random(len(groups))
    sel = split.draw_stratified(u, groups, np.array([5000, 3000, 2000]), 100)
    assert np.bincount(groups[sel]).tolist() == [50, 30, 20]
    for g in range(3):
        rows = np.flatnonzero(groups == g)
        assert set(sel[groups[sel] == g]) == set(rows[np.argsort(u[rows])][:np.sum(groups[sel] == g)])


def test_draw_stratified_stops_if_the_pool_is_too_small():
    with pytest.raises(ValueError, match="pool_rate"):
        split.draw_stratified(np.linspace(0, 1, 10), np.zeros(10, dtype=int), np.array([100]), 20)


def test_test_part_is_a_fifth_of_every_group():
    rng = np.random.default_rng(1)
    groups = np.repeat([0, 1], [100, 41])
    test = split.test_part(groups, rng.random(len(groups)))
    assert int(test[groups == 0].sum()) == 20
    assert int(test[groups == 1].sum()) == round(0.2 * 41)


def test_test_part_is_the_same_every_time():
    groups = np.repeat([0, 1], [50, 50])
    v = np.linspace(0, 1, 100)
    assert np.array_equal(split.test_part(groups, v), split.test_part(groups, v))


def random_entries(n_games=60, seed=2):
    """A pool of n_games games, each with moves and blunders in a few buckets, ordered by game."""
    rng = np.random.default_rng(seed)
    rows, bks, moves, blunders = [], [], [], []
    for g in range(n_games):
        for bucket in rng.choice(N_B, size=rng.integers(1, 5), replace=False):
            m = int(rng.integers(1, 30))
            rows.append(g)
            bks.append(bucket)
            moves.append(m)
            blunders.append(int(rng.binomial(m, 0.06)))
    return entries.Entries(n_games, np.array(rows), np.array(bks), np.array(moves), np.array(blunders))


def brute_force(ent, ids):
    moves, blunders = np.zeros(N_B), np.zeros(N_B)
    for g in ids:
        for i in range(ent.start[g], ent.start[g + 1]):
            moves[ent.buckets[i]] += ent.moves[i]
            blunders[ent.buckets[i]] += ent.blunders[i]
    return moves, blunders


def test_entries_totals_match_a_plain_loop():
    ent = random_entries()
    ids = np.array([3, 7, 7, 10, 59, 0])
    moves, blunders, games = ent.totals(ids)
    want_m, want_k = brute_force(ent, ids)
    assert np.array_equal(moves, want_m) and np.array_equal(blunders, want_k)
    assert games.sum() == sum(ent.start[g + 1] - ent.start[g] for g in ids)


def test_entries_per_game_match_a_plain_loop():
    ent = random_entries()
    ids = np.array([5, 1, 42])
    moves, blunders = ent.per_game(ids)
    for row, g in enumerate(ids):
        want_m, want_k = brute_force(ent, [g])
        assert np.array_equal(moves[row], want_m) and np.array_equal(blunders[row], want_k)


def test_entries_need_rows_in_order():
    with pytest.raises(ValueError):
        entries.Entries(2, np.array([1, 0]), np.array([0, 1]), np.array([1, 1]), np.array([0, 0]))


def test_weights_are_the_inverse_chance():
    games = pd.DataFrame({"group": [0, 0, 1], "split": ["train", "test", "train"]})
    out = weights.chance_by_group(games, pd.Series([100, 10], index=[0, 1]))
    assert out["prob"].tolist() == [0.02, 0.02, 0.1]
    assert out["weight"].tolist() == [50.0, 50.0, 10.0]


def test_balance_swaps_within_groups_and_lowers_the_gap():
    ent = random_entries(n_games=200, seed=3)
    groups = np.repeat([0, 1], 100)
    sel = np.r_[0:40, 100:140]
    spare = np.r_[40:100, 140:200]
    month_moves, month_blunders, _ = ent.totals(np.arange(200))
    rate = month_blunders / np.where(month_moves > 0, month_moves, np.nan)
    share = month_moves / month_moves.sum()
    new, stats = swap.balance(sel, spare, groups, ent, rate, share, 0)
    assert len(new) == len(sel)
    assert np.bincount(groups[new]).tolist() == np.bincount(groups[sel]).tolist()
    assert set(new) - set(sel) <= set(spare)
    assert len(set(new)) == len(new)
    assert stats["end_loss_blunders"] + stats["end_loss_moves"] <= stats["loss_blunders"] + stats["loss_moves"]
    again, _ = swap.balance(sel, spare, groups, ent, rate, share, 0)
    assert np.array_equal(new, again)


class FakePool:
    """What balance_part needs of the pool: games, split values, groups, entries and the month's bucket rates."""

    def __init__(self, n_games=300, seed=4):
        rng = np.random.default_rng(seed)
        self.entries = random_entries(n_games, seed)
        self.games = pd.DataFrame(index=range(n_games))
        self.v = rng.random(n_games)
        self.group = np.repeat([0, 1, 2], n_games // 3)
        moves, blunders, _ = self.entries.totals(np.arange(n_games))
        self.buckets = pd.DataFrame({"rate": blunders / np.where(moves > 0, moves, np.nan),
                                     "share": moves / moves.sum()})


def test_spare_games_of_the_two_parts_never_overlap():
    pool = FakePool()
    sel = np.arange(0, 300, 3)
    for_test, for_train = swap.spare(pool, sel, "test"), swap.spare(pool, sel, "train")
    assert not set(for_test) & set(for_train)
    assert set(for_test) | set(for_train) == set(range(300)) - set(sel)
    assert (pool.v[for_test] < split.TEST_FRAC).all() and (pool.v[for_train] >= split.TEST_FRAC).all()


def test_both_runs_get_the_same_balanced_test_part(monkeypatch):
    monkeypatch.setattr(swap, "PATIENCE", 2000)
    pool = FakePool()
    sel = np.arange(0, 300, 2)
    is_test = split.test_part(pool.group[sel], pool.v[sel])
    test_a, stats = swap.balance_part(pool, sel, sel[is_test], "test")        # the stratified run
    test_b, _ = swap.balance_part(pool, sel, sel[is_test], "test")            # the balanced run
    assert np.array_equal(test_a, test_b)
    assert np.bincount(pool.group[test_a]).tolist() == np.bincount(pool.group[sel[is_test]]).tolist()
    assert stats["end_loss_blunders"] + stats["end_loss_moves"] <= stats["loss_blunders"] + stats["loss_moves"]
    train, _ = swap.balance_part(pool, sel, sel[~is_test], "train")
    assert not set(train) & set(test_a)                                     # the parts never share a game
