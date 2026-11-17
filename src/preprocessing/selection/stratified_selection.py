"""Selection step 2 - stratified selection: draw the games of the month group by group and write them in their
original PGN text.

Usage:  uv run python -m src.preprocessing.selection.stratified_selection     (after census, then probabilities)

1. Groups (speed x Elo of the mean of both ratings): every group gets exactly its share of the month's
   complete games, n_games x share rounded by the largest remainders. Inside a group the games are drawn at
   random: the ones with the smallest random number u of the census.
2. Training and test games: inside every group, test_frac of the drawn games become test games (those with
   the smallest split number, a second fixed random number of the game id), so both parts keep the shares.
3. Balancing, each part on its own: single games are swapped for other random candidate games of the same
   group until the blunder probability of every bucket of moves (speed x Elo of the mover x time left) equals
   the month's, while the share of moves per bucket stays the month's as well. The training part takes its
   replacement games only from candidates with a split number of at least test_frac, the test part only from
   those below it, so no game can end up in both parts.
4. The PGN text of the selected games is copied unchanged from the candidates of the census.

Outputs (data/stratified_selection/; probabilities.py adds the chance of every game to be selected)
summary.md                           the main numbers and the games per group
selection.json                       settings, balancing statistics, totals
games/train.pgn.zst, test.pgn.zst    the selected games in their original PGN text, in the order of the monthly file
games/games.csv                      one row per game: id, part, speed, ratings, group, plies, brought in by the balancing
tables/groups.csv                    games per group: month, training and test part (probabilities.py adds the chances)
tables/buckets.csv                   moves, blunders and blunder probability per bucket: month, training and test part
logs/selection.log
"""
from __future__ import annotations

import json
import logging
import time
from compression import zstd
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import CENSUS_DIR, CFG, GAMES_SUB, LOGS_SUB, MONTH, SEED, STRATIFIED_DIR, TABLES_SUB, setup_log
from src.preprocessing.selection.census import N_B, uniform

log = logging.getLogger("stratified_selection")
SEL = CFG["selection"]
N, TEST_FRAC, MIN_GAMES = SEL["n_games"], SEL["test_frac"], SEL["min_games"]
LAMBDA = 0.1                     # weight of the move mix against the blunder probabilities in the balancing
PARTS = ("train", "test")
GAMES_DIR, TABLES_DIR, LOGS_DIR = (STRATIFIED_DIR / d for d in (GAMES_SUB, TABLES_SUB, LOGS_SUB))


# ------------------------------------------------------------------------------------------------------- draw
def allocate(n: int, shares: np.ndarray) -> np.ndarray:
    """n split over the groups in proportion to `shares`, rounded by the largest remainders (sums to n)."""
    raw = n * shares / shares.sum()
    out = np.floor(raw).astype(np.int64)
    out[np.argsort(-(raw - out), kind="stable")[: n - out.sum()]] += 1
    return out


def draw(cand: pd.DataFrame, group_games: np.ndarray) -> np.ndarray:
    """Row indices (into `cand`) of n games: exactly the share of every group, random inside it."""
    need, out = allocate(N, group_games), []
    u, h = cand["u"].to_numpy(), cand["group"].to_numpy()
    for group in np.flatnonzero(need):
        rows = np.flatnonzero(h == group)
        if len(rows) < need[group]:
            raise ValueError(f"group {group} needs {need[group]:,} games but has {len(rows):,} candidates: "
                             "raise selection.candidate_rate in config.yaml and run the census again")
        out.append(rows[np.argsort(u[rows], kind="stable")[: need[group]]])
    return np.concatenate(out)


def test_part(groups: np.ndarray, v: np.ndarray) -> np.ndarray:
    """True for the games that become test games: inside every group the test_frac share (rounded) with the
    smallest split number v."""
    test = np.zeros(len(groups), dtype=bool)
    for group in np.unique(groups):
        pos = np.flatnonzero(groups == group)
        pos = pos[np.argsort(v[pos], kind="stable")]
        test[pos[: int(round(TEST_FRAC * len(pos)))]] = True
    return test


# -------------------------------------------------------------------------------------------------- balancing
class Entries:
    """Counts per bucket of the candidate games as a sparse table: the entries of game i are indptr[i]:indptr[i+1]."""

    def __init__(self, n_cand: int, rows, buckets, moves, blunders):
        if np.any(np.diff(rows) < 0):
            raise ValueError("bucket entries are not ordered by game")
        self.indptr = np.searchsorted(rows, np.arange(n_cand + 1)).astype(np.int64)
        self.buckets, self.moves, self.blunders = buckets.astype(np.int64), moves.astype(float), blunders.astype(float)

    def _index(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(position in `ids`, entry) of every entry of the games `ids`."""
        start, end = self.indptr[ids], self.indptr[ids + 1]
        lens = end - start
        idx = np.repeat(start - np.concatenate(([0], np.cumsum(lens)[:-1])), lens) + np.arange(lens.sum())
        return np.repeat(np.arange(len(ids)), lens), idx

    def dense(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Moves and blunders per bucket of the games `ids`, one row per game."""
        pos, idx = self._index(ids)
        key, size = pos * N_B + self.buckets[idx], len(ids) * N_B
        return (np.bincount(key, weights=self.moves[idx], minlength=size).reshape(len(ids), N_B),
                np.bincount(key, weights=self.blunders[idx], minlength=size).reshape(len(ids), N_B))

    def totals(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Moves, blunders and games per bucket, summed over the games `ids`."""
        _, idx = self._index(ids)
        b = self.buckets[idx]
        return (np.bincount(b, weights=self.moves[idx], minlength=N_B),
                np.bincount(b, weights=self.blunders[idx], minlength=N_B), np.bincount(b, minlength=N_B))


def balance(sel: np.ndarray, pool: np.ndarray, groups: np.ndarray, ent: Entries, rate: np.ndarray, share: np.ndarray,
            seed: int, tries: int = 32, patience: int = 20_000, max_props: int = 1_500_000) -> tuple[np.ndarray, dict]:
    """Swap games of `sel` for games of `pool` of the same group until the blunder probability of every bucket
    equals `rate` (to within rounding), keeping the share of moves per bucket close to `share` as well.

    One proposal: a random game of the sample against `tries` random pool games of its group; the best of them
    replaces it if that lowers  sum_b d_b^2 / v_b + LAMBDA * sum_b e_b^2 / t_b,  where d_b = blunders - rate_b x
    moves (in blunders), v_b its binomial variance in the starting sample, e_b = moves - target moves of the
    bucket and t_b the target. Stops when `patience` proposals in a row bring no improvement."""
    rng = np.random.default_rng([SEED, 7, seed])
    sel, pool = sel.copy(), pool.copy()
    m, k, _ = ent.totals(sel)
    r = np.nan_to_num(rate)
    t = share * m.sum()
    v = r * (1 - r) * np.maximum(m, 1) + 0.25
    w2 = LAMBDA / np.maximum(t, 1)
    d, e = k - r * m, m - t
    start = {"loss_blunders": float((d ** 2 / v).sum()), "loss_moves": float((e ** 2 * w2).sum())}
    loss = start["loss_blunders"] + start["loss_moves"]
    by_s = {h: np.flatnonzero(groups[sel] == h) for h in np.unique(groups[sel])}          # positions in sel
    pool_by = {h: pool[groups[pool] == h] for h in by_s}
    hs = np.array([h for h in by_s if len(pool_by[h])])                                  # groups with spare games
    p_h = np.array([len(by_s[h]) for h in hs], dtype=float)
    p_h /= p_h.sum()
    swaps, since, props = 0, 0, 0
    swapped = np.zeros(len(sel), dtype=bool)
    while since < patience and props < max_props:
        props += 1
        h = hs[rng.choice(len(hs), p=p_h)]
        pos = by_s[h][rng.integers(len(by_s[h]))]
        ph = pool_by[h]
        j = rng.integers(len(ph), size=min(tries, len(ph)))
        om, ok = ent.dense(sel[pos:pos + 1])
        im, ik = ent.dense(ph[j])
        nd = d - (ok[0] - r * om[0]) + (ik - r * im)
        ne = e - om[0] + im
        new = (nd ** 2 / v).sum(1) + (ne ** 2 * w2).sum(1)
        b = int(np.argmin(new))
        if new[b] < loss - 1e-9:
            loss, d, e = float(new[b]), nd[b], ne[b]
            sel[pos], ph[j[b]] = ph[j[b]], sel[pos]                 # the swapped-out game goes back to the pool
            swapped[pos] = True
            swaps, since = swaps + 1, 0
        else:
            since += 1
    stats = {**start, "end_loss_blunders": float((d ** 2 / v).sum()), "end_loss_moves": float((e ** 2 * w2).sum()),
             "proposals": props, "swaps": swaps, "games_replaced": int(swapped.sum()),
             "max_abs_blunder_gap": float(np.abs(d).max()), "moves_change": float((e + t).sum() / t.sum() - 1)}
    return sel, stats


# ----------------------------------------------------------------------------------------------------- output
def write_pgn(games: pd.DataFrame) -> None:
    """Copy the PGN text of the selected games from the candidates, unchanged, into train.pgn.zst and test.pgn.zst."""
    part = dict(zip(games["game_id"], games["split"]))
    found = dict.fromkeys(PARTS, 0)
    files = {p: zstd.open(GAMES_DIR / f"{p}.pgn.zst", "wb", level=12) for p in PARTS}
    try:
        for batch in pq.ParquetFile(CENSUS_DIR / "candidates.parquet").iter_batches(20_000, columns=["game_id", "pgn"]):
            for gid, pgn in zip(batch.column("game_id").to_pylist(), batch.column("pgn").to_pylist()):
                p = part.get(gid)
                if p:
                    files[p].write(pgn.encode("utf-8"))
                    found[p] += 1
    finally:
        for f in files.values():
            f.close()
    expected = games["split"].value_counts().to_dict()
    if any(found[p] != expected.get(p, 0) for p in PARTS):
        raise ValueError(f"PGN text found for {found} games, expected {expected}")


def bucket_error(b: pd.DataFrame, p: str) -> float:
    """Mean difference of the blunder probability per bucket, part against month, in points (weighted by the
    month's moves, over the buckets with a move in the part)."""
    on = b[f"moves_{p}"] > 0
    return 100 * float((b.loc[on, "moves"] * (b.loc[on, f"rate_{p}"] - b.loc[on, "rate"]).abs()).sum()
                       / b.loc[on, "moves"].sum())


def write_tables(games, bt, gt, parts, census, stats, seconds) -> None:
    s = gt[["speed", "mean_elo", "games", "share"]].copy()
    for p in PARTS:
        g = games[games["split"] == p].groupby("group").size().reindex(s.index, fill_value=0)
        s[f"games_{p}"], s[f"share_{p}"] = g, g / g.sum()
    s.to_csv(TABLES_DIR / "groups.csv")
    b = bt[["speed", "mover_elo", "time_left", "moves", "blunders", "games", "rate", "share", "few_games"]].copy()
    for p, (m, k, g) in parts.items():
        b[f"moves_{p}"], b[f"blunders_{p}"], b[f"games_{p}"] = m.astype(np.int64), k.astype(np.int64), g
        b[f"rate_{p}"] = k / np.where(m > 0, m, np.nan)
        b[f"share_{p}"] = m / m.sum()
        b[f"few_games_{p}"] = g < MIN_GAMES
    b.to_csv(TABLES_DIR / "buckets.csv")

    n = {p: int((games["split"] == p).sum()) for p in PARTS}
    moves = {p: int(b[f"moves_{p}"].sum()) for p in PARTS}
    rate = {p: b[f"blunders_{p}"].sum() / moves[p] for p in PARTS}
    err = {p: bucket_error(b, p) for p in PARTS}
    meta = {"method": "stratified", "month": MONTH, "n_games": N, "test_frac": TEST_FRAC, "seed": SEED, "games": n,
            "moves": moves, "blunder_rate": rate, "bucket_error_pp": err, "balance": stats, "census": census,
            "minutes": round(seconds / 60, 1), "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (STRATIFIED_DIR / "selection.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pct = lambda x: "<0.01" if 0 < x < 0.0001 else f"{100 * x:.2f}"
    lines = [f"# Lichess {MONTH}: {N:,} games, stratified selection", "",
             f"Drawn from every complete game of the month: {census['n_complete_games']:,} games, "
             f"{census['n_moves']:,} moves, blunder probability {100 * census['blunder_rate']:.3f}%.", "",
             "| | training | test |", "|:--|--:|--:|",
             f"| games | {n['train']:,} | {n['test']:,} |",
             f"| moves | {moves['train']:,} | {moves['test']:,} |",
             f"| blunder probability (%) | {100 * rate['train']:.3f} | {100 * rate['test']:.3f} |",
             f"| games swapped by the balancing | {stats['train']['games_replaced']:,} | "
             f"{stats['test']['games_replaced']:,} |",
             f"| blunder probability per bucket, mean difference to the month (points) | {err['train']:.3f} | "
             f"{err['test']:.3f} |", "",
             "## Games per group (speed x Elo of the mean rating), share in %", "",
             "| speed | Elo | games in the month | share | training games | share | test games | share |",
             "|:--|:--|--:|--:|--:|--:|--:|--:|"]
    for _, r in s.iterrows():
        lines.append(f"| {r['speed']} | {r['mean_elo']} | {int(r['games']):,} | {pct(r['share'])} | "
                     f"{int(r['games_train']):,} | {pct(r['share_train'])} | {int(r['games_test']):,} | "
                     f"{pct(r['share_test'])} |")
    (STRATIFIED_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------------------------------------------------------------------------------------- run
def run() -> None:
    meta_path = CENSUS_DIR / "census.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} is missing: run uv run python -m src.preprocessing.selection.census first")
    census = json.loads(meta_path.read_text(encoding="utf-8"))
    for d in (GAMES_DIR, TABLES_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    setup_log(LOGS_DIR / "selection.log")
    t0 = time.time()
    bt = pd.read_csv(CENSUS_DIR / "buckets.csv", index_col="bucket")
    gt = pd.read_csv(CENSUS_DIR / "groups.csv", index_col="group")
    cand = pd.read_parquet(CENSUS_DIR / "candidates.parquet",
                           columns=["game_id", "speed", "white_elo", "black_elo", "group", "plies", "u"])
    z = np.load(CENSUS_DIR / "candidates_buckets.npz")
    ent = Entries(len(cand), z["rows"], z["bucket"], z["moves"], z["blunders"])
    log.info("%s: %s complete games, %s candidates", CENSUS_DIR.name, f"{census['n_complete_games']:,}", f"{len(cand):,}")

    groups = cand["group"].to_numpy()
    v = np.array([uniform(g, "ie500-strata-split") for g in cand["game_id"]])      # split number (salt kept as it was)
    sel = draw(cand, gt["games"].to_numpy())
    test = test_part(groups[sel], v[sel])
    free = np.setdiff1d(np.arange(len(cand)), sel)
    start = {"train": (sel[~test], free[v[free] >= TEST_FRAC], 100), "test": (sel[test], free[v[free] < TEST_FRAC], 101)}
    chosen, stats, parts = {}, {}, {}
    for p, (first, pool, seed) in start.items():
        chosen[p], stats[p] = balance(first, pool, groups, ent, bt["rate"].to_numpy(), bt["share"].to_numpy(), seed)
        parts[p] = ent.totals(chosen[p])
        log.info("%s part: %s games, %s replaced, %.1f min", p, f"{len(first):,}", f"{stats[p]['games_replaced']:,}",
                 (time.time() - t0) / 60)

    ids = np.concatenate([chosen[p] for p in PARTS])
    games = cand.loc[ids, ["game_id", "speed", "white_elo", "black_elo", "group", "plies"]].reset_index(drop=True)
    games.insert(1, "split", np.repeat(PARTS, [len(chosen[p]) for p in PARTS]))
    games.insert(3, "mean_elo", gt["mean_elo"].to_numpy()[games["group"].to_numpy()])
    games["replacement"] = ~np.isin(ids, sel)                                   # brought in by the balancing
    games.sort_values("game_id", ignore_index=True).to_csv(GAMES_DIR / "games.csv", index=False)
    write_pgn(games)
    write_tables(games, bt, gt, parts, census, stats, time.time() - t0)
    log.info("%s: %s training and %s test games, %.1f min", STRATIFIED_DIR.name, f"{len(chosen['train']):,}",
             f"{len(chosen['test']):,}", (time.time() - t0) / 60)


if __name__ == "__main__":
    run()
