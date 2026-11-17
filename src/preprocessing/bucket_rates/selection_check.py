"""Bucket rates step 2 - how close the selected games come to the month, bucket by bucket.

Usage:  uv run python -m src.preprocessing.bucket_rates.selection_check      (after rates.py and the selection)

For every part of the selection (training, test) and every bucket b with m_b moves and k_b blunders in the part and
the month's blunder probability p_b:

    z_b = (k_b - p_b m_b) / sqrt(p_b (1 - p_b) m_b)

Per part: chi2 = the sum of z_b^2 over the buckets with a move in the part, df = the number of these buckets, the
share of buckets with |z_b| > 2, and the mean absolute difference of the probabilities in points, weighted by the
month's moves.

The moves of one game are not independent (a player who blunders once tends to blunder again), and whole games are
drawn, so even a perfectly random draw gives a chi2 above df. The yardstick is therefore `check_draws` fresh random
stratified draws of the same size from the census's candidates (same games per group, same share of test games, no
balancing): the selection is as close to the month as chance allows if its chi2 lies among theirs. A balanced
selection lies far below them, because the balancing fits the buckets to the month on purpose.

Outputs (data/bucket_rates/)
selection_check.csv               per bucket: the month's and every part's moves, blunders, rate, 95% interval, z
selection_by_time_left.csv        per speed and time step (all Elo bands): month, training and test part with
                                  95% intervals (the figure of the paper)
selection_check.json, .md         the statistics of the selection next to those of the random draws
logs/selection_check.log
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src.config import BUCKET_RATES_DIR, CENSUS_DIR, CFG, LOGS_SUB, SEED, STRATIFIED_DIR, TABLES_SUB, setup_log
from src.preprocessing.bucket_rates.rates import add_rates, by_time_left
from src.preprocessing.selection.stratified_selection import Entries, draw, test_part

log = logging.getLogger("selection_check")
PARTS = ("train", "test")
DRAWS = int(CFG["selection"].get("check_draws", 20))


def part_stats(m: np.ndarray, k: np.ndarray, p: np.ndarray, month_moves: np.ndarray) -> tuple[dict, np.ndarray]:
    """Statistics of one part against the month, and z per bucket (NaN where the part has no move)."""
    m, k = m.astype(float), k.astype(float)
    ok = (m > 0) & (p > 0) & (p < 1)
    z = np.full(len(m), np.nan)
    z[ok] = (k[ok] - p[ok] * m[ok]) / np.sqrt(p[ok] * (1 - p[ok]) * m[ok])
    on = m > 0
    diff = np.abs(k[on] / m[on] - p[on])
    chi2, df = float(np.sum(z[ok] ** 2)), int(ok.sum())
    return {"chi2": chi2, "df": df, "chi2_per_df": chi2 / max(df, 1),
            "share_abs_z_over_2": float(np.mean(np.abs(z[ok]) > 2)) if df else float("nan"),
            "mean_abs_diff_pp": 100 * float(np.sum(month_moves[on] * diff) / np.sum(month_moves[on])),
            "moves": int(m.sum()), "blunder_rate": float(k.sum() / m.sum())}, z


def random_draws(p: np.ndarray, month_moves: np.ndarray, n_games: int) -> dict[str, list[dict]]:
    """Statistics of DRAWS random stratified draws of the selection's size, without balancing."""
    cand = pd.read_parquet(CENSUS_DIR / "candidates.parquet", columns=["game_id", "group"])
    z = np.load(CENSUS_DIR / "candidates_buckets.npz")
    ent = Entries(len(cand), z["rows"], z["bucket"], z["moves"], z["blunders"])
    group_games = pd.read_csv(CENSUS_DIR / "groups.csv", index_col="group")["games"].to_numpy()
    groups = cand["group"].to_numpy()
    out = {q: [] for q in PARTS}
    for r in range(DRAWS):
        rng = np.random.default_rng([SEED, 11, r])
        sel = draw(cand.assign(u=rng.random(len(cand))), group_games, n_games)
        test = test_part(groups[sel], rng.random(len(sel)))
        for q, ids in (("train", sel[~test]), ("test", sel[test])):
            m, k, _ = ent.totals(ids)
            out[q].append(part_stats(m, k, p, month_moves)[0])
        log.info("random draw %d of %d: chi2/df training %.2f, test %.2f", r + 1, DRAWS,
                 out["train"][-1]["chi2_per_df"], out["test"][-1]["chi2_per_df"])
    return out


def run() -> None:
    rates_path, sel_path = BUCKET_RATES_DIR / "rates.csv", STRATIFIED_DIR / TABLES_SUB / "buckets.csv"
    for path, step in ((rates_path, "bucket_rates.rates"), (sel_path, "selection.stratified_selection")):
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run uv run python -m src.preprocessing.{step} first")
    (BUCKET_RATES_DIR / LOGS_SUB).mkdir(parents=True, exist_ok=True)
    setup_log(BUCKET_RATES_DIR / LOGS_SUB / "selection_check.log")
    sel_meta = json.loads((STRATIFIED_DIR / "selection.json").read_text(encoding="utf-8"))
    n_games = int(sel_meta["n_games"])            # the random draws have the size of the selection on disk

    rates = pd.read_csv(rates_path, index_col="bucket")
    sb = pd.read_csv(sel_path, index_col="bucket")
    p, month_moves = rates["rate"].fillna(0).to_numpy(), rates["moves"].to_numpy(dtype=float)
    table = rates[["speed", "mover_elo", "time_left", "moves", "blunders", "rate", "rate_lo", "rate_hi"]].copy()
    stats = {}
    for q in PARTS:
        m, k = sb[f"moves_{q}"].to_numpy(), sb[f"blunders_{q}"].to_numpy()
        stats[q], table[f"z_{q}"] = part_stats(m, k, p, month_moves)
        table[f"moves_{q}"], table[f"blunders_{q}"] = m, k
        add_rates(table, f"moves_{q}", f"blunders_{q}", f"_{q}")
    table.to_csv(BUCKET_RATES_DIR / "selection_check.csv")
    by_time_left(table, {"": ("moves", "blunders"), **{f"_{q}": (f"moves_{q}", f"blunders_{q}") for q in PARTS}}
                 ).to_csv(BUCKET_RATES_DIR / "selection_by_time_left.csv", index=False)

    draws = random_draws(p, month_moves, n_games) if DRAWS > 0 else {q: [] for q in PARTS}
    balanced = {q: sel_meta["balance"][q].get("balanced", True) for q in PARTS}     # older runs were always balanced
    summary = {"selection_balanced": balanced, "draws": DRAWS, "selection": stats, "random_draws": {}}
    for q in PARTS:
        if not draws[q]:
            continue
        c = np.array([d["chi2"] for d in draws[q]])
        summary["random_draws"][q] = {
            "chi2_mean": float(c.mean()), "chi2_min": float(c.min()), "chi2_max": float(c.max()),
            "chi2_q05": float(np.quantile(c, 0.05)), "chi2_q95": float(np.quantile(c, 0.95)),
            "mean_abs_diff_pp_mean": float(np.mean([d["mean_abs_diff_pp"] for d in draws[q]])),
            "share_draws_with_lower_chi2": float(np.mean(c < stats[q]["chi2"])), "all": draws[q]}
    (BUCKET_RATES_DIR / "selection_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Selection against the month, bucket by bucket", "",
             "z of a bucket: (blunders - month's probability x moves) / its binomial standard deviation. A part is "
             "as close to the month as chance allows if its chi2 lies among those of the random draws.", "",
             f"Balanced selection: training {balanced['train']}, test {balanced['test']}.", "",
             "| | training | test |", "|:--|--:|--:|"]
    row = lambda name, f: lines.append(f"| {name} | " + " | ".join(f(q) for q in PARTS) + " |")
    row("moves", lambda q: f"{stats[q]['moves']:,}")
    row("blunder probability (%)", lambda q: f"{100 * stats[q]['blunder_rate']:.3f}")
    row("buckets with a move (df)", lambda q: f"{stats[q]['df']}")
    row("chi2", lambda q: f"{stats[q]['chi2']:.0f}")
    row("chi2 / df", lambda q: f"{stats[q]['chi2_per_df']:.2f}")
    row("share of buckets with abs(z) > 2", lambda q: f"{100 * stats[q]['share_abs_z_over_2']:.1f}%")
    row("mean difference to the month (points)", lambda q: f"{stats[q]['mean_abs_diff_pp']:.3f}")
    if summary["random_draws"]:
        rd = summary["random_draws"]
        lines.append(f"| chi2 of {DRAWS} random draws (5%-95%) | " + " | ".join(
            f"{rd[q]['chi2_q05']:.0f}-{rd[q]['chi2_q95']:.0f}" for q in PARTS) + " |")
        lines.append("| random draws with a lower chi2 | " + " | ".join(
            f"{100 * rd[q]['share_draws_with_lower_chi2']:.0f}%" for q in PARTS) + " |")
        lines.append("| mean difference to the month, random draws (points) | " + " | ".join(
            f"{rd[q]['mean_abs_diff_pp_mean']:.3f}" for q in PARTS) + " |")
    (BUCKET_RATES_DIR / "selection_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for q in PARTS:
        log.info("%s part: chi2 %.0f on %d buckets (%.2f per bucket), mean difference %.3f points", q,
                 stats[q]["chi2"], stats[q]["df"], stats[q]["chi2_per_df"], stats[q]["mean_abs_diff_pp"])


if __name__ == "__main__":
    run()
