"""Step 1.5: the three selections against the month, bucket by bucket, and fresh stratified draws that show how far
a selection strays from the month by chance.

Usage:  uv run python -m src.01_selection.comparison.compare   (after the selections and the blunder rates)
Output: data/01_selection/comparison/: summary.md, comparison.json, buckets.csv, by_time_left.csv, weights.csv;
        figures from compare_report.ipynb
"""
import json
import logging

import numpy as np
import pandas as pd

from src.config import CFG, COMPARISON_DIR, MONTH_DIR, SEED, SELECTION_ROOT, SELECTIONS, log_path, setup_log
from ..month.intervals import add_rates
from ..month.rates import by_time_left
from ..sampling.draw import Pool, draw_stratified, test_part

log = logging.getLogger("comparison")
PARTS = ("train", "test")
DRAWS = int(CFG["selection"]["check_draws"])
N = CFG["selection"]["n_games"]


# How far one part is from the month, overall and per bucket
def part_stats(m, k, p, month_moves) -> tuple[dict, np.ndarray]:
    m, k = m.astype(float), k.astype(float)
    ok = (m > 0) & (p > 0) & (p < 1)
    z = np.full(len(m), np.nan)
    z[ok] = (k[ok] - p[ok] * m[ok]) / np.sqrt(p[ok] * (1 - p[ok]) * m[ok])
    on = m > 0
    diff = np.abs(k[on] / m[on] - p[on])
    chi2, df = float(np.sum(z[ok] ** 2)), int(ok.sum())
    share_over_2 = float(np.mean(np.abs(z[ok]) > 2)) if df else float("nan")
    stats = {"chi2": chi2, "df": df, "chi2_per_df": chi2 / max(df, 1), "share_abs_z_over_2": share_over_2,
             "mean_abs_diff_pp": 100 * float(np.sum(month_moves[on] * diff) / np.sum(month_moves[on])),
             "moves": int(m.sum()), "blunder_rate": float(k.sum() / m.sum())}
    return stats, z


# The same for fresh stratified draws, to see how far chance alone goes
def fresh_draws(p, month_moves) -> dict:
    pool = Pool()
    group_games = pool.groups["games"].to_numpy()
    out = {"train": [], "test": []}
    for r in range(DRAWS):
        rng = np.random.default_rng([SEED, 11, r])
        sel = draw_stratified(rng.random(len(pool.games)), pool.group, group_games, N)
        test = test_part(pool.group[sel], rng.random(len(sel)))
        for q, ids in (("train", sel[~test]), ("test", sel[test])):
            m, k, _ = pool.entries.totals(ids)
            out[q].append(part_stats(m, k, p, month_moves)[0])
        log.info("fresh draw %d of %d: chi2/df training %.2f, test %.2f", r + 1, DRAWS,
                 out["train"][-1]["chi2_per_df"], out["test"][-1]["chi2_per_df"])
    return out


def draw_summary(draws: list[dict]) -> dict:
    chi2 = np.array([d["chi2_per_df"] for d in draws])
    return {"chi2_per_df_q05": float(np.quantile(chi2, 0.05)), "chi2_per_df_q95": float(np.quantile(chi2, 0.95)),
            "mean_abs_diff_pp": float(np.mean([d["mean_abs_diff_pp"] for d in draws])), "all": draws}


# Chance and weight of each group, for every selection
def weights_table(names: list[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        g = pd.read_csv(SELECTION_ROOT / name / "groups.csv", index_col="group")
        for group, r in g.iterrows():
            rows.append({"selection": name, "group": group, "speed": r["speed"], "mean_elo": r["mean_elo"],
                         "games_month": r["games"], "games": r["games_train"] + r["games_test"],
                         "prob": r["prob"], "weight": r["weight"]})
    return pd.DataFrame(rows)


def summary_text(stats: dict, draws: dict) -> str:
    lines = ["# The three selections against the month, bucket by bucket", "",
             "z of a bucket: (blunders - month's probability x moves) / its binomial standard deviation. A selection "
             "is as close to the month as chance allows if its chi2 per bucket lies inside the range of the fresh "
             "draws.", "",
             "| selection | part | blunder probability (%) | chi2 per bucket | buckets with abs(z) > 2 (%) | "
             "mean difference to the month (points) |", "|:--|:--|--:|--:|--:|--:|"]
    for name, parts in stats.items():
        for q, s in parts.items():
            lines.append(f"| {name} | {q} | {100 * s['blunder_rate']:.3f} | {s['chi2_per_df']:.2f} | "
                         f"{100 * s['share_abs_z_over_2']:.1f} | {s['mean_abs_diff_pp']:.3f} |")
    for q, d in draws.items():
        lines.append(f"| {DRAWS} fresh draws (5-95%) | {q} | | {d['chi2_per_df_q05']:.2f}-{d['chi2_per_df_q95']:.2f} "
                     f"| | {d['mean_abs_diff_pp']:.3f} |")
    return "\n".join(lines) + "\n"


def run() -> None:
    rates_path = MONTH_DIR / "rates.csv"
    if not rates_path.exists():
        raise FileNotFoundError(f"{rates_path} is missing: run the blunder rates first")
    names = [s for s in SELECTIONS if (SELECTION_ROOT / s / "buckets.csv").exists()]
    if not names:
        raise FileNotFoundError(f"no selection in {SELECTION_ROOT}: run at least one selection first")
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    setup_log(log_path(SELECTION_ROOT, "compare"))

    rates = pd.read_csv(rates_path, index_col="bucket")
    p = rates["rate"].fillna(0).to_numpy()
    month_moves = rates["moves"].to_numpy(dtype=float)
    table = rates[["speed", "mover_elo", "time_left", "moves", "blunders", "rate", "rate_lo", "rate_hi"]].copy()
    columns = {"": ("moves", "blunders")}
    stats = {}
    for name in names:
        b = pd.read_csv(SELECTION_ROOT / name / "buckets.csv", index_col="bucket")
        stats[name] = {}
        for q in PARTS:
            col = f"{name}_{q}"
            m, k = b[f"moves_{q}"].to_numpy(), b[f"blunders_{q}"].to_numpy()
            stats[name][q], table[f"z_{col}"] = part_stats(m, k, p, month_moves)
            table[f"moves_{col}"], table[f"blunders_{col}"] = m, k
            add_rates(table, f"moves_{col}", f"blunders_{col}", f"_{col}")
            columns[f"_{col}"] = (f"moves_{col}", f"blunders_{col}")
            log.info("%s %s: chi2 per bucket %.2f, mean difference %.3f points", name, q,
                     stats[name][q]["chi2_per_df"], stats[name][q]["mean_abs_diff_pp"])
    table.to_csv(COMPARISON_DIR / "buckets.csv")
    by_time_left(table, columns).to_csv(COMPARISON_DIR / "by_time_left.csv", index=False)
    weights_table(names).to_csv(COMPARISON_DIR / "weights.csv", index=False)

    draws = {q: draw_summary(d) for q, d in fresh_draws(p, month_moves).items()}
    result = {"selections": names, "draws": DRAWS, "stats": stats, "fresh_draws": draws}
    (COMPARISON_DIR / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (COMPARISON_DIR / "summary.md").write_text(summary_text(stats, draws), encoding="utf-8")


if __name__ == "__main__":
    run()
