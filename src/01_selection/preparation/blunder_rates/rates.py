"""The blunder probability of every bucket of moves, from the counts of the cleanup.

Usage:  uv run python -m src.01_selection.preparation.blunder_rates.rates

Writes to data/blunder_rates/.
"""
import json
import logging

import numpy as np
import pandas as pd

from src.config import BLUNDER_RATES_DIR, CLEANUP_DIR, LOGS_SUB, MONTH, setup_log
from src.lichess.share import download
from src.shared.buckets import SPEEDS, shrunk, time_order
from .intervals import add_rates

log = logging.getLogger("blunder_rates")


def bucket_rates() -> pd.DataFrame:
    """Blunder probability per bucket of the month, with the baseline probability and its fallback."""
    bt = pd.read_csv(CLEANUP_DIR / "buckets.csv", index_col="bucket")
    out = add_rates(bt[["speed", "mover_elo", "time_left", "moves", "blunders", "games", "few_games"]].copy())
    by_elo = bt.groupby(["speed", "mover_elo"])[["moves", "blunders"]].transform("sum")
    by_speed = bt.groupby("speed")[["moves", "blunders"]].transform("sum")
    rate_elo = by_elo["blunders"] / by_elo["moves"].replace(0, np.nan)
    rate_speed = by_speed["blunders"] / by_speed["moves"].replace(0, np.nan)
    out["fallback"] = np.where(out["moves"] > 0, "", np.where(by_elo["moves"] > 0, "speed x elo", "speed"))
    out["rate_baseline"] = shrunk(out["blunders"], out["moves"], rate_elo.fillna(rate_speed))
    return out


def by_time_left(df: pd.DataFrame, cols: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Moves and blunders summed over the Elo bands, per speed and time step; `cols` maps a suffix to the
    (moves, blunders) columns."""
    sums = []
    for m, k in cols.values():
        sums += [m, k]
    t = df.groupby(["speed", "time_left"], sort=False)[sums].sum().reset_index()
    for suffix, (m, k) in cols.items():
        add_rates(t, m, k, suffix)
    t["_o"] = t["speed"].map(SPEEDS.index)
    t["_s"] = time_order(t["time_left"])
    t = t.sort_values(["_o", "_s"]).drop(columns=["_o", "_s"])
    return t.reset_index(drop=True)


def run() -> None:
    meta_path = CLEANUP_DIR / "cleanup.json"
    if not meta_path.exists():
        print("no cleanup on this computer: downloading it from the team's server")
        download(["cleanup"])
    cleanup = json.loads(meta_path.read_text(encoding="utf-8"))
    (BLUNDER_RATES_DIR / LOGS_SUB).mkdir(parents=True, exist_ok=True)
    setup_log(BLUNDER_RATES_DIR / LOGS_SUB / "rates.log")

    rates = bucket_rates()
    rates.to_csv(BLUNDER_RATES_DIR / "rates.csv")
    tl = by_time_left(rates, {"": ("moves", "blunders")})
    tl.to_csv(BLUNDER_RATES_DIR / "by_time_left.csv", index=False)

    used = rates["moves"] > 0
    lines = [f"# Lichess {MONTH}: blunder probability per bucket of moves", "",
             f"From every complete game of the month: {cleanup['n_complete_games']:,} games, {cleanup['n_moves']:,} "
             f"moves, blunder probability {100 * cleanup['blunder_rate']:.3f}%.", "",
             f"- buckets (speed x Elo of the mover x time left): {len(rates)}, with a move: {int(used.sum())}, "
             f"with fewer than the minimum of games: {int(rates['few_games'].sum())}",
             f"- blunder probability per bucket: {100 * rates['rate'].min():.2f}% to {100 * rates['rate'].max():.2f}%",
             f"- widest 95% interval: {100 * (rates['rate_hi'] - rates['rate_lo']).max():.2f} points", "",
             "## By speed and time left (all Elo bands), blunder probability in %", "",
             "| speed | time left | moves | rate | 95% interval |", "|:--|:--|--:|--:|--:|"]
    for _, r in tl.iterrows():
        lines.append(f"| {r['speed']} | {r['time_left']} | {int(r['moves']):,} | {100 * r['rate']:.2f} | "
                     f"{100 * r['rate_lo']:.2f}-{100 * r['rate_hi']:.2f} |")
    (BLUNDER_RATES_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("%d buckets, %d with a move, rates %.2f%% to %.2f%%", len(rates), int(used.sum()),
             100 * rates["rate"].min(), 100 * rates["rate"].max())


if __name__ == "__main__":
    run()
