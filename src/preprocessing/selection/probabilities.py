"""Selection step 3 - the chance of every selected game to be in the selection, and its weight.

Usage:  uv run python -m src.preprocessing.selection.probabilities [folder in data/]   (default stratified_selection)

The chances come from the month's counts, which the selection keeps in its own groups.csv: the complete games of
the whole month per group (speed x Elo of the mean rating) next to the games it drew.

- Stratified selection: a group holds N_g complete games in the month, n_g of them were selected. Every game of
  the group had the same chance n_g / N_g to be selected, and a selected game stands for N_g / n_g games of the
  month (its weight). The training and test part each keep the shares of the groups, so inside a part the same
  holds with the games of that part.
- Random selection: every complete game of the month had the same chance n / N.

The weights of the selected games add up to the month's complete games (apart from groups without a selected
game), so a sum or a mean over the selection, weighted by them, estimates the same figure for the whole month.
The balancing of the stratified selection swaps single games for other games of the same group: it changes which
games of a group are selected, not how many, so the chances per group stay the same.

Outputs (in the folder of the selection)
tables/groups.csv          gets the columns prob and weight (whole selection), prob_train, weight_train, prob_test
                           and weight_test
games/games.csv            gets the columns prob and weight (whole selection), part_prob and part_weight (the game's
                           part)
logs/probabilities.log
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA, GAMES_SUB, LOGS_SUB, STRATIFIED_DIR, TABLES_SUB, setup_log

log = logging.getLogger("probabilities")
PARTS = ("train", "test")
GAME_COLUMNS = ("prob", "weight", "part_prob", "part_weight")
GROUP_COLUMNS = ("prob", "weight") + tuple(f"{c}_{p}" for p in PARTS for c in ("prob", "weight"))


def group_probabilities(groups: pd.DataFrame, method: str) -> pd.DataFrame:
    """Chance and weight per group for the whole selection and for every part, from the month's games per group
    (`games`) and the selected ones (`games_train`, `games_test`)."""
    month = groups["games"]
    out = groups[["speed", "mean_elo"]].copy()
    out["games_month"] = month
    selected = {"": groups["games_train"] + groups["games_test"], **{f"_{p}": groups[f"games_{p}"] for p in PARTS}}
    for suffix, n in selected.items():
        if method == "stratified":
            prob = n / month.replace(0, np.nan)
        elif method == "random":
            prob = pd.Series(n.sum() / month.sum(), index=groups.index)
        else:
            raise ValueError(f"unknown selection method {method!r}")
        out[f"games{suffix or '_selected'}"] = n
        out[f"prob{suffix}"] = prob
        out[f"weight{suffix}"] = 1 / prob.where(prob > 0)
    return out


def run(folder: Path = STRATIFIED_DIR) -> None:
    games_dir, tables_dir, logs_dir = folder / GAMES_SUB, folder / TABLES_SUB, folder / LOGS_SUB
    meta = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    setup_log(logs_dir / "probabilities.log")
    groups = pd.read_csv(tables_dir / "groups.csv", index_col="group")
    groups = groups.drop(columns=[c for c in GROUP_COLUMNS if c in groups])     # from an earlier run
    games = pd.read_csv(games_dir / "games.csv")
    games = games.drop(columns=[c for c in GAME_COLUMNS if c in games])

    counts = games.groupby(["group", "split"]).size().unstack(fill_value=0).reindex(groups.index, fill_value=0)
    for p in PARTS:
        if not (counts.get(p, 0) == groups[f"games_{p}"]).all():
            raise ValueError(f"games.csv and groups.csv disagree on the {p} games per group: run the selection again")

    pg = group_probabilities(groups, meta["method"])
    groups.join(pg[list(GROUP_COLUMNS)]).to_csv(tables_dir / "groups.csv")
    games["prob"] = games["group"].map(pg["prob"])
    games["weight"] = games["group"].map(pg["weight"])
    for p in PARTS:
        on = games["split"] == p
        games.loc[on, "part_prob"] = games.loc[on, "group"].map(pg[f"prob_{p}"])
        games.loc[on, "part_weight"] = games.loc[on, "group"].map(pg[f"weight_{p}"])
    games.to_csv(games_dir / "games.csv", index=False)

    month = int(pg["games_month"].sum())
    log.info("%s (%s selection): %s games, the month %s complete games", folder.name, meta["method"],
             f"{len(games):,}", f"{month:,}")
    for name, w in [("whole selection", games["weight"])] + [
            (f"{p} part", games.loc[games["split"] == p, "part_weight"]) for p in PARTS]:
        log.info("%-16s weights add up to %s of the month's games (%.4f%%), from %.2f to %.2f per game", name,
                 f"{w.sum():,.0f}", 100 * w.sum() / month, w.min(), w.max())
    empty = pg[(pg["games_selected"] == 0) & (pg["games_month"] > 0)]
    for g, r in empty.iterrows():
        log.info("group %d (%s, %s): %s games in the month, none selected", g, r["speed"], r["mean_elo"],
                 f"{int(r['games_month']):,}")


if __name__ == "__main__":
    run(DATA / sys.argv[1] if len(sys.argv) > 1 else STRATIFIED_DIR)
