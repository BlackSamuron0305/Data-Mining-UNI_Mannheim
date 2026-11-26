"""Write a selection to data/01_selection/<name>/: the games (games.csv, train.pgn.zst, test.pgn.zst), the group
and bucket tables (groups.csv, buckets.csv), selection.json and summary.md."""
import json
from compression import zstd
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import CFG, MONTH, POOL_DIR, SEED, SELECTION_ROOT
from .draw import TEST_FRAC

N = CFG["selection"]["n_games"]
MIN_GAMES = CFG["selection"]["min_games"]
PARTS = ("train", "test")


def make_games(pool, train: np.ndarray, test: np.ndarray) -> pd.DataFrame:
    ids = np.concatenate([train, test])
    games = pool.games.loc[ids, ["game_id", "speed", "white_elo", "black_elo", "group", "plies"]]
    games = games.reset_index(drop=True)
    games.insert(1, "split", ["train"] * len(train) + ["test"] * len(test))
    games.insert(3, "mean_elo", pool.groups["mean_elo"].to_numpy()[games["group"].to_numpy()])
    return games


# Copy the PGN text of the selected games into train and test files
def write_pgn(out, games: pd.DataFrame) -> None:
    part_of = dict(zip(games["game_id"], games["split"]))
    found = {"train": 0, "test": 0}
    with zstd.open(out / "train.pgn.zst", "wb", level=12) as train_file, \
            zstd.open(out / "test.pgn.zst", "wb", level=12) as test_file:
        files = {"train": train_file, "test": test_file}
        parquet = pq.ParquetFile(POOL_DIR / "pool.parquet")
        for batch in parquet.iter_batches(20_000, columns=["game_id", "pgn"]):
            for game_id, pgn in zip(batch.column("game_id").to_pylist(), batch.column("pgn").to_pylist()):
                part = part_of.get(game_id)
                if part:
                    files[part].write(pgn.encode("utf-8"))
                    found[part] += 1
    expected = games["split"].value_counts().to_dict()
    for p in PARTS:
        if found[p] != expected.get(p, 0):
            raise ValueError(f"PGN text found for {found} games, expected {expected}")


# Average gap between the blunder rates of a part and of the month
def bucket_error(b: pd.DataFrame, p: str) -> float:
    b = b[b[f"moves_{p}"] > 0]
    return 100 * float((b["moves"] * (b[f"rate_{p}"] - b["rate"]).abs()).sum() / b["moves"].sum())


def percent(x: float) -> str:
    if 0 < x < 0.0001:
        return "<0.01"
    return f"{100 * x:.2f}"


def group_table(pool, games: pd.DataFrame) -> pd.DataFrame:
    g = pool.groups[["speed", "mean_elo", "games", "share"]].copy()
    for p in PARTS:
        counts = games[games["split"] == p].groupby("group").size().reindex(g.index, fill_value=0)
        g[f"games_{p}"] = counts
        g[f"share_{p}"] = counts / counts.sum()
        part = games[games["split"] == p]
        g[f"prob_{p}"] = part.groupby("group")["part_prob"].first().reindex(g.index)
        g[f"weight_{p}"] = 1 / g[f"prob_{p}"]
    g["prob"] = games.groupby("group")["prob"].first().reindex(g.index)
    g["weight"] = 1 / g["prob"]
    return g


def bucket_table(pool, part_ids: dict) -> pd.DataFrame:
    b = pool.buckets[["speed", "mover_elo", "time_left", "moves", "blunders", "games", "rate", "share",
                      "few_games"]].copy()
    for p, ids in part_ids.items():
        moves, blunders, n_games = pool.entries.totals(ids)
        b[f"moves_{p}"] = moves.astype(np.int64)
        b[f"blunders_{p}"] = blunders.astype(np.int64)
        b[f"games_{p}"] = n_games
        b[f"rate_{p}"] = blunders / np.where(moves > 0, moves, np.nan)
        b[f"share_{p}"] = moves / moves.sum()
        b[f"few_games_{p}"] = n_games < MIN_GAMES
    return b


def summary(name: str, pool, g: pd.DataFrame, n: dict, moves: dict, rate: dict, err: dict, swapped: dict) -> str:
    month = pool.meta
    lines = [f"# Lichess {MONTH}: {N:,} games, {name} selection", "",
             f"Drawn from every complete game of the month: {month['n_complete_games']:,} games, "
             f"{month['n_moves']:,} moves, blunder probability {100 * month['blunder_rate']:.3f}%.", "",
             "| | training | test |", "|:--|--:|--:|",
             f"| games | {n['train']:,} | {n['test']:,} |",
             f"| moves | {moves['train']:,} | {moves['test']:,} |",
             f"| blunder probability (%) | {100 * rate['train']:.3f} | {100 * rate['test']:.3f} |",
             f"| games swapped by the balancing | {swapped['train']:,} | {swapped['test']:,} |",
             f"| blunder probability per bucket, mean difference to the month (points) | {err['train']:.3f} | "
             f"{err['test']:.3f} |", "",
             "## Games per group (speed x Elo of the mean rating), share in %", "",
             "| speed | Elo | games in the month | share | training games | share | test games | share | weight |",
             "|:--|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for _, r in g.iterrows():
        lines.append(f"| {r['speed']} | {r['mean_elo']} | {int(r['games']):,} | {percent(r['share'])} | "
                     f"{int(r['games_train']):,} | {percent(r['share_train'])} | {int(r['games_test']):,} | "
                     f"{percent(r['share_test'])} | {r['weight']:.1f} |")
    return "\n".join(lines) + "\n"


# Write games, tables and summary into data/01_selection/<name>/
def write_selection(name: str, pool, games: pd.DataFrame, part_ids: dict, balance: dict | None,
                    seconds: float) -> None:
    out = SELECTION_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    games.sort_values("game_id", ignore_index=True).to_csv(out / "games.csv", index=False)
    write_pgn(out, games)
    g = group_table(pool, games)
    g.to_csv(out / "groups.csv")
    b = bucket_table(pool, part_ids)
    b.to_csv(out / "buckets.csv")

    n = {p: int((games["split"] == p).sum()) for p in PARTS}
    moves = {p: int(b[f"moves_{p}"].sum()) for p in PARTS}
    rate = {p: float(b[f"blunders_{p}"].sum() / moves[p]) for p in PARTS}
    err = {p: bucket_error(b, p) for p in PARTS}
    swapped = {p: 0 for p in PARTS}
    if balance:
        for p in balance:
            swapped[p] = balance[p]["games_replaced"]
    meta = {"method": name, "month": MONTH, "n_games": N, "test_frac": TEST_FRAC, "seed": SEED, "games": n,
            "moves": moves, "blunder_rate": rate, "bucket_error_pp": err, "balance": balance, "cleanup": pool.meta,
            "minutes": round(seconds / 60, 1), "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "selection.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(summary(name, pool, g, n, moves, rate, err, swapped), encoding="utf-8")
