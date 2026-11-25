"""Step 1.2: one pass over the month. Keeps the complete cases, counts the moves and blunders of every bucket and
group, and keeps a share of the games as the pool that the selections draw from.

Usage:  uv run python -m src.01_selection.cleanup
        IE500_MONTH=2026-08 uv run python -m src.01_selection.cleanup --no-pool   (August)
Output: data/01_selection/<month>/: cleanup.json, funnel.csv, groups.csv, buckets.csv;
        data/01_selection/work/: pool.parquet, pool_buckets.parquet
"""
import argparse
import json
import logging
import time
from collections import Counter
from compression import zstd
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (CFG, MONTH, MONTH_DIR, MONTH_NAMES, POOL_DIR, SEED, SELECTION_ROOT, WORKERS, log_path,
                        setup_log)
from .storage import MonthFile
from src.common.buckets import (B_OFF, N_B, N_E, N_G, N_W, SPEEDS, WIN, elo_label, game_group, move_buckets,
                                uniform)
from .filters import FILTERS, STEPS, complete_game, split_games

log = logging.getLogger("cleanup")
SEL = CFG["selection"]
RATE = SEL["pool_rate"]
POOL_SCHEMA = pa.schema([("game_id", pa.string()), ("speed", pa.string()), ("white_elo", pa.int32()),
                         ("black_elo", pa.int32()), ("group", pa.int16()), ("plies", pa.int16()),
                         ("u", pa.float64()), ("pgn", pa.string())])


def clock_label(sec: int) -> str:
    return f"{sec // 60}:{sec % 60:02d}"


# Speed, rating and time left of every bucket
def bucket_table() -> pd.DataFrame:
    rows = []
    for s, sp in enumerate(SPEEDS):
        for e in range(N_E):
            for w in range(N_W[s]):
                lo = clock_label(w * WIN)
                if w == N_W[s] - 1:
                    time_left = f"{lo}+"
                else:
                    time_left = f"{lo}-{clock_label((w + 1) * WIN)}"
                rows.append({"bucket": B_OFF[s] + e * N_W[s] + w, "speed": sp, "mover_elo": elo_label(e),
                             "time_left": time_left})
    return pd.DataFrame(rows).set_index("bucket")


# Speed and rating band of every group
def group_table() -> pd.DataFrame:
    rows = []
    for s, sp in enumerate(SPEEDS):
        for e in range(N_E):
            rows.append({"group": s * N_E + e, "speed": sp, "mean_elo": elo_label(e)})
    return pd.DataFrame(rows).set_index("group")


def cat(xs: list, dt=np.int64) -> np.ndarray:
    if xs:
        return np.concatenate(xs).astype(dt)
    return np.empty(0, dtype=dt)


def clean_text(text: str, rate: float) -> dict:
    """Count the complete games in a piece of text and keep the pool games (a share `rate`, chosen by game id).

    Returns the funnel and, per bucket of the month: moves, blunders, games (with a move in it); per group: games,
    moves, blunders (group_*); the pool games (pool) and, for each pool game and bucket it has a move in, one entry:
    the game's row in the pool (rows), the bucket, and its moves and blunders there (b_moves, b_blunders)."""
    funnel = Counter()
    # the month, per bucket: the bucket of every move and of every blunder, and the buckets of every game
    bucket_of_move, bucket_of_blunder, buckets_of_game = [], [], []
    # the month, per game: its group, moves and blunders
    group_of_game, moves_of_game, blunders_of_game = [], [], []
    # the pool: its games, and their entries (see Entries in sampling.py)
    pool, entry_row, entry_bucket, entry_moves, entry_blunders = [], [], [], [], []
    for chunk in split_games(text):
        game = complete_game(chunk, funnel)
        if game is None:
            continue
        buckets = move_buckets(game.speed, game.white_elo, game.black_elo, game.clk)
        own_buckets, which = np.unique(buckets, return_inverse=True)     # which: the position of each move's bucket
        bucket_of_move.append(buckets)
        bucket_of_blunder.append(buckets[game.blunder])
        buckets_of_game.append(own_buckets)
        group = game_group(game.speed, game.white_elo, game.black_elo)
        group_of_game.append(group)
        moves_of_game.append(len(game.clk))
        blunders_of_game.append(int(game.blunder.sum()))
        u = uniform(game.game_id)
        if u < rate:
            entry_row.append(np.full(len(own_buckets), len(pool), dtype=np.int32))
            pool.append((game.game_id, game.speed, game.white_elo, game.black_elo, group, len(game.clk), u,
                         chunk.rstrip("\n") + "\n\n"))
            entry_bucket.append(own_buckets)
            entry_moves.append(np.bincount(which))
            entry_blunders.append(np.bincount(which, weights=game.blunder))
    group_of_game = np.array(group_of_game, dtype=np.int64)
    moves_of_game = np.array(moves_of_game, dtype=float)
    blunders_of_game = np.array(blunders_of_game, dtype=float)
    out = {"funnel": funnel}
    out["moves"] = np.bincount(cat(bucket_of_move), minlength=N_B)
    out["blunders"] = np.bincount(cat(bucket_of_blunder), minlength=N_B)
    out["games"] = np.bincount(cat(buckets_of_game), minlength=N_B)
    out["group_games"] = np.bincount(group_of_game, minlength=N_G)
    out["group_moves"] = np.bincount(group_of_game, weights=moves_of_game, minlength=N_G).astype(np.int64)
    out["group_blunders"] = np.bincount(group_of_game, weights=blunders_of_game, minlength=N_G).astype(np.int64)
    out["pool"] = pool
    out["rows"] = cat(entry_row, np.int32)
    out["bucket"] = cat(entry_bucket, np.int16)
    out["b_moves"] = cat(entry_moves, np.int16)
    out["b_blunders"] = cat(entry_blunders, np.int16)
    return out


# Clean the games of one block of the file
def clean_frame(frame: bytes, rate: float) -> dict:
    data = zstd.decompress(frame)
    first = data.find(b"\n[Event ") + 1
    last = data.rfind(b"\n[Event ") + 1
    if not 0 < first < last:
        raise ValueError("a frame holds fewer than two game starts")
    out = clean_text(data[first:last].decode("utf-8", "replace"), rate)
    out.update(head=data[:first], tail=data[last:], nbytes=len(frame))
    return out


SUMS = ("moves", "blunders", "games", "group_games", "group_moves", "group_blunders")


# Totals over all blocks
class Total:

    def __init__(self, path):
        self.funnel = Counter()
        self.arr = dict.fromkeys(SUMS, 0)
        self.rows = []
        self.bucket = []
        self.b_moves = []
        self.b_blunders = []
        self.n_pool = 0
        self.pending = []
        self.writer = None
        if path is not None:
            self.writer = pq.ParquetWriter(path, POOL_SCHEMA, compression="zstd")

    def add(self, r: dict) -> None:
        self.funnel.update(r["funnel"])
        for k in SUMS:
            self.arr[k] = self.arr[k] + r[k]
        self.rows.append(r["rows"] + np.int32(self.n_pool))
        self.bucket.append(r["bucket"])
        self.b_moves.append(r["b_moves"])
        self.b_blunders.append(r["b_blunders"])
        self.pending += r["pool"]
        self.n_pool += len(r["pool"])
        if len(self.pending) >= 20_000:
            self.flush()

    def flush(self) -> None:
        if self.pending and self.writer is not None:
            columns = {}
            for i, name in enumerate(POOL_SCHEMA.names):
                columns[name] = [game[i] for game in self.pending]
            self.writer.write_table(pa.table(columns, schema=POOL_SCHEMA))
            self.pending = []

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()


# Clean the month (max_frames: only the first blocks, for a quick test)
def run(pool: bool = True, max_frames: int | None = None) -> None:
    rate = RATE if pool else 0.0
    MONTH_DIR.mkdir(parents=True, exist_ok=True)
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    (MONTH_DIR / "cleanup.json").unlink(missing_ok=True)
    setup_log(log_path(SELECTION_ROOT, f"cleanup_{MONTH_NAMES[MONTH]}"))
    t0 = time.time()
    month = MonthFile(MONTH)
    log.info("%s: %.1f GB, %d processes, pool rate %.2f", month.location, month.size / 1e9, WORKERS, rate)
    tot = Total(POOL_DIR / "pool.parquet" if pool else None)
    clean = partial(clean_frame, rate=rate)
    tail = b""
    done = 0
    with ProcessPoolExecutor(WORKERS) as workers:
        for k, r in enumerate(workers.map(clean, month.frames(max_frames), buffersize=2 * WORKERS)):
            tot.add(clean_text((tail + r["head"]).decode("utf-8", "replace"), rate))
            tot.add(r)
            tail = r["tail"]
            done += r["nbytes"]
            if (k + 1) % 250 == 0:
                el = time.time() - t0
                log.info("%5.1f of %.1f GB, %10s complete games, %9s pool games, %5.1f min, about %.0f min left",
                         done / 1e9, month.size / 1e9, f"{int(tot.arr['group_games'].sum()):,}", f"{tot.n_pool:,}",
                         el / 60, el / done * (month.size - done) / 60)
    tot.add(clean_text(tail.decode("utf-8", "replace"), rate))
    tot.close()
    write(tot, month, rate, time.time() - t0)


def write(tot: Total, month: MonthFile, rate: float, seconds: float) -> None:
    a = tot.arr
    bt = bucket_table()
    bt["moves"] = a["moves"]
    bt["blunders"] = a["blunders"]
    bt["games"] = a["games"]
    bt["rate"] = bt["blunders"] / bt["moves"].replace(0, np.nan)
    bt["share"] = bt["moves"] / bt["moves"].sum()
    bt["few_games"] = bt["games"] < SEL["min_games"]
    bt.to_csv(MONTH_DIR / "buckets.csv")
    gt = group_table()
    gt["games"] = a["group_games"]
    gt["moves"] = a["group_moves"]
    gt["blunders"] = a["group_blunders"]
    gt["share"] = gt["games"] / gt["games"].sum()
    gt["rate"] = gt["blunders"] / gt["moves"].replace(0, np.nan)
    gt.to_csv(MONTH_DIR / "groups.csv")
    rows = []
    for (step, sp), n in tot.funnel.items():
        rows.append((step, sp, n))
    funnel = pd.DataFrame(rows, columns=["step", "speed", "games"])
    funnel = funnel.pivot(index="step", columns="speed", values="games")
    cols = [c for c in SPEEDS + ["other"] if c in funnel]
    funnel = funnel.reindex(index=STEPS, columns=cols).fillna(0).astype(np.int64)
    funnel["all"] = funnel.sum(axis=1)
    funnel.to_csv(MONTH_DIR / "funnel.csv")
    if tot.writer is not None:
        table = pa.table({"row": np.concatenate(tot.rows), "bucket": np.concatenate(tot.bucket),
                          "moves": np.concatenate(tot.b_moves), "blunders": np.concatenate(tot.b_blunders)})
        pq.write_table(table, POOL_DIR / "pool_buckets.parquet", compression="zstd")
    meta = {"month": MONTH, "source": month.location, "source_bytes": month.size,
            "source_sha256": month.sha256, "whole_file_read": month.sha256 is not None,
            "seed": SEED, "filters": FILTERS, "pool_rate": rate, "n_complete_games": int(gt["games"].sum()),
            "n_moves": int(bt["moves"].sum()), "n_blunders": int(bt["blunders"].sum()),
            "blunder_rate": float(bt["blunders"].sum() / bt["moves"].sum()), "n_pool": tot.n_pool,
            "minutes": round(seconds / 60, 1), "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (MONTH_DIR / "cleanup.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("cleanup: %s", meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-pool", action="store_true", help="only count, keep no pool")
    args = ap.parse_args()
    run(pool=not args.no_pool)
