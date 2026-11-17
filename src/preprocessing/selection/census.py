"""Selection step 1 - read every game of the month once and count the complete games.

Usage:  uv run python -m src.preprocessing.selection.census

The monthly file is read straight from the team's S3 storage (src/lichess/storage.py), one zstd frame of about
14,000 games at a time, and `workers` processes decompress and parse the frames in parallel. The raw file is
not changed, and nothing of it is stored apart from the candidate games below.

Complete case (src/lichess/pgn.py): a rated standard game of speed bullet to classical, both ratings known, no
computer player, normal end or time forfeit, at least 20 plies, a clock and an engine evaluation on every ply.

Two kinds of buckets:
- a move falls into speed x Elo of the mover x the mover's time left before the move (30 s steps up to
  `window_max` of the speed, the last step open). The blunder probability (Lichess mark ??) per bucket is what
  the selected games have to reproduce;
- a game falls into one group: speed x Elo of the mean of both ratings. A game passes through many time
  steps and has two players, so only these groups of games can be used to draw whole games.

Every complete game gets a fixed pseudo-random number u in [0, 1) from the seed and its game id. The games with
u < `candidate_rate` are the candidates: their counts per bucket and their original PGN text are kept, so that
the selection (stratified_selection.py) never has to read the 29 GB file again.

Outputs (cache/census/, working files of the selection, not part of data/)
buckets.csv              moves, blunders, games and blunder probability per bucket of moves, whole month
groups.csv               games, moves and blunders per group, whole month
funnel.csv               games by speed after every filter step
time_controls.csv        complete games by speed and time control
candidates.parquet       game id, speed, ratings, group, plies, u and the PGN text of every candidate game
candidates_buckets.npz   row (into the parquet), bucket, moves and blunders of the candidate games
census.json              totals, source file and its sha256, run time; written last, so it marks a finished census
census.log
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from compression import zstd
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import CENSUS_DIR, CFG, MONTH, SEED, WORKERS, setup_log
from src.lichess.pgn import STEPS, complete_game, split_games
from src.lichess.storage import MonthFile

log = logging.getLogger("census")
SEL = CFG["selection"]
SPEEDS = CFG["filters"]["speeds"]
N_S = len(SPEEDS)
ELO_BIN, ELO_LO, ELO_HI = SEL["elo_bin"], SEL["elo_min"], SEL["elo_max"]
N_E = (ELO_HI - ELO_LO) // ELO_BIN + 1          # Elo bins: below elo_min + elo_bin, ..., elo_max and above
WIN = SEL["window"]
N_W = np.array([SEL["window_max"][s] // WIN + 1 for s in SPEEDS])    # time steps per speed, the last one open
B_OFF = np.concatenate(([0], np.cumsum(N_E * N_W)))                   # first bucket of every speed
N_B = int(B_OFF[-1])                                                  # buckets of moves
N_G = N_S * N_E                                                       # groups of games
RATE = SEL["candidate_rate"]
CAND_SCHEMA = pa.schema([("game_id", pa.string()), ("speed", pa.string()), ("white_elo", pa.int32()),
                         ("black_elo", pa.int32()), ("group", pa.int16()), ("plies", pa.int16()),
                         ("u", pa.float64()), ("pgn", pa.string())])


# ----------------------------------------------------------------------------------------------------- labels
def elo_label(e: int) -> str:
    lo = ELO_LO + ELO_BIN * e
    if e == 0:
        return f"<{lo + ELO_BIN}"
    return f"{lo}+" if e == N_E - 1 else f"{lo}-{lo + ELO_BIN}"


def clock_label(sec: int) -> str:
    return f"{sec // 60}:{sec % 60:02d}"


def bucket_table() -> pd.DataFrame:
    """Speed, Elo of the mover and time left of every bucket of moves, in bucket order."""
    rows = []
    for s, sp in enumerate(SPEEDS):
        for e in range(N_E):
            for w in range(N_W[s]):
                lo = clock_label(w * WIN)
                rows.append({"bucket": B_OFF[s] + e * N_W[s] + w, "speed": sp, "mover_elo": elo_label(e),
                             "time_left": f"{lo}+" if w == N_W[s] - 1 else f"{lo}-{clock_label((w + 1) * WIN)}"})
    return pd.DataFrame(rows).set_index("bucket")


def group_table() -> pd.DataFrame:
    """Speed and Elo of the mean rating of every group of games, in group order."""
    return pd.DataFrame([{"group": s * N_E + e, "speed": sp, "mean_elo": elo_label(e)}
                         for s, sp in enumerate(SPEEDS) for e in range(N_E)]).set_index("group")


# ----------------------------------------------------------------------------------------------------- census
def elo_bin(elo) -> np.ndarray:
    return np.clip((np.asarray(elo) - ELO_LO) // ELO_BIN, 0, N_E - 1).astype(np.int64)


def uniform(game_id: str, salt: str = "ie500-strata") -> float:
    """A fixed pseudo-random number in [0, 1) of a game, from the seed and the game id. The salt keeps its old
    text: a new one would give every game a new number and so a different selection."""
    h = hashlib.shake_256(f"{salt}:{SEED}:{game_id}".encode("utf-8")).digest(8)
    return int.from_bytes(h, "little") / 2.0 ** 64


def census_text(text: str) -> dict:
    """Counts of the complete games in `text` (which must hold whole games) and the candidates among them."""
    funnel, tcs = Counter(), Counter()
    b_moves, b_blunders, b_games, groups, g_moves, g_blunders = [], [], [], [], [], []
    cand, c_rows, c_bucket, c_moves, c_blunders = [], [], [], [], []
    for chunk in split_games(text):
        g = complete_game(chunk, funnel)
        if g is None:
            continue
        n, s = len(g.clk), SPEEDS.index(g.speed)
        tcs[(g.speed, g.time_control)] += 1
        white = np.arange(n) % 2 == 0
        elo = elo_bin(np.where(white, g.white_elo, g.black_elo))
        own = np.concatenate((g.clk[:2], g.clk[:-2]))          # the mover's clock before the move
        b = B_OFF[s] + elo * N_W[s] + np.minimum(own // WIN, N_W[s] - 1)
        ub, inv = np.unique(b, return_inverse=True)
        b_moves.append(b)
        b_blunders.append(b[g.blunder])
        b_games.append(ub)
        h = s * N_E + int(elo_bin((g.white_elo + g.black_elo) / 2))
        groups.append(h)
        g_moves.append(n)
        g_blunders.append(int(g.blunder.sum()))
        u = uniform(g.game_id)
        if u < RATE:
            c_rows.append(np.full(len(ub), len(cand), dtype=np.int32))
            cand.append((g.game_id, g.speed, g.white_elo, g.black_elo, h, n, u, chunk.rstrip("\n") + "\n\n"))
            c_bucket.append(ub)
            c_moves.append(np.bincount(inv))
            c_blunders.append(np.bincount(inv, weights=g.blunder))
    cat = lambda xs, dt=np.int64: np.concatenate(xs).astype(dt) if xs else np.empty(0, dtype=dt)
    groups = np.array(groups, dtype=np.int64)
    return {"funnel": funnel, "tcs": tcs,
            "moves": np.bincount(cat(b_moves), minlength=N_B), "blunders": np.bincount(cat(b_blunders), minlength=N_B),
            "games": np.bincount(cat(b_games), minlength=N_B),
            "group_games": np.bincount(groups, minlength=N_G),
            "group_moves": np.bincount(groups, weights=np.array(g_moves, dtype=float), minlength=N_G).astype(np.int64),
            "group_blunders": np.bincount(groups, weights=np.array(g_blunders, dtype=float),
                                          minlength=N_G).astype(np.int64),
            "cand": cand, "rows": cat(c_rows, np.int32), "bucket": cat(c_bucket, np.int16),
            "b_moves": cat(c_moves, np.int16), "b_blunders": cat(c_blunders, np.int16)}


def census_frame(frame: bytes) -> dict:
    """Census of the games that start in one zstd frame. The text before the first game start and from the
    start of the last game on is returned raw, to be joined with the neighbouring frames."""
    data = zstd.decompress(frame)
    first, last = data.find(b"\n[Event ") + 1, data.rfind(b"\n[Event ") + 1
    if not 0 < first < last:
        raise ValueError("a frame holds fewer than two game starts")
    out = census_text(data[first:last].decode("utf-8", "replace"))
    out.update(head=data[:first], tail=data[last:], nbytes=len(frame))
    return out


SUMS = ("moves", "blunders", "games", "group_games", "group_moves", "group_blunders")


class Total:
    """Running sums over the frames; the candidates go to the parquet file as they come."""

    def __init__(self, path):
        self.funnel, self.tcs, self.arr = Counter(), Counter(), dict.fromkeys(SUMS, 0)
        self.rows, self.bucket, self.b_moves, self.b_blunders = [], [], [], []
        self.n_cand, self.pending = 0, []
        self.writer = pq.ParquetWriter(path, CAND_SCHEMA, compression="zstd")

    def add(self, r: dict) -> None:
        self.funnel.update(r["funnel"])
        self.tcs.update(r["tcs"])
        for k in SUMS:
            self.arr[k] = self.arr[k] + r[k]
        self.rows.append(r["rows"] + np.int32(self.n_cand))
        self.bucket.append(r["bucket"])
        self.b_moves.append(r["b_moves"])
        self.b_blunders.append(r["b_blunders"])
        self.pending += r["cand"]
        self.n_cand += len(r["cand"])
        if len(self.pending) >= 20_000:
            self.flush()

    def flush(self) -> None:
        if self.pending:
            self.writer.write_table(pa.table(dict(zip(CAND_SCHEMA.names, map(list, zip(*self.pending)))),
                                             schema=CAND_SCHEMA))
            self.pending = []

    def close(self) -> None:
        self.flush()
        self.writer.close()


def run(max_frames: int | None = None) -> None:
    """The census of the month; `max_frames` reads only the first frames (a quick test, not a census)."""
    CENSUS_DIR.mkdir(parents=True, exist_ok=True)
    (CENSUS_DIR / "census.json").unlink(missing_ok=True)       # absent until the census has finished
    setup_log(CENSUS_DIR / "census.log")
    t0 = time.time()
    month = MonthFile(MONTH)
    log.info("%s: %.1f GB, %d processes, candidate rate %.2f", month.location, month.size / 1e9, WORKERS, RATE)
    tot, tail, done = Total(CENSUS_DIR / "candidates.parquet"), b"", 0
    with ProcessPoolExecutor(WORKERS) as pool:
        for k, r in enumerate(pool.map(census_frame, month.frames(max_frames), buffersize=2 * WORKERS)):
            tot.add(census_text((tail + r["head"]).decode("utf-8", "replace")))      # the game across the border
            tot.add(r)
            tail, done = r["tail"], done + r["nbytes"]
            if (k + 1) % 250 == 0:
                el = time.time() - t0
                log.info("%5.1f of %.1f GB, %10s complete games, %9s candidates, %5.1f min, about %.0f min left",
                         done / 1e9, month.size / 1e9, f"{int(tot.arr['group_games'].sum()):,}", f"{tot.n_cand:,}",
                         el / 60, el / done * (month.size - done) / 60)
    tot.add(census_text(tail.decode("utf-8", "replace")))
    tot.close()
    write(tot, month, time.time() - t0)


def write(tot: Total, month: MonthFile, seconds: float) -> None:
    a = tot.arr
    bt = bucket_table()
    bt["moves"], bt["blunders"], bt["games"] = a["moves"], a["blunders"], a["games"]
    bt["rate"] = bt["blunders"] / bt["moves"].replace(0, np.nan)
    bt["share"] = bt["moves"] / bt["moves"].sum()
    bt["few_games"] = bt["games"] < SEL["min_games"]
    bt.to_csv(CENSUS_DIR / "buckets.csv")
    gt = group_table()
    gt["games"], gt["moves"], gt["blunders"] = a["group_games"], a["group_moves"], a["group_blunders"]
    gt["share"] = gt["games"] / gt["games"].sum()
    gt["rate"] = gt["blunders"] / gt["moves"].replace(0, np.nan)
    gt.to_csv(CENSUS_DIR / "groups.csv")
    rows = [(step, sp, n) for (step, sp), n in tot.funnel.items()]
    funnel = pd.DataFrame(rows, columns=["step", "speed", "games"]).pivot(index="step", columns="speed", values="games")
    funnel = funnel.reindex(index=STEPS, columns=[c for c in SPEEDS + ["other"] if c in funnel]).fillna(0).astype(np.int64)
    funnel["all"] = funnel.sum(axis=1)
    funnel.to_csv(CENSUS_DIR / "funnel.csv")
    tc = pd.DataFrame([(sp, t, n) for (sp, t), n in tot.tcs.items()], columns=["speed", "time_control", "games"])
    tc.sort_values(["speed", "games"], ascending=[True, False]).to_csv(CENSUS_DIR / "time_controls.csv", index=False)
    np.savez_compressed(CENSUS_DIR / "candidates_buckets.npz", rows=np.concatenate(tot.rows),
                        bucket=np.concatenate(tot.bucket), moves=np.concatenate(tot.b_moves),
                        blunders=np.concatenate(tot.b_blunders))
    meta = {"month": MONTH, "source": month.location, "source_bytes": month.size,
            "source_sha256": month.sha256, "whole_file_read": month.sha256 is not None,
            "seed": SEED, "candidate_rate": RATE, "n_complete_games": int(gt["games"].sum()),
            "n_moves": int(bt["moves"].sum()), "n_blunders": int(bt["blunders"].sum()),
            "blunder_rate": float(bt["blunders"].sum() / bt["moves"].sum()), "n_candidates": tot.n_cand,
            "minutes": round(seconds / 60, 1), "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (CENSUS_DIR / "census.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("census: %s", meta)


if __name__ == "__main__":
    run()
