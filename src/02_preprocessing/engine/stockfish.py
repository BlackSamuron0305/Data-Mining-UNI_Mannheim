"""Step 2.2: Stockfish on the position before every move (how hard is the choice the mover faces?). Takes about 15
hours on 16 cores; with data.stockfish: download in config.yaml our results are downloaded instead (README.md).

Usage:  uv run python -m src.02_preprocessing.engine.stockfish [train|test ...] [--merge]
Output: data/02_preprocessing/stockfish/train.parquet, test.parquet, engine.json: one row per analysed move, of every
        selection. While it runs: one file per batch of games in data/02_preprocessing/work/stockfish/<part>/.
        IE500_SHARD="i/n" does only every n-th batch; after the last shard, --merge adds the batches to the result.

A game that is already analysed with the same engine and settings is skipped: in the result or in the batches (a
stopped run continues). After a change of the selection only its new games are analysed; results of games that are no
longer selected stay in the files and are left out when the tables are joined.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (CFG, KEYS, PREPROCESSING_ROOT, SELECTION_DIR, STOCKFISH_DIR, WORKERS, env, log_path,
                        setup_log)
from src.common.pgn import capped_cp, iter_pgn, read_game, win_prob
from src.common.buckets import uniform

log = logging.getLogger("stockfish")
PARTS = ("train", "test")
ENG = CFG["engine"]
MARKS = CFG["features"]["marks"]
BATCH_DIR = PREPROCESSING_ROOT / "work" / "stockfish"
TOP = 5
THREADS = 1                  # per engine; one engine per worker process
HASH_MB = 16
BATCH = 50                   # games per output file, so that a stopped run continues where it was


def engine_path() -> str:
    path = env("STOCKFISH_PATH") or "stockfish"
    found = shutil.which(path)
    if not found and Path(path).exists():
        found = path
    if not found:
        raise FileNotFoundError(f"Stockfish not found at {path!r}: install it and set STOCKFISH_PATH in .env")
    return found


def settings() -> dict:
    return {"top": ENG["top"], "worst": ENG["worst"], "threads": THREADS, "hash_mb": HASH_MB, "batch": BATCH,
            "marks": MARKS}


def open_engine() -> chess.engine.SimpleEngine:
    eng = chess.engine.SimpleEngine.popen_uci(engine_path())
    eng.configure({"Threads": THREADS, "Hash": HASH_MB})
    return eng


def ranked(eng: chess.engine.SimpleEngine, board: chess.Board, limit: chess.engine.Limit, k: int,
           gid: str) -> tuple[np.ndarray, chess.Move]:
    """Win probabilities of the mover after the k best moves (best first), and the best move."""
    infos = []
    for info in eng.analyse(board, limit, multipv=k, game=gid):
        if "score" in info and info.get("pv"):
            infos.append(info)
    if not infos:
        raise RuntimeError(f"no score from the engine in game {gid}, position {board.fen()}")
    wps = win_prob([capped_cp(info["score"].pov(board.turn)) for info in infos])
    order = np.argsort(-wps, kind="stable")
    best_move = infos[int(order[0])]["pv"][0]
    return wps[order], best_move


def position_features(top: np.ndarray, every: np.ndarray, n_legal: int) -> dict:
    """The columns of one position from the top pass and the pass over every move (best first)."""
    gap = top[0] - top
    loss = every[0] - every
    row = {"sf_wp_best": top[0], "sf_n_scored": len(every), "sf_n_close": int(np.sum(loss < MARKS["inaccuracy"]))}
    for k in range(2, TOP + 1):
        row[f"sf_gap_{k}"] = gap[k - 1] if len(gap) >= k else np.nan
    for k in range(1, TOP + 1):
        row[f"sf_worst_{k}"] = loss[-k] if len(loss) >= k else np.nan
    for name, threshold in MARKS.items():
        if len(every) == n_legal:
            row[f"sf_share_{name}"] = float(np.sum(loss >= threshold)) / n_legal
        else:
            row[f"sf_share_{name}"] = np.nan
    return row


def analyse_game(eng: chess.engine.SimpleEngine, pgn: str) -> list[dict]:
    game = read_game(pgn)
    gid = game.headers.get("Site", "").rsplit("/", 1)[-1]
    top_limit = chess.engine.Limit(**ENG["top"]["limit"])
    worst_limit = chess.engine.Limit(**ENG["worst"]["limit"])
    board = game.board()
    rows = []
    for t, node in enumerate(game.mainline()):
        n_legal = board.legal_moves.count()
        top, best = ranked(eng, board, top_limit, min(int(ENG["top"]["lines"]), n_legal), gid)
        every, _ = ranked(eng, board, worst_limit, n_legal, gid)
        rows.append({"game_id": gid, "ply": t, **position_features(top, every, n_legal), "sf_best_move": best.uci()})
        board.push(node.move)
    return rows


def analyse_texts(texts) -> pd.DataFrame:
    """The engine columns of the games in `texts` (PGN text of one game each)."""
    # one engine per batch: an engine left open would keep the worker process from exiting
    eng = open_engine()
    rows = []
    try:
        for pgn in texts:
            rows.extend(analyse_game(eng, pgn))
    finally:
        eng.quit()
    df = pd.DataFrame(rows)
    for col in df.columns:
        if col.startswith("sf_") and col != "sf_best_move":
            df[col] = df[col].astype(np.float32)
    df["ply"] = df["ply"].astype(np.int16)
    return df


def analyse_batch(job: tuple[str, tuple[str, ...]]) -> tuple[int, int, float]:
    """Analyse a batch of games and write its file; returns games, positions, seconds."""
    out = Path(job[0])
    texts = job[1]
    t0 = time.time()
    df = analyse_texts(texts)
    tmp = out.with_suffix(".part")
    df.to_parquet(tmp, index=False, compression="zstd")
    # on Windows a virus scanner may hold the new file for a moment
    for attempt in range(30):
        try:
            tmp.replace(out)
            break
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(2)
    return len(texts), len(df), time.time() - t0


def chosen_games(part: str) -> set[str]:
    """The games of the part to analyse: all, or a fixed random subset of `engine.games` (not set: all)."""
    games = pd.read_csv(SELECTION_DIR / "games.csv", usecols=["game_id", "split"])
    ids = games.loc[games["split"] == part, "game_id"].tolist()
    n = (ENG.get("games") or {}).get(part)
    if n is None or n >= len(ids):
        return set(ids)
    ids.sort(key=lambda g: uniform(g, "ie500-engine"))
    return set(ids[:n])


def shard() -> tuple[int, int]:
    """(i, n) from IE500_SHARD="i/n": do only the batches k with k % n == i. Unset = (0, 1)."""
    i, _, n = (os.environ.get("IE500_SHARD") or "0/1").partition("/")
    i = int(i)
    n = int(n or 1)
    if not 0 <= i < n:
        raise ValueError(f"IE500_SHARD must be i/n with 0 <= i < n, not {os.environ.get('IE500_SHARD')!r}")
    return i, n


def game_id_of(pgn: str) -> str:
    start = pgn.find('[Site "')
    if start == -1:
        return ""
    site = pgn[start + 7:].split('"', 1)[0]
    return site.rsplit("/", 1)[-1]


def next_number(out_dir: Path) -> int:
    """The number of the next batch file: after every numbered file already there."""
    numbers = [int(f.stem) for f in out_dir.glob("*.parquet") if f.stem.isdigit()]
    return max(numbers, default=-1) + 1


def jobs(part: str, out_dir: Path, keep: set[str], first: int):
    """(file, games) of every batch of this shard; the files are numbered from `first` on."""
    my_shard, n_shards = shard()
    batch = []
    k = 0
    for pgn in iter_pgn(SELECTION_DIR / f"{part}.pgn.zst"):
        if game_id_of(pgn) not in keep:
            continue
        batch.append(pgn)
        if len(batch) == BATCH:
            out = out_dir / f"{first + k:05d}.parquet"
            if k % n_shards == my_shard and not out.exists():
                yield str(out), tuple(batch)
            batch = []
            k += 1
    out = out_dir / f"{first + k:05d}.parquet"
    if batch and k % n_shards == my_shard and not out.exists():
        yield str(out), tuple(batch)


def same_settings(meta: dict, name: str) -> bool:
    return meta.get("engine") == name and {k: meta.get(k) for k in settings()} == settings()


def result_is_reusable(name: str) -> bool:
    """The result in data/02_preprocessing/stockfish/ was made with this engine and these settings."""
    meta_path = STOCKFISH_DIR / "engine.json"
    return meta_path.exists() and same_settings(json.loads(meta_path.read_text(encoding="utf-8")), name)


def batch_files(part: str, name: str) -> list[Path]:
    """The batch files of the part; they must come from this engine and these settings."""
    files = sorted((BATCH_DIR / part).glob("*.parquet"))
    meta_path = BATCH_DIR / part / "engine.json"
    if files and not (meta_path.exists() and same_settings(json.loads(meta_path.read_text(encoding="utf-8")), name)):
        raise RuntimeError(f"{BATCH_DIR / part} holds an analysis with another engine or other settings (or without "
                           "engine.json); delete the folder or restore the settings")
    return files


# Games of the part already analysed with the same engine and settings: in the result or in the batches
def done_before(part: str, name: str) -> set[str]:
    games = set()
    result = STOCKFISH_DIR / f"{part}.parquet"
    if result.exists():
        if result_is_reusable(name):
            games.update(pq.read_table(result, columns=["game_id"]).column("game_id").unique().to_pylist())
        else:
            log.info("%s: %s used another engine or other settings, not reused", part, result)
    for f in batch_files(part, name):
        games.update(pd.read_parquet(f, columns=["game_id"])["game_id"].unique())
    if games:
        log.info("%s: %s games already analysed, skipped", part, f"{len(games):,}")
    return games


def merge(part: str, name: str) -> None:
    """Add the batches of the part to the result (each move once) and empty the batch folder."""
    files = batch_files(part, name)
    if not files:
        return
    result = STOCKFISH_DIR / f"{part}.parquet"
    tables = [pq.read_table(f) for f in files]
    if result.exists() and result_is_reusable(name):
        tables.insert(0, pq.read_table(result))
    df = pa.concat_tables(tables, promote_options="default").to_pandas()
    df = df.drop_duplicates(KEYS).sort_values(KEYS, kind="stable").reset_index(drop=True)
    STOCKFISH_DIR.mkdir(parents=True, exist_ok=True)
    tmp = result.with_suffix(".part")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd", row_group_size=1_000_000)
    tmp.replace(result)
    meta = {"engine": name, **settings()}
    (STOCKFISH_DIR / "engine.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for f in files:
        f.unlink()
    log.info("%s: %s moves of %s games in %s", part, f"{len(df):,}", f"{df['game_id'].nunique():,}", result)


def engine_name() -> str:
    eng = open_engine()
    name = eng.id.get("name", "?")
    eng.quit()
    return name


def run_part(part: str) -> None:
    out_dir = BATCH_DIR / part
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_log(log_path(PREPROCESSING_ROOT, f"stockfish_{part}"))
    name = engine_name()
    chosen = chosen_games(part)
    keep = chosen - done_before(part, name)

    meta_path = out_dir / "engine.json"
    meta = {"engine": name, "part": part, "games": len(chosen), **settings()}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    first = next_number(out_dir)
    my_shard, n_shards = shard()
    log.info("%s: %s, %s of %s games still to analyse, %d workers, shard %d of %d", part, name, f"{len(keep):,}",
             f"{len(chosen):,}", WORKERS, my_shard + 1, n_shards)
    if not keep:
        return
    t0 = time.time()
    games = 0
    positions = 0
    with ProcessPoolExecutor(WORKERS) as pool:
        for g, p, _ in pool.map(analyse_batch, jobs(part, out_dir, keep, first), buffersize=2 * WORKERS):
            games += g
            positions += p
            elapsed = time.time() - t0
            left = len(keep) / n_shards - games
            log.info("%s: %s games, %s positions, %.1f min, about %.0f min left", part, f"{games:,}",
                     f"{positions:,}", elapsed / 60, elapsed / games * max(left, 0) / 60)
    log.info("%s: done, %d batch files", part, len(list(out_dir.glob("*.parquet"))))


def run(parts=PARTS) -> None:
    for part in parts:
        run_part(part)
    if shard() == (0, 1):
        name = engine_name()
        for part in parts:
            merge(part, name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="*", choices=PARTS, default=list(PARTS))
    ap.add_argument("--merge", action="store_true", help="only add the batches of all shards to the result")
    args = ap.parse_args()
    if args.merge:
        setup_log(log_path(PREPROCESSING_ROOT, "stockfish_merge"))
        engine = engine_name()
        for p in args.parts or PARTS:
            merge(p, engine)
    else:
        run(args.parts or PARTS)
