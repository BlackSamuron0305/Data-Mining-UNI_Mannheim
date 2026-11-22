"""Stockfish on the position before every move (how hard is the choice the mover faces?). Takes about 15 hours
on 16 cores; the default is to download our results instead (README.md).

Usage:  uv run python -m src.02_preprocessing.with_stockfish.engine [train|test ...]
Output: data/preprocessing/<selection>/with_stockfish/stockfish/<part>/NNNNN.parquet (one file per batch of games), engine.json
IE500_SHARD="i/n" does only every n-th batch file; IE500_ENGINE_REUSE=<selection> reuses that selection's files.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pandas as pd

from src.config import CFG, DATA, env, GAMES_SUB, LOGS_SUB, SELECTION_DIR, WORKERS, setup_log
from src.common.pgn import capped_cp, iter_pgn, read_game, win_prob
from src.common.buckets import uniform
from ..common.paths import PRE_DIR, STOCKFISH_DIR

log = logging.getLogger("engine")
PARTS = ("train", "test")
ENG = CFG["engine"]
MARKS = CFG["features"]["marks"]
OUT_DIR = STOCKFISH_DIR
TOP = 5


def engine_path() -> str:
    path = env("STOCKFISH_PATH") or ENG["path"]
    found = shutil.which(path)
    if not found and Path(path).exists():
        found = path
    if not found:
        raise FileNotFoundError(f"Stockfish not found at {path!r}: install it and set STOCKFISH_PATH in .env or the "
                                "environment, or engine.path in config.yaml")
    return found


def settings() -> dict:
    return {"top": ENG["top"], "worst": ENG["worst"], "threads": ENG["threads"], "hash_mb": ENG["hash_mb"],
            "batch": ENG["batch"], "marks": MARKS}


def open_engine() -> chess.engine.SimpleEngine:
    eng = chess.engine.SimpleEngine.popen_uci(engine_path())
    eng.configure({"Threads": ENG["threads"], "Hash": ENG["hash_mb"]})
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
    """The games of the part to analyse: all, or a fixed random subset of `engine.games`."""
    games = pd.read_csv(SELECTION_DIR / GAMES_SUB / "games.csv", usecols=["game_id", "split"])
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


def jobs(part: str, out_dir: Path, keep: set[str]):
    """(file, games) of every batch of this shard whose file is still missing."""
    my_shard, n_shards = shard()
    batch = []
    k = 0
    for pgn in iter_pgn(SELECTION_DIR / GAMES_SUB / f"{part}.pgn.zst"):
        if game_id_of(pgn) not in keep:
            continue
        batch.append(pgn)
        if len(batch) == ENG["batch"]:
            out = out_dir / f"{k:05d}.parquet"
            if k % n_shards == my_shard and not out.exists():
                yield str(out), tuple(batch)
            batch = []
            k += 1
    out = out_dir / f"{k:05d}.parquet"
    if batch and k % n_shards == my_shard and not out.exists():
        yield str(out), tuple(batch)


def reuse(part: str, out_dir: Path, name: str) -> set[str]:
    """IE500_ENGINE_REUSE=<selection>: copy in that selection's engine files of the part (same engine and
    settings) as reuse_NNNNN.parquet and return their games, which are then not analysed again."""
    src = os.environ.get("IE500_ENGINE_REUSE")
    if not src:
        return set()
    src_dir = DATA / "preprocessing" / src / STOCKFISH_DIR.relative_to(PRE_DIR) / part
    if src_dir.resolve() == out_dir.resolve():
        return set()
    meta = json.loads((src_dir / "engine.json").read_text(encoding="utf-8"))
    same_settings = {k: meta.get(k) for k in settings()} == settings()
    if meta.get("engine") != name or not same_settings:
        raise RuntimeError(f"{src_dir} was made with another engine or other settings; it cannot be reused")
    games = set()
    files = sorted(src_dir.glob("[0-9]*.parquet"))
    for f in files:
        dst = out_dir / f"reuse_{f.name}"
        if not dst.exists():
            shutil.copyfile(f, dst)
        games.update(pd.read_parquet(f, columns=["game_id"])["game_id"].unique())
    log.info("%s: reused %d files with %s games from %s", part, len(files), f"{len(games):,}", src_dir)
    return games


def run_part(part: str) -> None:
    out_dir = OUT_DIR / part
    out_dir.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / LOGS_SUB).mkdir(parents=True, exist_ok=True)
    setup_log(OUT_DIR / LOGS_SUB / f"engine_{part}.log")
    eng = open_engine()
    name = eng.id.get("name", "?")
    eng.quit()
    keep = chosen_games(part) - reuse(part, out_dir, name)

    meta_path = out_dir / "engine.json"
    meta = {"engine": name, "part": part, "games": len(keep), **settings()}
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in meta} != meta:
            raise RuntimeError(f"{out_dir} holds an analysis with other settings ({old}); delete the folder or "
                               "restore the settings")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    done = len(list(out_dir.glob("[0-9]*.parquet")))
    my_shard, n_shards = shard()
    log.info("%s: %s, %s games to analyse, %d batch files already there, %d workers, shard %d of %d", part, name,
             f"{len(keep):,}", done, WORKERS, my_shard + 1, n_shards)
    t0 = time.time()
    games = 0
    positions = 0
    with ProcessPoolExecutor(WORKERS) as pool:
        for g, p, _ in pool.map(analyse_batch, jobs(part, out_dir, keep), buffersize=2 * WORKERS):
            games += g
            positions += p
            elapsed = time.time() - t0
            left = len(keep) / n_shards - done / n_shards * ENG["batch"] - games
            log.info("%s: %s games, %s positions, %.1f min, about %.0f min left", part, f"{games:,}",
                     f"{positions:,}", elapsed / 60, elapsed / games * max(left, 0) / 60)
    log.info("%s: done, %d batch files", part, len(list(out_dir.glob("*.parquet"))))


def run(parts=PARTS) -> None:
    for part in parts:
        run_part(part)


if __name__ == "__main__":
    run(sys.argv[1:] or PARTS)
