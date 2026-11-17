"""Features step 2 - Stockfish on the position before every move: how hard is the choice the mover faces?

Usage:  uv run python -m src.preprocessing.features.engine [train|test ...]       (after the selection)

Stockfish ranks the candidate moves of the position before move t (MultiPV). With `multipv: all` every legal move
gets a score, which gives the best 5, the worst 5 and how many moves would lose how much; with a number only the
best moves get one. Scores are turned into the mover's win probability as Lichess does it (src/lichess/pgn.py), and
every candidate move into its loss against the best move. The move actually played is never looked at: the
columns describe the options before the move, so they can predict whether the move to come is a blunder without
giving it away.

Settings in config.yaml (engine:): the program (STOCKFISH_PATH, else `path`), the search limit per position, MultiPV,
threads and hash per engine, and how many games per part (a fixed random subset; the analysis of every move of the
350,000 games takes days). Every worker process runs its own engine (one per batch of games) with one thread;
every game starts with `ucinewgame`, so the results do not depend on the order or the number of workers.

Columns (one row per move, keys as in positions/)
game_id, ply
sf_wp_best                        win probability after the best move (the engine's evaluation of the position)
sf_gap_2 ... sf_gap_5             win probability lost by the 2nd ... 5th best move (NaN if there are fewer moves)
sf_worst_1 ... sf_worst_5         win probability lost by the worst, 2nd worst, ... move (multipv: all only)
sf_n_scored                       moves with a score (every legal move with multipv: all)
sf_n_close                        moves within `features.marks.inaccuracy` of the best (among the scored moves)
sf_share_inaccuracy, sf_share_mistake, sf_share_blunder
                                  share of the legal moves that lose at least that much (multipv: all only)
sf_best_move                      the engine's best move (UCI): for checks, not a feature

Outputs (data/features/engine/<part>/)
NNNNN.parquet      one file per `batch` games, in the order of the games in the PGN file; a stopped run continues
                   with the first missing file
engine.json        settings and engine name; a new run with other settings stops instead of mixing them
logs/engine_<part>.log
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pandas as pd

from src.config import CFG, FEATURES_DIR, env, GAMES_SUB, LOGS_SUB, STRATIFIED_DIR, WORKERS, setup_log
from src.lichess.pgn import capped_cp, iter_pgn, read_game, win_prob
from src.preprocessing.selection.census import uniform

log = logging.getLogger("engine")
PARTS = ("train", "test")
ENG = CFG["engine"]
MARKS = CFG["features"]["marks"]
OUT_DIR = FEATURES_DIR / "engine"
TOP = 5


def engine_path() -> str:
    path = env("STOCKFISH_PATH") or ENG["path"]
    found = shutil.which(path) or (path if Path(path).exists() else None)
    if not found:
        raise FileNotFoundError(f"Stockfish not found at {path!r}: install it and set STOCKFISH_PATH in .env or the "
                                "environment, or engine.path in config.yaml")
    return found


def settings() -> dict:
    return {"limit": ENG["limit"], "multipv": ENG["multipv"], "threads": ENG["threads"], "hash_mb": ENG["hash_mb"],
            "batch": ENG["batch"], "marks": MARKS}


def open_engine() -> chess.engine.SimpleEngine:
    eng = chess.engine.SimpleEngine.popen_uci(engine_path())
    eng.configure({"Threads": ENG["threads"], "Hash": ENG["hash_mb"]})
    return eng


# ---------------------------------------------------------------------------------------------- one position
def position_features(wps: np.ndarray, n_legal: int, all_moves: bool) -> dict:
    """Columns of one position from the win probabilities of its scored moves (best first)."""
    loss = wps[0] - wps
    row = {"sf_wp_best": wps[0], "sf_n_scored": len(wps), "sf_n_close": int(np.sum(loss < MARKS["inaccuracy"]))}
    for k in range(2, TOP + 1):
        row[f"sf_gap_{k}"] = loss[k - 1] if len(loss) >= k else np.nan
    for k in range(1, TOP + 1):
        row[f"sf_worst_{k}"] = loss[-k] if all_moves and len(loss) >= k else np.nan
    for name, th in MARKS.items():
        row[f"sf_share_{name}"] = float(np.sum(loss >= th)) / n_legal if all_moves else np.nan
    return row


def analyse_game(eng: chess.engine.SimpleEngine, pgn: str, limit: chess.engine.Limit) -> list[dict]:
    game = read_game(pgn)
    gid = game.headers.get("Site", "").rsplit("/", 1)[-1]
    board, rows = game.board(), []
    for t, node in enumerate(game.mainline()):
        n_legal = board.legal_moves.count()
        all_moves = ENG["multipv"] == "all" or int(ENG["multipv"]) >= n_legal
        k = n_legal if all_moves else int(ENG["multipv"])
        infos = [i for i in eng.analyse(board, limit, multipv=k, game=gid) if "score" in i and i.get("pv")]
        if not infos:
            raise RuntimeError(f"no score from the engine for ply {t} of game {gid}")
        wps = np.sort(win_prob([capped_cp(i["score"].pov(board.turn)) for i in infos]))[::-1]
        all_moves = all_moves and len(wps) == n_legal
        rows.append({"game_id": gid, "ply": t, **position_features(wps, n_legal, all_moves),
                     "sf_best_move": infos[0]["pv"][0].uci()})
        board.push(node.move)                    # the played move only leads to the next position
    return rows


def analyse_batch(job: tuple[str, tuple[str, ...]]) -> tuple[int, int, float]:
    """Analyse a batch of games and write its file; returns games, positions, seconds."""
    out, texts = Path(job[0]), job[1]
    t0, limit = time.time(), chess.engine.Limit(**ENG["limit"])
    eng = open_engine()             # one engine per batch: an engine left open would keep the worker from exiting
    try:
        rows = [r for pgn in texts for r in analyse_game(eng, pgn, limit)]
    finally:
        eng.quit()
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c.startswith("sf_") and c != "sf_best_move":
            df[c] = df[c].astype(np.float32)
    df["ply"] = df["ply"].astype(np.int16)
    tmp = out.with_suffix(".part")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(out)
    return len(texts), len(df), time.time() - t0


# -------------------------------------------------------------------------------------------------------- run
def chosen_games(part: str) -> set[str]:
    """The games of the part to analyse: all, or the `games` ones with the smallest fixed random number."""
    games = pd.read_csv(STRATIFIED_DIR / GAMES_SUB / "games.csv", usecols=["game_id", "split"])
    ids = games.loc[games["split"] == part, "game_id"].tolist()
    n = (ENG.get("games") or {}).get(part)
    if n is None or n >= len(ids):
        return set(ids)
    return set(sorted(ids, key=lambda g: uniform(g, "ie500-engine"))[:n])


def jobs(part: str, out_dir: Path, keep: set[str]):
    """(file, games) of every batch whose file is still missing, in the order of the PGN file."""
    batch, k = [], 0
    for pgn in iter_pgn(STRATIFIED_DIR / GAMES_SUB / f"{part}.pgn.zst"):
        site = pgn[pgn.find('[Site "') + 7:].split('"', 1)[0] if '[Site "' in pgn else ""
        if site.rsplit("/", 1)[-1] not in keep:
            continue
        batch.append(pgn)
        if len(batch) == ENG["batch"]:
            out = out_dir / f"{k:05d}.parquet"
            if not out.exists():
                yield str(out), tuple(batch)
            batch, k = [], k + 1
    if batch and not (out_dir / f"{k:05d}.parquet").exists():
        yield str(out_dir / f"{k:05d}.parquet"), tuple(batch)


def run_part(part: str) -> None:
    out_dir = OUT_DIR / part
    out_dir.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / LOGS_SUB).mkdir(parents=True, exist_ok=True)
    setup_log(OUT_DIR / LOGS_SUB / f"engine_{part}.log")
    keep = chosen_games(part)
    eng = open_engine()
    name = eng.id.get("name", "?")
    eng.quit()
    meta_path = out_dir / "engine.json"
    meta = {"engine": name, "part": part, "games": len(keep), **settings()}
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if {k: old.get(k) for k in meta} != meta:
            raise RuntimeError(f"{out_dir} holds an analysis with other settings ({old}); delete the folder or "
                               "restore the settings")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    done = len(list(out_dir.glob("*.parquet")))
    log.info("%s: %s, %s games to analyse, %d batch files already there, %d workers", part, name, f"{len(keep):,}",
             done, WORKERS)
    t0, games, positions = time.time(), 0, 0
    with ProcessPoolExecutor(WORKERS) as pool:
        for g, p, _ in pool.map(analyse_batch, jobs(part, out_dir, keep), buffersize=2 * WORKERS):
            games, positions = games + g, positions + p
            el = time.time() - t0
            left = len(keep) - done * ENG["batch"] - games
            log.info("%s: %s games, %s positions, %.1f min, about %.0f min left", part, f"{games:,}",
                     f"{positions:,}", el / 60, el / games * max(left, 0) / 60)
    log.info("%s: done, %d batch files", part, len(list(out_dir.glob("*.parquet"))))


def run(parts=PARTS) -> None:
    for part in parts:
        run_part(part)


if __name__ == "__main__":
    run(sys.argv[1:] or PARTS)
