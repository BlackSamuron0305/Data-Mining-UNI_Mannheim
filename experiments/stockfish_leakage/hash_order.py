"""Does the order of Lichess's analysis put the played move into the evaluation before it?

Lichess's fishnet analyses a game backwards, in chunks of 6 positions that share the engine's hash (fishnet
src/queue.rs: positions.reverse(); src/stockfish.rs: ucinewgame once per chunk; one warm-up position per chunk,
its result discarded; Lichess's manual analysis: about 857,000 nodes per position). So when it evaluates the
position before move t, the hash already holds deep searches of the positions after the move that was played.

We evaluate the same positions twice with the same node budget:
  F  forward, hash cleared before every position: nothing of the game's future can get in
  R  as fishnet does it: backwards, chunks of 6 with a shared hash, warm-up position discarded
and write one row per position (mover's win probability of both, and Lichess's eval_wp from the tables).

Usage (from the repository): uv run python -m experiments.stockfish_leakage.hash_order --games 300 --nodes 500000
"""
from __future__ import annotations

import argparse
import importlib
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pandas as pd

from src.common.pgn import capped_cp, iter_pgn, read_game, win_prob

engine_path = importlib.import_module("src.02_preprocessing.engine.stockfish").engine_path

OUT = Path("data/experiments/stockfish_leakage")
CHUNK = 6


def boards_of(pgn: str) -> tuple[str, list[chess.Board]]:
    game = read_game(pgn)
    gid = game.headers["Site"].rsplit("/", 1)[-1]
    board = game.board()
    out = []
    for node in game.mainline():
        out.append(board.copy(stack=True))     # the position before the move, with its history
        board.push(node.move)
    return gid, out


def analyse(job) -> list[dict]:
    pgn, nodes = job
    gid, boards = boards_of(pgn)
    eng = chess.engine.SimpleEngine.popen_uci(engine_path())
    eng.configure({"Threads": 1, "Hash": 16})
    try:
        fwd = []
        for b in boards:
            # a new game for every position clears the hash
            fwd.append(score_new(eng, b, nodes))
        rev = [np.nan] * len(boards)
        order = list(range(len(boards)))[::-1]
        # chunks of CHUNK positions, each but the first starting with a warm-up copy of the previous chunk's last one
        start = 0
        while start < len(order):
            chunk = order[start:start + CHUNK]
            warm = [order[start - 1]] if start > 0 else []
            first = True
            for k in warm + chunk:
                if first:
                    v = score_new(eng, boards[k], nodes)
                    first = False
                else:
                    v = score(eng, boards[k], nodes)
                if not (warm and k == warm[0]):
                    rev[k] = v
            start += CHUNK
    finally:
        eng.quit()
    return [{"game_id": gid, "ply": t, "wp_fwd": fwd[t], "wp_rev": rev[t]} for t in range(len(boards))]


_GAME = [0]


def score_new(eng, board, nodes) -> float:
    """Score with a cleared hash: python-chess sends ucinewgame whenever the game object changes."""
    _GAME[0] += 1
    info = eng.analyse(board, chess.engine.Limit(nodes=nodes), game=_GAME[0])
    return float(win_prob(capped_cp(info["score"].pov(board.turn))))


def score(eng, board, nodes) -> float:
    """Score with the hash of the positions before it in this chunk (same game object)."""
    info = eng.analyse(board, chess.engine.Limit(nodes=nodes), game=_GAME[0])
    return float(win_prob(capped_cp(info["score"].pov(board.turn))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--nodes", type=int, default=500_000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--part", default="train")
    args = ap.parse_args()
    texts = list(iter_pgn(f"data/01_selection/stratified/{args.part}.pgn.zst"))
    random.Random(42).shuffle(texts)
    texts = texts[:args.games]
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(analyse, [(t, args.nodes) for t in texts]), 1):
            rows.extend(r)
            if i % 20 == 0:
                print(f"{i} games, {len(rows)} positions", flush=True)
    df = pd.DataFrame(rows)
    lab = pd.read_parquet(f"data/02_preprocessing/stratified/{args.part}.parquet",
                          columns=["game_id", "ply", "blunder", "eval_wp"],
                          filters=[("game_id", "in", sorted(df["game_id"].unique()))])
    df = df.merge(lab, on=["game_id", "ply"], how="left", validate="one_to_one")
    out = OUT / f"hash_order_{args.part}_{args.games}g_{args.nodes}n.parquet"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{len(df)} positions of {df['game_id'].nunique()} games -> {out}")


if __name__ == "__main__":
    main()
