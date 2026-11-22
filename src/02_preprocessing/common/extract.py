"""One row per move of the selected games, with every column we can read from a game.

This includes the columns that are only known after the move or after the game (false predictors). They are
removed in the next step, so that the removal is visible.

Usage:  uv run python -m src.02_preprocessing.common.extract [train|test ...]
Output: data/preprocessing/extracted/train.parquet, test.parquet, extract.json
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import batched

import chess
import chess.pgn
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import CFG, GAMES_SUB, LOGS_SUB, SELECTION_DIR, WORKERS, setup_log
from src.common.pgn import CP_CAP, SPEED_RE, capped_cp, iter_pgn, read_game, time_control, win_prob
from src.common.buckets import SPEEDS, game_group, move_buckets
from .paths import EXTRACTED_DIR

log = logging.getLogger("extract")
PARTS = ("train", "test")
OUT_DIR = EXTRACTED_DIR
RESULTS = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}
K = int(CFG["features"]["recent_moves"])
ECO_FROM = int(CFG["features"].get("eco_from_ply", 12))
EVENT_RE = re.compile(r"(?:^| )(game|tournament|swiss)(?: |$)")
EVENT_TYPES = {"game": "lobby", "tournament": "arena", "swiss": "swiss"}
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
# Lichess marks: 1 = ?!, 2 = ?, 3 = ??
MARKS = {chess.pgn.NAG_DUBIOUS_MOVE: 1, chess.pgn.NAG_MISTAKE: 2, chess.pgn.NAG_BLUNDER: 3}


# The history helpers get the values of one player's moves (x) and, for every move to predict,
# how many of those moves came before it (upto). They return one value per move to predict.
def prefix_mean(x: np.ndarray, upto: np.ndarray, k: int | None = None) -> np.ndarray:
    """Mean of the earlier values (all, or the last k), NaN skipped."""
    ok = ~np.isnan(x)
    sums = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    counts = np.concatenate(([0], np.cumsum(ok)))
    if k is None:
        start = np.zeros_like(upto)
    else:
        start = np.maximum(upto - k, 0)
    n = counts[upto] - counts[start]
    return np.where(n > 0, (sums[upto] - sums[start]) / np.maximum(n, 1), np.nan)


def recent_mean(x: np.ndarray, upto: np.ndarray) -> np.ndarray:
    return prefix_mean(x, upto, K)


def prefix_sum(x: np.ndarray, upto: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(x, dtype=float)))[upto]


def prefix_max(x: np.ndarray, upto: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.full(len(upto), np.nan)
    return np.concatenate(([np.nan], np.fmax.accumulate(x)))[upto]


def last_value(x: np.ndarray, upto: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.full(len(upto), np.nan)
    return np.where(upto > 0, x[np.maximum(upto - 1, 0)], np.nan)


def since_last(x: np.ndarray, upto: np.ndarray) -> np.ndarray:
    """Moves since the last True value (1 = the move just before), NaN if there is none."""
    if len(x) == 0:
        return np.full(len(upto), np.nan)
    last_true = np.maximum.accumulate(np.where(x, np.arange(len(x)), -1))
    last = np.concatenate(([-1], last_true))[upto]
    return np.where(last >= 0, upto - last, np.nan)


def by_side(x: np.ndarray, fn, opponent: bool = False) -> np.ndarray:
    """For every ply, fn over the earlier moves of the mover (or of the opponent). x has one value per ply."""
    n = len(x)
    out = np.full(n, np.nan)
    for side in (0, 1):
        plies = np.arange(side, n, 2)
        own_before = np.arange(len(plies))
        if opponent:
            # White to move has seen as many Black moves as own moves, Black one more White move
            out[plies] = fn(x[1 - side::2], own_before + side)
        else:
            out[plies] = fn(x[side::2], own_before)
    return out


def material(board: chess.Board, color: bool) -> int:
    total = 0
    for piece, value in PIECE_VALUES.items():
        total += value * chess.popcount(board.pieces_mask(piece, color))
    return total


def rating_diff(text: str) -> float:
    """A RatingDiff header such as '+5' as a number, NaN if it is missing."""
    try:
        return float(text)
    except ValueError:
        return np.nan


def f32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def game_features(pgn: str) -> dict[str, np.ndarray] | None:
    """The rows of one game (column name -> one value per ply), or None if it cannot be read."""
    game = read_game(pgn)
    if game is None or game.errors:
        return None
    h = game.headers
    match = SPEED_RE.match(h.get("Event", ""))
    speed = match.group(1).lower() if match else ""
    nodes = list(game.mainline())
    n = len(nodes)
    if speed not in SPEEDS or n == 0 or any(node.clock() is None for node in nodes):
        return None
    white_elo = int(h["WhiteElo"])
    black_elo = int(h["BlackElo"])
    utc = h.get("UTCTime", "")
    hour = int(utc[:2]) if utc[:2].isdigit() else -1
    kind = EVENT_RE.search(h.get("Event", ""))
    event_type = EVENT_TYPES[kind.group(1)] if kind else None
    eco = h.get("ECO", "?")[:1]
    if eco not in ("A", "B", "C", "D", "E"):
        eco = None
    base, inc = time_control(h.get("TimeControl", "0+0"))
    result_white = RESULTS.get(h.get("Result", ""), np.nan)
    diff_white = rating_diff(h.get("WhiteRatingDiff", ""))
    diff_black = rating_diff(h.get("BlackRatingDiff", ""))
    clk = np.rint([node.clock() for node in nodes]).astype(np.int64)

    # go through the game: the board before each move, the evaluation and mark after it
    legal = np.empty(n, np.int16)
    check = np.empty(n, bool)
    mat_w = np.empty(n, np.int16)
    mat_b = np.empty(n, np.int16)
    cp_white = np.full(n, np.nan)
    mark = np.zeros(n, np.int8)
    board = game.board()
    for t, node in enumerate(nodes):
        legal[t] = board.legal_moves.count()
        check[t] = board.is_check()
        mat_w[t] = material(board, chess.WHITE)
        mat_b[t] = material(board, chess.BLACK)
        board.push(node.move)
        ev = node.eval()
        if ev is not None:
            cp_white[t] = capped_cp(ev.white())
        elif board.is_checkmate():
            cp_white[t] = CP_CAP if t % 2 == 0 else -CP_CAP
        mark[t] = max((MARKS.get(nag, 0) for nag in node.nags), default=0)

    t = np.arange(n)
    white = t % 2 == 0
    sign = np.where(white, 1.0, -1.0)
    eval_cp = sign * np.concatenate(([np.nan], cp_white[:-1]))
    wp_before = win_prob(eval_cp)
    loss = np.clip(wp_before - win_prob(sign * cp_white), 0.0, None)
    clock = np.concatenate((clk[:2], clk[:-2])).astype(float)
    opp_clock = np.concatenate(([base], clk[:-1])).astype(float)
    spent = np.maximum(clock - clk + inc, 0.0)
    est_duration = max(base + 40 * inc, 1)
    mover_elo = np.where(white, white_elo, black_elo)
    opp_elo = np.where(white, black_elo, white_elo)
    mat_own = np.where(white, mat_w, mat_b)
    mat_opp = np.where(white, mat_b, mat_w)
    inaccuracies = (mark == 1).astype(float)
    mistakes = (mark == 2).astype(float)
    blunders = (mark == 3).astype(float)
    return {
        "game_id": np.full(n, h.get("Site", "").rsplit("/", 1)[-1], dtype=object),
        "ply": t.astype(np.int16),
        "bucket": move_buckets(speed, white_elo, black_elo, clk).astype(np.int16),
        "group": np.full(n, game_group(speed, white_elo, black_elo), np.int16),
        "blunder": mark == 3,
        "speed": np.full(n, speed, dtype=object),
        "base_time": np.full(n, base, np.int32),
        "increment": np.full(n, inc, np.int32),
        "white": white,
        "move_no": (t // 2 + 1).astype(np.int16),
        "hour": np.full(n, hour, np.int8),
        "event_type": np.full(n, event_type, dtype=object),
        "eco": np.where(t >= ECO_FROM, eco, None).astype(object),
        "mover_elo": mover_elo.astype(np.int16),
        "opp_elo": opp_elo.astype(np.int16),
        "elo_diff": (mover_elo - opp_elo).astype(np.int16),
        "clock": f32(clock),
        "opp_clock": f32(opp_clock),
        "clock_rel": f32(clock / est_duration),
        "clock_diff": f32(clock - opp_clock),
        "own_last_spent": f32(by_side(spent, last_value)),
        "own_spent_recent": f32(by_side(spent, recent_mean)),
        "opp_last_spent": f32(by_side(spent, last_value, opponent=True)),
        "eval_cp": f32(eval_cp),
        "eval_wp": f32(wp_before),
        "own_last_loss": f32(by_side(loss, last_value)),
        "own_loss_mean": f32(by_side(loss, prefix_mean)),
        "own_loss_recent": f32(by_side(loss, recent_mean)),
        "own_loss_max": f32(by_side(loss, prefix_max)),
        "own_inaccuracies": f32(by_side(inaccuracies, prefix_sum)),
        "own_mistakes": f32(by_side(mistakes, prefix_sum)),
        "own_blunders": f32(by_side(blunders, prefix_sum)),
        "own_moves_since_error": f32(by_side(mark >= 2, since_last)),
        "opp_last_loss": f32(by_side(loss, last_value, opponent=True)),
        "opp_loss_recent": f32(by_side(loss, recent_mean, opponent=True)),
        "opp_mistakes": f32(by_side(mistakes, prefix_sum, opponent=True)),
        "opp_blunders": f32(by_side(blunders, prefix_sum, opponent=True)),
        "legal_moves": legal,
        "in_check": check,
        "material_own": mat_own,
        "material_opp": mat_opp,
        "material_diff": (mat_own - mat_opp).astype(np.int16),
        "material_total": (mat_w + mat_b).astype(np.int16),
        "mover_name": np.where(white, h.get("White", ""), h.get("Black", "")).astype(object),
        "opp_name": np.where(white, h.get("Black", ""), h.get("White", "")).astype(object),
        "opening": np.full(n, h.get("Opening", ""), dtype=object),
        "eco_full": np.full(n, eco, dtype=object),
        "result": f32(sign * result_white),
        "rating_change": f32(np.where(white, diff_white, diff_black)),
        "termination": np.full(n, h.get("Termination", ""), dtype=object),
        "game_plies": np.full(n, n, np.int16),
        "clock_after": f32(clk),
        "spent": f32(spent),
        "eval_after": f32(sign * cp_white),
        "loss": f32(loss),
        "mark": mark,
    }


def features_batch(texts: tuple[str, ...]) -> tuple[pd.DataFrame, int]:
    """The rows of a batch of games, and how many games could not be read."""
    games = []
    for pgn in texts:
        features = game_features(pgn)
        if features is not None:
            games.append(features)
    failed = len(texts) - len(games)
    if not games:
        return pd.DataFrame(), failed
    columns = {}
    for name in games[0]:
        columns[name] = np.concatenate([g[name] for g in games])
    return pd.DataFrame(columns), failed


def run_part(part: str) -> dict:
    src = SELECTION_DIR / GAMES_SUB / f"{part}.pgn.zst"
    if not src.exists():
        raise FileNotFoundError(f"{src} is missing: run the selection first (README.md)")
    out = OUT_DIR / f"{part}.parquet"
    tmp = OUT_DIR / f"{part}.parquet.part"
    t0 = time.time()
    rows = 0
    games = 0
    failed = 0
    writer = None
    schema = None
    try:
        with ProcessPoolExecutor(WORKERS) as pool:
            batches = pool.map(features_batch, batched(iter_pgn(src), 200), buffersize=2 * WORKERS)
            for k, (df, bad) in enumerate(batches):
                failed += bad
                if df.empty:
                    continue
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
                writer.write_table(table.cast(schema))
                rows += len(df)
                games += df["game_id"].nunique()
                if (k + 1) % 100 == 0:
                    log.info("%s: %s games, %s moves, %.1f min", part, f"{games:,}", f"{rows:,}", (time.time() - t0) / 60)
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(out)
    minutes = (time.time() - t0) / 60
    log.info("%s: %s games, %s moves, %d games not readable, %.1f min", part, f"{games:,}", f"{rows:,}", failed, minutes)
    return {"games": games, "moves": rows, "games_not_readable": failed, "minutes": round(minutes, 1)}


def run(parts=PARTS) -> None:
    (OUT_DIR / LOGS_SUB).mkdir(parents=True, exist_ok=True)
    setup_log(OUT_DIR / LOGS_SUB / "extract.log")
    meta_path = OUT_DIR / "extract.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["recent_moves"] = K
    for part in parts:
        meta[part] = run_part(part)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run(sys.argv[1:] or PARTS)
