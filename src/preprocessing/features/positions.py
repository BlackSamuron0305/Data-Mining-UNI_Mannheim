"""Features step 1 - one row per move of the selected games, with what is known before the move is played.

Usage:  uv run python -m src.preprocessing.features.positions [train|test ...]     (after the selection)

A row is ply t of a game (t = 0 is White's first move): the move to predict. The target `blunder` says that Lichess
marked move t with ??. Every feature is known before move t is played: the position, both clocks, the ratings and
the moves 0..t-1 with their Lichess marks and evaluations. Nothing of move t itself or of a later move enters, and
neither does the length or the result of the game.

Columns
keys      game_id, ply, bucket (speed x Elo of the mover x time left, as in the census), group
target    blunder
game      speed, base_time, increment (seconds), white (the mover plays White), move_no, mover_elo, opp_elo, elo_diff
clock     clock (the mover's time left, seconds), opp_clock, clock_rel (clock / estimated duration of the game,
          start time + 40 x increment), clock_diff (clock - opp_clock), own_last_spent (time the mover spent on
          their previous move), own_spent_recent (mean over their last `recent_moves` moves), opp_last_spent
eval      eval_cp (Lichess's evaluation of the position, the mover's side, capped at +-1000; NaN before the first
          move), eval_wp (the same as win probability)
history   own_last_loss, own_loss_mean, own_loss_recent, own_loss_max: loss in win probability of the mover's
          previous moves (the last one, all, the last `recent_moves`, the largest); own_inaccuracies, own_mistakes,
          own_blunders: Lichess marks of the mover so far; own_moves_since_error: own moves since their last mistake
          or blunder (1 = the previous one, NaN = none yet); opp_last_loss, opp_loss_recent, opp_mistakes,
          opp_blunders: the same for the opponent
board     legal_moves, in_check, material_own, material_opp, material_diff, material_total (pawn 1, knight and
          bishop 3, rook 5, queen 9)

Outputs (data/features/positions/)
train.parquet, test.parquet       one row per move
positions.json                    games and moves per part, settings
logs/positions.log
"""
from __future__ import annotations

import json
import logging
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

from src.config import CFG, FEATURES_DIR, GAMES_SUB, LOGS_SUB, STRATIFIED_DIR, WORKERS, setup_log
from src.lichess.pgn import CP_CAP, SPEED_RE, capped_cp, iter_pgn, read_game, time_control, win_prob
from src.preprocessing.selection.census import SPEEDS, game_group, move_buckets

log = logging.getLogger("positions")
PARTS = ("train", "test")
OUT_DIR = FEATURES_DIR / "positions"
K = int(CFG["features"]["recent_moves"])
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
MARKS = {chess.pgn.NAG_DUBIOUS_MOVE: 1, chess.pgn.NAG_MISTAKE: 2, chess.pgn.NAG_BLUNDER: 3}     # ?!, ?, ??


# ------------------------------------------------------------------------------------------- history helpers
# A helper gets the moves of one side in order (x) and, for every move to predict, how many of them lie before it
# (upto); it returns one value per move to predict.
def prefix_mean(x: np.ndarray, upto: np.ndarray, k: int | None = None) -> np.ndarray:
    """Mean of the values before (all, or the last k), NaN values skipped; NaN if there is none."""
    ok = ~np.isnan(x)
    s = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    c = np.concatenate(([0], np.cumsum(ok)))
    lo = np.zeros_like(upto) if k is None else np.maximum(upto - k, 0)
    n = c[upto] - c[lo]
    return np.where(n > 0, (s[upto] - s[lo]) / np.maximum(n, 1), np.nan)


def prefix_sum(x: np.ndarray, upto: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(x, dtype=float)))[upto]


def prefix_max(x: np.ndarray, upto: np.ndarray) -> np.ndarray:
    m = np.concatenate(([np.nan], np.fmax.accumulate(x) if len(x) else []))
    return m[upto]


def last_value(x: np.ndarray, upto: np.ndarray) -> np.ndarray:
    return np.where(upto > 0, x[np.maximum(upto - 1, 0)] if len(x) else np.nan, np.nan)


def since_last(x: np.ndarray, upto: np.ndarray) -> np.ndarray:
    """Moves since the last True before (1 = the move just before), NaN if there is none."""
    last = np.concatenate(([-1], np.maximum.accumulate(np.where(x, np.arange(len(x)), -1)) if len(x) else []))
    li = last[upto]
    return np.where(li >= 0, upto - li, np.nan)


def by_side(x: np.ndarray, fn, opponent: bool = False) -> np.ndarray:
    """For every ply t, fn over the earlier moves of the mover (or of the opponent) of ply t. x has one value per ply."""
    n = len(x)
    out = np.full(n, np.nan)
    for p in (0, 1):
        idx = np.arange(p, n, 2)                  # the plies of this side
        j = np.arange(len(idx))                   # own moves before ply idx[j]: j
        if opponent:                              # opponent moves before: j for White to move, j + 1 for Black
            out[idx] = fn(x[1 - p::2], j + p)
        else:
            out[idx] = fn(x[p::2], j)
    return out


# ------------------------------------------------------------------------------------------------- one game
def material(board: chess.Board, color: bool) -> int:
    return sum(v * chess.popcount(board.pieces_mask(pt, color)) for pt, v in PIECE_VALUES.items())


def game_features(pgn: str) -> dict[str, np.ndarray] | None:
    """The rows of one game (a column name -> one value per ply), or None if it cannot be read."""
    game = read_game(pgn)
    if game is None or game.errors:
        return None
    h = game.headers
    m = SPEED_RE.match(h.get("Event", ""))
    speed = m.group(1).lower() if m else ""
    nodes = list(game.mainline())
    n = len(nodes)
    if speed not in SPEEDS or n == 0 or any(nd.clock() is None for nd in nodes):
        return None
    w_elo, b_elo = int(h["WhiteElo"]), int(h["BlackElo"])
    base, inc = time_control(h.get("TimeControl", "0+0"))
    clk = np.rint([nd.clock() for nd in nodes]).astype(np.int64)

    legal, check = np.empty(n, np.int16), np.empty(n, bool)
    mat_w, mat_b = np.empty(n, np.int16), np.empty(n, np.int16)
    cp_white = np.full(n, np.nan)                 # Lichess evaluation after ply t, White's side
    mark = np.zeros(n, np.int8)                   # 1 ?!, 2 ?, 3 ??
    board = game.board()
    for t, nd in enumerate(nodes):
        legal[t], check[t] = board.legal_moves.count(), board.is_check()
        mat_w[t], mat_b[t] = material(board, chess.WHITE), material(board, chess.BLACK)
        board.push(nd.move)
        ev = nd.eval()
        if ev is not None:
            cp_white[t] = capped_cp(ev.white())
        elif board.is_checkmate():
            cp_white[t] = CP_CAP if t % 2 == 0 else -CP_CAP
        mark[t] = max((MARKS.get(g, 0) for g in nd.nags), default=0)

    t = np.arange(n)
    white = t % 2 == 0
    sign = np.where(white, 1.0, -1.0)
    eval_cp = sign * np.concatenate(([np.nan], cp_white[:-1]))      # before ply t, the mover's side
    wp_before = win_prob(eval_cp)
    loss = np.clip(wp_before - win_prob(sign * cp_white), 0.0, None)   # of ply t, used only for later plies
    clock = np.concatenate((clk[:2], clk[:-2])).astype(float)       # the mover's clock before the move
    opp_clock = np.concatenate(([base], clk[:-1])).astype(float)
    spent = np.maximum(clock - clk + inc, 0.0)                      # of ply t, used only for later plies
    est = max(base + 40 * inc, 1)
    elo_mover = np.where(white, w_elo, b_elo)
    elo_opp = np.where(white, b_elo, w_elo)
    mat_own, mat_opp = np.where(white, mat_w, mat_b), np.where(white, mat_b, mat_w)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    return {
        "game_id": np.full(n, h.get("Site", "").rsplit("/", 1)[-1], dtype=object),
        "ply": t.astype(np.int16),
        "bucket": move_buckets(speed, w_elo, b_elo, clk).astype(np.int16),
        "group": np.full(n, game_group(speed, w_elo, b_elo), np.int16),
        "blunder": mark == 3,
        "speed": np.full(n, speed, dtype=object),
        "base_time": np.full(n, base, np.int32), "increment": np.full(n, inc, np.int32),
        "white": white, "move_no": (t // 2 + 1).astype(np.int16),
        "mover_elo": elo_mover.astype(np.int16), "opp_elo": elo_opp.astype(np.int16),
        "elo_diff": (elo_mover - elo_opp).astype(np.int16),
        "clock": f32(clock), "opp_clock": f32(opp_clock), "clock_rel": f32(clock / est),
        "clock_diff": f32(clock - opp_clock),
        "own_last_spent": f32(by_side(spent, last_value)),
        "own_spent_recent": f32(by_side(spent, lambda x, u: prefix_mean(x, u, K))),
        "opp_last_spent": f32(by_side(spent, last_value, opponent=True)),
        "eval_cp": f32(eval_cp), "eval_wp": f32(wp_before),
        "own_last_loss": f32(by_side(loss, last_value)),
        "own_loss_mean": f32(by_side(loss, prefix_mean)),
        "own_loss_recent": f32(by_side(loss, lambda x, u: prefix_mean(x, u, K))),
        "own_loss_max": f32(by_side(loss, prefix_max)),
        "own_inaccuracies": f32(by_side((mark == 1).astype(float), prefix_sum)),
        "own_mistakes": f32(by_side((mark == 2).astype(float), prefix_sum)),
        "own_blunders": f32(by_side((mark == 3).astype(float), prefix_sum)),
        "own_moves_since_error": f32(by_side(mark >= 2, since_last)),
        "opp_last_loss": f32(by_side(loss, last_value, opponent=True)),
        "opp_loss_recent": f32(by_side(loss, lambda x, u: prefix_mean(x, u, K), opponent=True)),
        "opp_mistakes": f32(by_side((mark == 2).astype(float), prefix_sum, opponent=True)),
        "opp_blunders": f32(by_side((mark == 3).astype(float), prefix_sum, opponent=True)),
        "legal_moves": legal, "in_check": check,
        "material_own": mat_own, "material_opp": mat_opp, "material_diff": (mat_own - mat_opp).astype(np.int16),
        "material_total": (mat_w + mat_b).astype(np.int16),
    }


def features_batch(texts: tuple[str, ...]) -> tuple[pd.DataFrame, int]:
    """The rows of a batch of games, and how many games could not be read."""
    rows = [r for r in map(game_features, texts) if r is not None]
    if not rows:
        return pd.DataFrame(), len(texts)
    return pd.DataFrame({c: np.concatenate([r[c] for r in rows]) for c in rows[0]}), len(texts) - len(rows)


# -------------------------------------------------------------------------------------------------------- run
def run_part(part: str) -> dict:
    src = STRATIFIED_DIR / GAMES_SUB / f"{part}.pgn.zst"
    if not src.exists():
        raise FileNotFoundError(f"{src} is missing: run the selection first (README.md)")
    out, tmp = OUT_DIR / f"{part}.parquet", OUT_DIR / f"{part}.parquet.part"
    t0, rows, games, failed, writer, schema = time.time(), 0, 0, 0, None, None
    try:
        with ProcessPoolExecutor(WORKERS) as pool:
            for k, (df, bad) in enumerate(pool.map(features_batch, batched(iter_pgn(src), 200),
                                                   buffersize=2 * WORKERS)):
                failed += bad
                if df.empty:
                    continue
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(tmp, schema, compression="zstd")
                writer.write_table(table.cast(schema))
                rows, games = rows + len(df), games + df["game_id"].nunique()
                if (k + 1) % 100 == 0:
                    log.info("%s: %s games, %s moves, %.1f min", part, f"{games:,}", f"{rows:,}", (time.time() - t0) / 60)
    finally:
        if writer is not None:
            writer.close()
    tmp.replace(out)
    log.info("%s: %s games, %s moves, %d games not readable, %.1f min", part, f"{games:,}", f"{rows:,}", failed,
             (time.time() - t0) / 60)
    return {"games": games, "moves": rows, "games_not_readable": failed, "minutes": round((time.time() - t0) / 60, 1)}


def run(parts=PARTS) -> None:
    (OUT_DIR / LOGS_SUB).mkdir(parents=True, exist_ok=True)
    setup_log(OUT_DIR / LOGS_SUB / "positions.log")
    meta_path = OUT_DIR / "positions.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["recent_moves"] = K
    for part in parts:
        meta[part] = run_part(part)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run(sys.argv[1:] or PARTS)
