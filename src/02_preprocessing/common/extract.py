"""One row per move of the selected games, with every column we can read from a game.

This includes the columns that are only known after the move or after the game (false predictors). They are
removed in the next step, so that the removal is visible.

Usage:  uv run python -m src.02_preprocessing.common.extract [train|test ...]
Output: data/02_preprocessing/<selection>/extracted/train.parquet, test.parquet, extract.json
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
from src.common.server import ensure
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


HISTORY = ["own_last_spent", "own_spent_recent", "opp_last_spent", "own_last_loss", "own_loss_mean",
           "own_loss_recent", "own_loss_max", "own_inaccuracies", "own_mistakes", "own_blunders",
           "own_moves_since_error", "opp_last_loss", "opp_loss_recent", "opp_mistakes", "opp_blunders"]


def running_mean(sums: list, counts: list, last: int | None = None) -> float:
    """Mean of a player's values so far (or of the last `last` of them), missing values skipped. sums and counts are
    running totals (sums[i]: the sum of the first i values), so the mean of the last k values is the total now minus
    the total k moves ago, divided by the number of values in between."""
    i = len(sums) - 1
    start = 0 if last is None else max(i - last, 0)
    n = counts[i] - counts[start]
    return (sums[i] - sums[start]) / n if n > 0 else np.nan


class Player:
    """What one player has done so far in the game."""

    def __init__(self):
        self.spent = []              # seconds spent on each of their moves
        self.loss = []               # win probability lost by each move (NaN without an evaluation)
        self.spent_sum = [0.0]       # running totals: spent_sum[i] is the time of the first i moves
        self.spent_count = [0]
        self.loss_sum = [0.0]        # ... the loss of the first i moves, a missing loss counted as 0
        self.loss_count = [0]        # ... and how many of them have a loss
        self.loss_max = np.nan       # the biggest loss so far
        self.marks = {1: 0.0, 2: 0.0, 3: 0.0}   # inaccuracies (?!), mistakes (?), blunders (??) so far
        self.last_error = None       # the number of their last move that was a mistake or a blunder

    def add(self, spent: float, loss: float, mark: int) -> None:
        """Remember one more move of this player."""
        self.spent.append(spent)
        self.loss.append(loss)
        self.spent_sum.append(self.spent_sum[-1] + spent)
        self.spent_count.append(self.spent_count[-1] + 1)
        has_loss = not np.isnan(loss)
        self.loss_sum.append(self.loss_sum[-1] + (loss if has_loss else 0.0))
        self.loss_count.append(self.loss_count[-1] + int(has_loss))
        self.loss_max = np.fmax(self.loss_max, loss)
        if mark in self.marks:
            self.marks[mark] += 1
        if mark >= 2:
            self.last_error = len(self.loss) - 1


def last(values: list) -> float:
    return values[-1] if values else np.nan


def history(spent: np.ndarray, loss: np.ndarray, mark: np.ndarray) -> dict[str, np.ndarray]:
    """What the player to move ("own") and the opponent ("opp") did before every move, one value per ply.

    A plain walk through the game: at every ply we first read off what both players have done so far and only then
    add the ply to its player, so nothing of the move itself or of a later move can get in."""
    n = len(mark)
    out = {name: np.full(n, np.nan) for name in HISTORY}
    players = (Player(), Player())           # White, Black
    for t in range(n):
        own, opp = players[t % 2], players[1 - t % 2]
        out["own_last_spent"][t] = last(own.spent)
        out["own_spent_recent"][t] = running_mean(own.spent_sum, own.spent_count, K)
        out["opp_last_spent"][t] = last(opp.spent)
        out["own_last_loss"][t] = last(own.loss)
        out["own_loss_mean"][t] = running_mean(own.loss_sum, own.loss_count)
        out["own_loss_recent"][t] = running_mean(own.loss_sum, own.loss_count, K)
        out["own_loss_max"][t] = own.loss_max
        out["own_inaccuracies"][t] = own.marks[1]
        out["own_mistakes"][t] = own.marks[2]
        out["own_blunders"][t] = own.marks[3]
        if own.last_error is not None:
            out["own_moves_since_error"][t] = len(own.loss) - own.last_error
        out["opp_last_loss"][t] = last(opp.loss)
        out["opp_loss_recent"][t] = running_mean(opp.loss_sum, opp.loss_count, K)
        out["opp_mistakes"][t] = opp.marks[2]
        out["opp_blunders"][t] = opp.marks[3]
        own.add(spent[t], loss[t], int(mark[t]))
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
    past = history(spent, loss, mark)
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
        "own_last_spent": f32(past["own_last_spent"]),
        "own_spent_recent": f32(past["own_spent_recent"]),
        "opp_last_spent": f32(past["opp_last_spent"]),
        "eval_cp": f32(eval_cp),
        "eval_wp": f32(wp_before),
        "own_last_loss": f32(past["own_last_loss"]),
        "own_loss_mean": f32(past["own_loss_mean"]),
        "own_loss_recent": f32(past["own_loss_recent"]),
        "own_loss_max": f32(past["own_loss_max"]),
        "own_inaccuracies": f32(past["own_inaccuracies"]),
        "own_mistakes": f32(past["own_mistakes"]),
        "own_blunders": f32(past["own_blunders"]),
        "own_moves_since_error": f32(past["own_moves_since_error"]),
        "opp_last_loss": f32(past["opp_last_loss"]),
        "opp_loss_recent": f32(past["opp_loss_recent"]),
        "opp_mistakes": f32(past["opp_mistakes"]),
        "opp_blunders": f32(past["opp_blunders"]),
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
    src = ensure(SELECTION_DIR / GAMES_SUB / f"{part}.pgn.zst", "the selection (src/01_selection/sampling/)")
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
