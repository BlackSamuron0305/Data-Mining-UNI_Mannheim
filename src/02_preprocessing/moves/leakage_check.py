"""Leakage check: cut games at a random move, change everything from that move on, and check that the features
of the moves up to the cut stay the same. The false predictors must change, which shows that the check works.

Usage:  uv run python -m src.02_preprocessing.moves.leakage_check [--games 300] [--engine 5] [--pgn FILE]
Output: data/02_preprocessing/<selection>/leakage_check.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
import sys

import chess
import chess.engine
import chess.pgn
import numpy as np
import pandas as pd

from src.config import PREPROCESSING_DIR, PREPROCESSING_ROOT, SEED, SELECTION_DIR, TARGET, log_path, setup_log
from src.common.pgn import iter_pgn, read_game, time_control
from .extract import game_features
from .preprocess import FALSE_PREDICTORS

log = logging.getLogger("leakage_check")


def cut_game(pgn: str, n: int, rng: random.Random) -> str:
    """The game up to ply n, with a random clock, evaluation and mark on move n and no result."""
    game = read_game(pgn)
    out = chess.pgn.Game()
    out.headers.update(game.headers)
    out.headers["Result"] = "*"
    out.headers["Termination"] = "Unterminated"
    node = out
    for t, old in enumerate(itertools.islice(game.mainline(), n + 1)):
        node = node.add_variation(old.move, comment=old.comment, nags=set(old.nags))
        if t == n:
            node.nags = {rng.choice([chess.pgn.NAG_BLUNDER, chess.pgn.NAG_MISTAKE, chess.pgn.NAG_DUBIOUS_MOVE])}
            # Lichess does not run the clock before a player's first move, so that clock is known in advance
            if n >= 2:
                node.set_clock(rng.uniform(0, 600))
            node.set_eval(chess.engine.PovScore(chess.engine.Cp(rng.randint(-900, 900)), chess.WHITE))
    return str(out)


def compare(full: pd.DataFrame, cut: pd.DataFrame, n: int) -> list[str]:
    """Feature columns whose rows 0..n differ between the full and the cut game."""
    a = full.iloc[:n + 1].reset_index(drop=True)
    b = cut.iloc[:n + 1].reset_index(drop=True)
    bad = []
    for col in a.columns:
        if col == TARGET:
            continue
        same = (a[col] == b[col]) | (a[col].isna() & b[col].isna())
        if not same.all():
            bad.append(col)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=300, help="games to check for extract.py")
    ap.add_argument("--engine", type=int, default=0, help="games to check for stockfish.py as well (needs Stockfish)")
    ap.add_argument("--pgn", default=str(SELECTION_DIR / "train.pgn.zst"))
    args = ap.parse_args()
    setup_log(log_path(PREPROCESSING_ROOT, "leakage_check"))
    rng = random.Random(SEED)

    leaks = {}
    checked = 0
    first_clock = []
    target_changed = 0
    engine_pairs = []
    for pgn in iter_pgn(args.pgn):
        if checked >= args.games:
            break
        features = game_features(pgn)
        if features is None or len(features["ply"]) < 2:
            continue
        full = pd.DataFrame(features)
        n = rng.randrange(len(full))
        cut_pgn = cut_game(pgn, n, rng)
        cut = pd.DataFrame(game_features(cut_pgn))
        for col in compare(full, cut, n):
            leaks.setdefault(col, []).append((full["game_id"].iloc[0], n))
        if full[TARGET].iloc[n] != cut[TARGET].iloc[n]:
            target_changed += 1
        # the clock after each side's first move should be the start time
        game = read_game(pgn)
        base, _ = time_control(game.headers.get("TimeControl", "0+0"))
        clocks = [node.clock() for node in itertools.islice(game.mainline(), 2)]
        first_clock.append(all(abs(c - base) <= 1 for c in clocks))
        if len(engine_pairs) < args.engine:
            engine_pairs.append((pgn, cut_pgn, n))
        checked += 1

    caught = {col: cases for col, cases in leaks.items() if col in FALSE_PREDICTORS}
    leaks = {col: cases for col, cases in leaks.items() if col not in FALSE_PREDICTORS}
    log.info("extract.py: %d games cut at a random ply, %d kept columns with a difference", checked, len(leaks))
    log.info("the cut changed the target of the cut ply in %d games (shows that move n itself was altered)",
             target_changed)
    for col, cases in leaks.items():
        log.info("  LEAK %s: %d games, e.g. game %s cut at ply %d", col, len(cases), *cases[0])
    log.info("false predictors that changed, as they should: %d of %d", len(caught), len(FALSE_PREDICTORS))
    for col in FALSE_PREDICTORS:
        log.info("  %-14s changed in %d games", col, len(caught.get(col, [])))
    log.info("first move of each side: clock after it = start time (within 1 s) in %.1f%% of %d games",
             100 * np.mean(first_clock), len(first_clock))

    engine_leaks = {}
    if engine_pairs:
        from ..engine.stockfish import analyse_texts
        for pgn, cut_pgn, n in engine_pairs:
            full = analyse_texts([pgn])
            cut = analyse_texts([cut_pgn])
            for col in compare(full, cut, n):
                engine_leaks.setdefault(col, []).append(n)
        leak_text = "".join(f"; LEAK {col}" for col in engine_leaks)
        log.info("stockfish.py: %d games, %d columns with a difference%s", len(engine_pairs), len(engine_leaks),
                 leak_text)
    ok = not leaks and not engine_leaks
    log.info("result: %s", "no leakage found" if ok else "LEAKAGE FOUND")
    result = {"games": checked, "leaks": sorted(leaks), "engine_games": len(engine_pairs),
              "engine_leaks": sorted(engine_leaks), "target_changed": target_changed,
              "false_predictors_changed": sorted(caught), "no_leakage": ok}
    PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    (PREPROCESSING_DIR / "leakage_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
