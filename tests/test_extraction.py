"""Step 2, extraction: one row per move, known before the move; the false predictors; the engine's columns."""
import importlib
import random

import numpy as np
import pandas as pd
import pytest

from pgn_factory import lichess_game
from src.common.pgn import win_prob

extract = importlib.import_module("src.02_preprocessing.moves.extract")
leakage = importlib.import_module("src.02_preprocessing.moves.leakage_check")
false_predictors = importlib.import_module("src.02_preprocessing.moves.preprocess")
engine = importlib.import_module("src.02_preprocessing.engine.stockfish")

EVALS = ["0.0", "0.0", "-3.00", "-3.00", "-3.00", "-3.00", "-3.00", "-3.00"]


def test_one_row_per_move_and_every_column():
    f = extract.game_features(lichess_game(8))
    assert len(f) == 56 and all(len(v) == 8 for v in f.values())
    assert f["ply"].tolist() == list(range(8)) and f["move_no"].tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
    assert f["white"].tolist() == [True, False] * 4


@pytest.mark.parametrize("n", [1, 3, 6, 11, 15])
def test_nothing_from_the_move_or_later_leaks(n):
    pgn = lichess_game(16, evals=["0.3", "0.1", "-2.5", "-2.4", "1.0", "0.9", "0.8", "-0.5"] * 2, blunders=(2, 9))
    full = extract.game_features(pgn)
    cut = extract.game_features(leakage.cut_game(pgn, n, random.Random(n)))
    changed = set(leakage.compare(pd.DataFrame(full), pd.DataFrame(cut), n))
    assert changed <= set(false_predictors.FALSE_PREDICTORS)           # only the false predictors may change
    assert {"result", "termination", "mark", "loss"} <= changed        # and they do
    assert ("game_plies" in changed) == (n < 15)                       # cut at the last move: same length


def test_history_of_the_mover_and_the_opponent():
    # White's second move (ply 2) drops the evaluation from 0 to -3 pawns and is marked as a blunder
    f = extract.game_features(lichess_game(8, evals=EVALS, blunders=(2,)))
    loss = win_prob(0) - win_prob(-300)
    assert f["loss"][2] == pytest.approx(loss, abs=1e-6)
    assert np.isnan(f["own_last_loss"][2])                             # White's first move: no eval before it
    assert f["own_last_loss"][4] == pytest.approx(loss, abs=1e-6)       # White's next move knows it
    assert f["own_blunders"].tolist() == [0, 0, 0, 0, 1, 0, 1, 0]       # White's count, from ply 4 on
    assert f["opp_blunders"].tolist() == [0, 0, 0, 1, 0, 1, 0, 1]       # Black sees it from ply 3 on
    assert f["own_moves_since_error"][4] == 1 and f["own_moves_since_error"][6] == 2
    assert np.isnan(f["own_moves_since_error"][1])                      # Black made no error


def test_evaluation_and_win_probability_before_the_move():
    f = extract.game_features(lichess_game(8, evals=EVALS))
    assert np.isnan(f["eval_cp"][0])                                    # no evaluation before the first move
    assert f["eval_cp"][3] == pytest.approx(300)                        # Black to move after -3: +3 for Black
    assert f["eval_wp"][3] == pytest.approx(win_prob(300), abs=1e-6)


def test_opening_letter_only_from_ply_12():
    f = extract.game_features(lichess_game(14))
    assert list(f["eco"][:12]) == [None] * 12 and list(f["eco"][12:]) == ["A", "A"]
    assert set(f["eco_full"]) == {"A"}                                  # the false predictor has it everywhere


def test_clock_before_the_move():
    clocks = ["0:03:00", "0:03:00", "0:02:50", "0:02:40", "0:02:30", "0:02:00"]
    f = extract.game_features(lichess_game(6, clocks=clocks))
    assert f["clock"].tolist() == [180, 180, 180, 180, 170, 160]        # first moves: the clock after them
    assert f["clock_rel"][4] == pytest.approx(170 / 180)                # 3+0: 180 s for the whole game
    assert f["own_last_spent"][4] == pytest.approx(10)                  # 3:00 -> 2:50 on White's second move


def test_drop_false_predictors():
    f = extract.game_features(lichess_game(4))
    kept = false_predictors.drop_false_predictors(pd.DataFrame(f))
    assert kept.shape[1] == 56 - 13
    assert not set(false_predictors.FALSE_PREDICTORS) & set(kept.columns)


def test_engine_position_columns():
    top = np.array([0.60, 0.55, 0.40, 0.30, 0.10])                     # win probability of the best 5 moves
    every = np.array([0.60, 0.58, 0.50, 0.40, 0.30, 0.10, 0.05])       # of every legal move, best first
    row = engine.position_features(top, every, n_legal=7)
    assert row["sf_wp_best"] == 0.60 and row["sf_n_scored"] == 7
    assert row["sf_gap_2"] == pytest.approx(0.05) and row["sf_gap_5"] == pytest.approx(0.50)
    assert row["sf_worst_1"] == pytest.approx(0.55)                    # the worst move loses 0.55
    assert row["sf_n_close"] == 2                                       # within 0.05 of the best: 0.60, 0.58
    assert row["sf_share_blunder"] == pytest.approx(4 / 7)             # loses 0.15 or more: 4 of 7


def test_engine_columns_with_few_moves():
    row = engine.position_features(np.array([0.5, 0.2]), np.array([0.5, 0.2]), n_legal=2)
    assert np.isnan(row["sf_gap_3"]) and np.isnan(row["sf_worst_3"])
