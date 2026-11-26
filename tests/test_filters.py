"""Step 1, cleanup: which games are complete cases, and what is read from them."""
import importlib
from collections import Counter

import numpy as np
import pytest

from pgn_factory import FOOLS_MATE, lichess_game

filters = importlib.import_module("src.01_selection.month.filters")


def keep(pgn: str):
    return filters.complete_game(pgn, Counter())


@pytest.mark.parametrize("plies", [1, 2, 5, 19, 60])
def test_every_length_is_kept(plies):
    game = keep(lichess_game(plies))
    assert game is not None
    assert len(game.clk) == plies


def test_game_without_moves_is_dropped():
    assert keep(lichess_game(0)) is None


@pytest.mark.parametrize("speed", ["UltraBullet", "Correspondence"])
def test_other_speeds_are_dropped(speed):
    funnel = Counter()
    assert filters.complete_game(lichess_game(speed=speed), funnel) is None
    assert funnel[("scanned", "other")] == 1


def test_bot_is_dropped():
    assert keep(lichess_game(title="BOT")) is None


def test_titled_human_is_kept():
    assert keep(lichess_game(title="FM")) is not None


@pytest.mark.parametrize("termination", ["Abandoned", "Rules infraction", "Unterminated"])
def test_unfinished_games_are_dropped(termination):
    assert keep(lichess_game(termination=termination)) is None


def test_time_forfeit_is_kept():
    assert keep(lichess_game(termination="Time forfeit")) is not None


def test_unknown_rating_is_dropped():
    assert keep(lichess_game(white_elo="?")) is None


def test_game_without_evaluations_is_dropped():
    funnel = Counter()
    assert filters.complete_game(lichess_game(evals=[None] * 6), funnel) is None
    assert funnel[("speed bullet-classical", "blitz")] == 1
    assert funnel[("has engine evaluations", "blitz")] == 0


def test_missing_clock_is_dropped():
    clocks = ["0:03:00"] * 6
    clocks[3] = None
    assert keep(lichess_game(clocks=clocks)) is None


def test_missing_evaluation_in_the_middle_is_dropped():
    evals = ["0.2"] * 6
    evals[2] = None
    assert keep(lichess_game(evals=evals)) is None


def test_final_checkmate_needs_no_evaluation():
    game = keep(lichess_game(FOOLS_MATE, result="0-1", evals=["0.0", "0.1", "-0.5", None]))
    assert game is not None
    assert len(game.clk) == 4


def test_last_move_without_evaluation_and_no_mate_is_dropped():
    assert keep(lichess_game(6, evals=["0.2"] * 5 + [None])) is None


def test_blunder_marks_are_the_label():
    game = keep(lichess_game(8, blunders=(2, 5)))
    assert game.blunder.tolist() == [False, False, True, False, False, True, False, False]


def test_values_read_from_the_game():
    game = keep(lichess_game(4, speed="Rapid", white_elo="1234", black_elo="2345", site="xyz98765",
                             clocks=["0:10:00", "0:10:00", "0:09:30", "0:09:55"]))
    assert (game.game_id, game.speed, game.white_elo, game.black_elo) == ("xyz98765", "rapid", 1234, 2345)
    assert np.array_equal(game.clk, [600, 600, 570, 595])


def test_funnel_counts_every_step():
    funnel = Counter()
    for pgn in [lichess_game(), lichess_game(title="BOT"), lichess_game(evals=[None] * 6)]:
        filters.complete_game(pgn, funnel)
    assert [funnel[(step, "blitz")] for step in filters.STEPS] == [3, 3, 2, 1, 1]


def test_split_games():
    text = lichess_game(site="a") + lichess_game(site="b") + lichess_game(site="c")
    parts = list(filters.split_games(text))
    assert len(parts) == 3
    assert all(p.startswith("[Event ") for p in parts)


def test_same_filters():
    assert filters.same_filters({"filters": dict(filters.FILTERS)})
    assert not filters.same_filters({"filters": {**filters.FILTERS, "min_plies": 20}})
    assert not filters.same_filters({})
