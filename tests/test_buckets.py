"""Groups of games and buckets of moves (src/common/buckets.py), used from step 1 to step 5."""
import numpy as np

from src.common import buckets as b


def test_sizes():
    assert b.SPEEDS == ["bullet", "blitz", "rapid", "classical"]
    assert b.N_E == 7                                   # below 750, 750-1000, ..., 2000 and above
    assert b.N_W.tolist() == [5, 11, 31, 61]            # 30 s steps up to 2, 5, 15 and 30 minutes, last one open
    assert b.N_B == 756
    assert b.N_G == 28


def test_elo_bands():
    assert [b.elo_label(e) for e in range(b.N_E)] == ["<750", "750-1000", "1000-1250", "1250-1500", "1500-1750",
                                                       "1750-2000", "2000+"]
    assert b.elo_bin([400, 749, 750, 1999, 2000, 3000]).tolist() == [0, 0, 1, 5, 6, 6]


def test_move_buckets_use_the_clock_before_the_move():
    # blitz, both 1500-1750; clocks after the moves: 180, 180, 170, 175, 20
    got = b.move_buckets("blitz", 1500, 1600, np.array([180, 180, 170, 175, 20]))
    s, elo = 1, 4
    # the first move of each side counts the clock after it, later moves the clock after the mover's last move
    windows = [180 // 30, 180 // 30, 180 // 30, 180 // 30, 170 // 30]
    assert got.tolist() == [int(b.B_OFF[s] + elo * b.N_W[s] + w) for w in windows]


def test_long_time_left_goes_into_the_last_step():
    got = b.move_buckets("bullet", 1000, 1000, np.array([600, 600]))
    assert got.tolist() == [int(b.B_OFF[0] + 2 * b.N_W[0] + b.N_W[0] - 1)] * 2


def test_group_is_speed_and_mean_elo():
    assert b.game_group("rapid", 1400, 1700) == 2 * b.N_E + 4          # mean 1550: 1500-1750
    assert b.game_group("bullet", 600, 700) == 0


def test_uniform_is_fixed_and_depends_on_the_salt():
    assert b.uniform("abc") == b.uniform("abc")
    assert b.uniform("abc") != b.uniform("abd")
    assert b.uniform("abc", "one") != b.uniform("abc", "two")
    values = [b.uniform(str(i)) for i in range(2000)]
    assert 0 <= min(values) and max(values) < 1
    assert abs(np.mean(values) - 0.5) < 0.03


def test_shrunk_pulls_small_buckets_to_the_parent():
    assert b.shrunk(0, 0, 0.05) == 0.05
    assert abs(b.shrunk(500, 1000, 0.05) - 0.5) < 0.03
