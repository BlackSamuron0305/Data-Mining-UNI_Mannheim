"""Step 1, reading the raw month: the file is cut into its zstd frames, and every frame is cleaned on its own."""
import hashlib
import importlib
import json
from compression import zstd

import pytest

from pgn_factory import lichess_game

storage = importlib.import_module("src.01_selection.storage")
cleanup = importlib.import_module("src.01_selection.cleanup")
SKIPPABLE = (0x184D2A50).to_bytes(4, "little")


@pytest.fixture
def month_file(tmp_path, monkeypatch):
    """Write bytes as the raw September file in a temporary folder; returns a function that does so."""
    monkeypatch.setenv("LICHESS_LOCAL_DIR", str(tmp_path))

    def write(data: bytes):
        path = tmp_path / storage.month_key("2026-09")
        path.write_bytes(data)
        return path
    return write


def test_frames_are_cut_where_zstd_cut_them(month_file):
    pieces = [b"first " * 1000, b"second " * 50_000, b"third"]
    skip = SKIPPABLE + (5).to_bytes(4, "little") + b"12345"
    data = zstd.compress(pieces[0]) + skip + zstd.compress(pieces[1]) + zstd.compress(pieces[2])
    path = month_file(data)
    month = storage.MonthFile("2026-09")
    frames = list(month.frames())
    assert [zstd.decompress(f) for f in frames] == pieces
    assert month.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_frames_with_a_limit(month_file):
    month_file(b"".join(zstd.compress(bytes([i]) * 100) for i in range(5)))
    month = storage.MonthFile("2026-09")
    assert len(list(month.frames(2))) == 2
    assert month.sha256 is None                          # not the whole file read


def test_an_incomplete_file_is_refused(month_file):
    month_file(zstd.compress(b"x" * 10_000) + zstd.compress(b"y" * 10_000)[:-3])
    with pytest.raises(ValueError, match="incomplete"):
        list(storage.MonthFile("2026-09").frames())


def test_other_data_is_refused(month_file):
    month_file(zstd.compress(b"x" * 100) + b"not a zstd frame at all")
    with pytest.raises(ValueError, match="no zstd frame"):
        list(storage.MonthFile("2026-09").frames())


def test_clean_text_counts_moves_and_blunders():
    text = lichess_game(6, site="g1", blunders=(1,)) + lichess_game(4, site="g2", title="BOT") + \
        lichess_game(10, site="g3", blunders=(4, 7))
    out = cleanup.clean_text(text, 1.0)
    assert int(out["group_games"].sum()) == 2            # the bot game is not a complete case
    assert int(out["moves"].sum()) == 16
    assert int(out["blunders"].sum()) == 3
    assert [game[0] for game in out["pool"]] == ["g1", "g3"]     # rate 1.0: every game into the pool
    assert int(out["b_moves"].sum()) == 16 and int(out["b_blunders"].sum()) == 3


def test_clean_text_keeps_no_pool_with_rate_zero():
    out = cleanup.clean_text(lichess_game(6) + lichess_game(8, site="b"), 0.0)
    assert out["pool"] == [] and int(out["moves"].sum()) == 14


def test_clean_frame_keeps_the_cut_games_for_the_neighbours():
    text = "".join(lichess_game(6, site=s) for s in ("first", "middle", "last"))
    out = cleanup.clean_frame(zstd.compress(text.encode()), 1.0)
    assert [game[0] for game in out["pool"]] == ["middle"]
    assert out["head"].decode().count("[Event ") == 1 and b"first" in out["head"]
    assert b"last" in out["tail"]


def test_cleanup_json_records_the_filters(tmp_path, monkeypatch):
    monkeypatch.setattr(cleanup, "MONTH_DIR", tmp_path)
    tot = cleanup.Total(None)
    tot.add(cleanup.clean_text(lichess_game(6), 0.0))

    class Month:
        location, size, sha256 = "test", 1, None
    cleanup.write(tot, Month(), 0.0, 1.0)
    meta = json.loads((tmp_path / "cleanup.json").read_text())
    assert meta["filters"] == cleanup.FILTERS
    assert meta["n_complete_games"] == 1 and meta["n_moves"] == 6
