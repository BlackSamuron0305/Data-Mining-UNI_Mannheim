"""The runner (src/pipeline.py): a step is done only if its result is newer than what it reads."""
import os

from src.pipeline import Step, is_done, mtime


def chain(tmp_path):
    files = [tmp_path / name for name in "abc"]
    for i, f in enumerate(files):
        f.write_text("x")
        os.utime(f, (1000 + i, 1000 + i))
    a, b, c = files
    steps = {"A": Step("x", "A", "m", None, lambda: mtime(a)),
             "B": Step("x", "B", "m", None, lambda: mtime(b), ["A"]),
             "C": Step("x", "C", "m", None, lambda: mtime(c), ["B"])}
    return files, steps


def done(steps):
    return {k: is_done(s, steps, set(steps)) for k, s in steps.items()}


def test_all_done_in_order(tmp_path):
    _, steps = chain(tmp_path)
    assert done(steps) == {"A": True, "B": True, "C": True}


def test_a_newer_input_makes_the_later_steps_stale(tmp_path):
    (a, _, _), steps = chain(tmp_path)
    os.utime(a, (2000, 2000))
    assert done(steps) == {"A": True, "B": False, "C": False}


def test_a_missing_result_makes_the_later_steps_stale(tmp_path):
    (_, b, _), steps = chain(tmp_path)
    b.unlink()
    assert done(steps) == {"A": True, "B": False, "C": False}


def test_a_failed_check_is_not_done(tmp_path):
    (a, _, _), steps = chain(tmp_path)
    steps["A"].ok = lambda: False
    assert done(steps)["A"] is False


def test_an_always_step_runs_every_time(tmp_path):
    _, steps = chain(tmp_path)
    steps["A"].always = True
    assert done(steps)["A"] is False
