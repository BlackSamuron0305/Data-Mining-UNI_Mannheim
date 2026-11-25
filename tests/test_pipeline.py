"""The runner (src/pipeline.py): a step is done only if its result is newer than what it reads; one step alone runs
only its own parts."""
import os
import subprocess
import sys

import pytest

import src.pipeline as pipeline
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


def test_without_a_step_the_plan_follows_config(monkeypatch):
    monkeypatch.setattr(pipeline, "START_WITH", "01_selection")
    monkeypatch.setattr(pipeline, "DOWNLOADED", ["01_selection"])
    assert pipeline.plan() == (["02_preprocessing", "03_transformation"], ["01_selection"])


def test_one_step_is_computed_whatever_the_start(monkeypatch):
    monkeypatch.setattr(pipeline, "START_WITH", "03_transformation")
    monkeypatch.setattr(pipeline, "DOWNLOADED", ["03_transformation"])
    assert pipeline.plan("01_selection") == (["01_selection"], [])


def test_one_step_downloads_only_the_earlier_steps(monkeypatch):
    monkeypatch.setattr(pipeline, "DOWNLOADED", ["01_selection", "02_preprocessing"])
    assert pipeline.plan("02_preprocessing") == (["02_preprocessing"], ["01_selection"])


def test_an_unknown_step_is_refused():
    with pytest.raises(ValueError):
        pipeline.plan("04_data_mining")


@pytest.mark.parametrize("step, own, other", [("01_selection", "cleanup of the month", "extraction"),
                                              ("02_preprocessing", "extraction", "cleanup of the month")])
def test_the_command_of_a_step_lists_only_its_parts(step, own, other):
    out = subprocess.run([sys.executable, "-m", f"src.{step}", "--dry-run"], capture_output=True, text=True,
                         cwd=pipeline.DATA.parent, check=True).stdout
    assert f"computed: {step}" in out
    assert own in out and other not in out
