"""Fill data/ as config.yaml says (data:): download the chosen step from our server, then compute every step after
it, up to step 3 (transformation). Steps 4 and 5 follow with: bash run_models.sh

Usage:  uv run python -m src.pipeline [--dry-run]

A part is skipped if its result is there and newer than everything it reads; so a stopped run continues, and after
a change only the parts that depend on it run again. Steps 2 and 3 run for the stratified run and for the balanced
run (stratified_balanced, the balancing experiment).
"""
import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.config import (CFG, COMPARISON_DIR, DATA, DOWNLOADED, MONTH_DIR, MONTH_NAMES, PREPROCESSING_ROOT, RAW_FROM,
                        SELECTION_ROOT, SELECTIONS, START_WITH, STATES, STOCKFISH, STOCKFISH_DIR, TRANSFORMATION_ROOT)

filters = importlib.import_module("src.01_selection.filters")
RUNS = ["stratified", "stratified_balanced"]
# a second month, only counted, to compare its blunder probability with ours
OTHER = CFG["compare_month"]
OTHER_DIR = SELECTION_ROOT / MONTH_NAMES[OTHER]


def mtime(path: Path) -> float | None:
    return path.stat().st_mtime if path.exists() else None


def cleanup_matches(meta: Path = MONTH_DIR / "cleanup.json") -> bool:
    """The cleanup on this computer used the filters of src/settings.py."""
    return meta.exists() and filters.same_filters(json.loads(meta.read_text(encoding="utf-8")))


def leakage_checked(path: Path) -> bool:
    return path.exists() and json.loads(path.read_text(encoding="utf-8")).get("no_leakage") is True


@dataclass
class Step:
    state: str                              # the step its result belongs to
    name: str
    module: str
    selection: str | None
    stamp: Callable[[], float | None]       # when its result was written (None: not there)
    reads: list[str] = field(default_factory=list)   # the parts whose results it reads
    ok: Callable[[], bool] = lambda: True   # a further condition for being done
    always: bool = False                    # runs every time (it continues by itself where it stopped)
    env: dict[str, str] = field(default_factory=dict)   # environment of its run
    args: list[str] = field(default_factory=list)       # command-line options

    @property
    def key(self) -> str:
        return f"{self.name} ({self.selection})" if self.selection else self.name


def steps() -> dict[str, Step]:
    sel = "01_selection"
    out = [Step(sel, "cleanup of the month", "src.01_selection.cleanup", None,
                lambda: mtime(MONTH_DIR / "cleanup.json"), ok=cleanup_matches),
           Step(sel, "blunder rates", "src.01_selection.rates", None,
                lambda: mtime(MONTH_DIR / "rates.csv"), ["cleanup of the month"])]
    for s in SELECTIONS:
        out.append(Step(sel, f"{s} selection", "src.01_selection.select", None,
                        lambda s=s: mtime(SELECTION_ROOT / s / "selection.json"), ["cleanup of the month"],
                        args=["--run", s]))
    out.append(Step(sel, "comparison of the selections", "src.01_selection.compare", None,
                    lambda: mtime(COMPARISON_DIR / "comparison.json"),
                    ["blunder rates"] + [f"{s} selection" for s in SELECTIONS]))
    month = {"IE500_MONTH": OTHER}
    out += [Step(sel, f"cleanup of {OTHER} (counts only)", "src.01_selection.cleanup", None,
                 lambda: mtime(OTHER_DIR / "cleanup.json"), ok=lambda: cleanup_matches(OTHER_DIR / "cleanup.json"),
                 env=month, args=["--no-pool"]),
            Step(sel, f"blunder rates of {OTHER}", "src.01_selection.rates", None,
                 lambda: mtime(OTHER_DIR / "rates.csv"), [f"cleanup of {OTHER} (counts only)"], env=month),
            Step(sel, "blunder rates by month", "src.01_selection.months", None,
                 lambda: mtime(COMPARISON_DIR / "months.csv"), ["blunder rates", f"blunder rates of {OTHER}"])]

    pre = "02_preprocessing"
    for run in RUNS:
        work = PREPROCESSING_ROOT / "work" / run
        final = PREPROCESSING_ROOT / run
        first = run == RUNS[0]
        out.append(Step(pre, "extraction", "src.02_preprocessing.extract", run,
                        lambda w=work: mtime(w / "extracted_test.parquet"), [f"{run} selection"]))
        # the balanced run takes the leakage check and the column decisions of the stratified run
        if first:
            check = final / "leakage_check.json"
            out.append(Step(pre, "leakage check", "src.02_preprocessing.leakage_check", run,
                            lambda c=check: mtime(c), [f"extraction ({run})"], ok=lambda c=check: leakage_checked(c)))
        reads = [f"extraction ({run})"]
        if STOCKFISH == "compute":
            out.append(Step(pre, "Stockfish", "src.02_preprocessing.stockfish", run,
                            lambda: mtime(STOCKFISH_DIR / "test.parquet"), [f"{run} selection"], always=True))
            reads.append(f"Stockfish ({run})")
        out.append(Step(pre, "preprocess", "src.02_preprocessing.preprocess", run,
                        lambda w=work: mtime(w / "clean_test.parquet"), reads))
        if first:
            out.append(Step(pre, "inspection", "src.02_preprocessing.inspection", run,
                            lambda f=final: mtime(f / "inspection" / "inspection.json"), [f"preprocess ({run})"]))
            reads = [f"inspection ({run})"]
        else:
            reads = [f"preprocess ({run})", f"reduce ({RUNS[0]})"]
        out.append(Step(pre, "reduce", "src.02_preprocessing.reduce", run,
                        lambda f=final: mtime(f / "removed.csv"), reads))

    for run in RUNS:
        out.append(Step("03_transformation", "transformation", "src.03_transformation.transform", run,
                        lambda run=run: mtime(TRANSFORMATION_ROOT / run / "transform.json"), [f"reduce ({run})"]))
    return {s.key: s for s in out}


def is_done(step: Step, every: dict[str, Step], todo: set[str]) -> bool:
    """Its result is there and newer than the results it reads (those computed here must be done themselves)."""
    if step.always:
        return False
    written = step.stamp()
    if written is None or not step.ok():
        return False
    for key in step.reads:
        read = every[key]
        if key in todo and not read.always and not is_done(read, every, todo):
            return False
        read_written = read.stamp()
        if read_written is not None and read_written > written:
            return False
    return True


def run_step(step: Step) -> None:
    env = {**os.environ, **step.env}
    if step.selection:
        env["IE500_SELECTION"] = step.selection
    t0 = time.time()
    result = subprocess.run([sys.executable, "-m", step.module, *step.args], env=env)
    if result.returncode != 0:
        sys.exit(f"{step.key} failed (python -m {step.module}, exit code {result.returncode}); see its log in "
                 f"data/{step.state}/logs/")
    print(f"--- {step.key} done in {(time.time() - t0) / 60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="only show what would be downloaded and computed")
    args = ap.parse_args()

    later = STATES[STATES.index(START_WITH) + 1:]
    every = steps()
    todo = [s for s in every.values() if s.state in later]
    keys = {s.key for s in todo}
    print(f"start with: {START_WITH}; downloaded: {', '.join(DOWNLOADED) or 'nothing'}; "
          f"computed: {', '.join(later) or 'nothing'}")
    if START_WITH == "raw":
        print(f"raw month from {RAW_FROM}")
    if "02_preprocessing" in later:
        print(f"Stockfish columns: {STOCKFISH}")
    for step in todo:
        mark = "always" if step.always else "done" if is_done(step, every, keys) else "to do"
        print(f"  [{mark:6s}] {step.key}")
    if args.dry_run:
        return

    if DOWNLOADED:
        from src.common.server import download
        download(DOWNLOADED)
    if START_WITH != "raw" and (MONTH_DIR / "cleanup.json").exists() and not cleanup_matches():
        sys.exit("the selected games on this computer come from a cleanup with other filters than src/settings.py: "
                 "start with raw, or download a step made with the current filters")
    for step in todo:
        if is_done(step, every, keys):
            print(f"--- skip {step.key}: already in {DATA.name}/ and up to date")
            continue
        print(f"--- {step.key}: python -m {step.module} {' '.join(step.args)}".rstrip())
        run_step(step)
    print("data/ is ready up to step 3. Next: bash run_models.sh (training and evaluation)")


if __name__ == "__main__":
    main()
