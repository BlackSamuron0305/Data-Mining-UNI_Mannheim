"""Fill data/ as config.yaml says (data:): download the chosen state from our server, then compute every step after
it, up to step 3 (transformation). Steps 4 and 5 follow with: bash run_models.sh

Usage:  uv run python -m src.pipeline [--dry-run]

A step is skipped if its result is there and newer than everything it reads; so a stopped run continues, and after
a change only the steps that depend on it run again. Steps 2 and 3 run for the stratified run and for the balanced run
(stratified_balanced, the balancing experiment).
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

from src.config import (BLUNDER_RATES_DIR, CFG, CLEANUP_DIR, COMPARISON_DIR, DATA, DOWNLOADED, LOGS_SUB, MONTH_NAMES,
                        PREPROCESSING_ROOT, RAW_FROM, SELECTION_ROOT, START_WITH, STATES, STOCKFISH,
                        TRANSFORMATION_ROOT)

filters = importlib.import_module("src.01_selection.preparation.cleanup.filters")
RUNS = ["stratified", "stratified_balanced"]
SELECTIONS = ["random", "stratified", "stratified_balanced"]
# a second month, only counted, to compare its blunder probability with ours
OTHER = CFG["compare_month"]
OTHER_CLEANUP = CLEANUP_DIR.parent / MONTH_NAMES[OTHER] / "cleanup.json"
OTHER_RATES = BLUNDER_RATES_DIR.parent / MONTH_NAMES[OTHER] / "rates.csv"
MONTHS = BLUNDER_RATES_DIR.parent / "months" / "months.csv"


def mtime(path: Path) -> float | None:
    return path.stat().st_mtime if path.exists() else None


def newest(folders: list[Path]) -> float | None:
    times = [f.stat().st_mtime for folder in folders for f in folder.rglob("*.parquet")]
    return max(times, default=None)


def cleanup_matches(meta: Path = CLEANUP_DIR / "cleanup.json") -> bool:
    """The cleanup on this computer used the filters of src/settings.py."""
    return meta.exists() and filters.same_filters(json.loads(meta.read_text(encoding="utf-8")))


def leakage_checked(log: Path) -> bool:
    return log.exists() and "no leakage found" in log.read_text(encoding="utf-8")


@dataclass
class Step:
    state: str                              # the state its result belongs to
    name: str
    module: str
    selection: str | None
    stamp: Callable[[], float | None]       # when its result was written (None: not there)
    reads: list[str] = field(default_factory=list)   # the steps whose results it reads
    ok: Callable[[], bool] = lambda: True   # a further condition for being done
    always: bool = False                    # runs every time (it continues by itself where it stopped)
    env: dict[str, str] = field(default_factory=dict)   # environment of its run
    args: list[str] = field(default_factory=list)       # command-line options

    @property
    def key(self) -> str:
        return f"{self.name} ({self.selection})" if self.selection else self.name


def steps() -> dict[str, Step]:
    out = [Step("selected", "cleanup of the month", "src.01_selection.preparation.cleanup.cleanup", None,
                lambda: mtime(CLEANUP_DIR / "cleanup.json"), ok=cleanup_matches),
           Step("selected", "blunder rates", "src.01_selection.preparation.blunder_rates.rates", None,
                lambda: mtime(BLUNDER_RATES_DIR / "rates.csv"), ["cleanup of the month"])]
    for s in SELECTIONS:
        out.append(Step("selected", f"{s} selection", f"src.01_selection.sampling.{s}.select", None,
                        lambda s=s: mtime(SELECTION_ROOT / s / "selection.json"), ["cleanup of the month"]))
    out.append(Step("selected", "comparison of the selections", "src.01_selection.sampling.comparison.compare", None,
                    lambda: mtime(COMPARISON_DIR / "comparison.json"),
                    ["blunder rates"] + [f"{s} selection" for s in SELECTIONS]))
    month = {"IE500_MONTH": OTHER}
    out += [Step("selected", f"cleanup of {OTHER} (counts only)", "src.01_selection.preparation.cleanup.cleanup", None,
                 lambda: mtime(OTHER_CLEANUP), ok=lambda: cleanup_matches(OTHER_CLEANUP), env=month,
                 args=["--no-pool"]),
            Step("selected", f"blunder rates of {OTHER}", "src.01_selection.preparation.blunder_rates.rates", None,
                 lambda: mtime(OTHER_RATES), [f"cleanup of {OTHER} (counts only)"], env=month),
            Step("selected", "blunder rates by month", "src.01_selection.preparation.blunder_rates.months", None,
                 lambda: mtime(MONTHS), ["blunder rates", f"blunder rates of {OTHER}"])]
    engine_dirs = [PREPROCESSING_ROOT / r / "with_stockfish" / "stockfish" for r in RUNS]
    for run in RUNS:
        pre = PREPROCESSING_ROOT / run
        # the balanced run takes the leakage check and the column decisions of the stratified run
        first = run == RUNS[0]
        extract = f"extraction ({run})"
        out.append(Step("preprocessed", "extraction", "src.02_preprocessing.common.extract", run,
                        lambda pre=pre: mtime(pre / "extracted" / "test.parquet"), [f"{run} selection"]))
        if first:
            log = pre / LOGS_SUB / "leakage_check.log"
            out.append(Step("preprocessed", "leakage check", "src.02_preprocessing.common.leakage_check", run,
                            lambda log=log: mtime(log), [extract], ok=lambda log=log: leakage_checked(log)))
        for table in ("without_stockfish", "with_stockfish"):
            reads = [f"without_stockfish: preprocess ({run})"] if table == "with_stockfish" else [extract]
            if table == "with_stockfish" and STOCKFISH == "compute":
                out.append(Step("preprocessed", "Stockfish", "src.02_preprocessing.with_stockfish.engine", run,
                                lambda: newest(engine_dirs), [f"{run} selection"], always=True))
                reads.append(f"Stockfish ({run})")
            out.append(Step("preprocessed", f"{table}: preprocess", f"src.02_preprocessing.{table}.preprocess", run,
                            lambda p=pre / table: mtime(p / "clean" / "test.parquet"), reads))
            before = f"{table}: preprocess ({run})"
            if first:
                out.append(Step("preprocessed", f"{table}: inspection", f"src.02_preprocessing.{table}.inspection",
                                run, lambda p=pre / table: mtime(p / "inspection" / "inspection.json"), [before]))
            # the reduction with Stockfish repeats the one without; the balanced run repeats the decisions of the
            # stratified run
            reads = [before]
            if table == "with_stockfish":
                reads.append(f"without_stockfish: reduce ({run})")
            if not first:
                reads.append(f"{table}: reduce ({RUNS[0]})")
            out.append(Step("preprocessed", f"{table}: reduce", f"src.02_preprocessing.{table}.reduce", run,
                            lambda p=pre / table: mtime(p / "final" / "removed.csv"), reads))
    for run in RUNS:
        out.append(Step("transformed", "transformation", "src.03_transformation.transform", run,
                        lambda run=run: mtime(TRANSFORMATION_ROOT / run / "transform.json"),
                        [f"{t}: reduce ({run})" for t in ("without_stockfish", "with_stockfish")]))
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
        sys.exit(f"{step.key} failed (python -m {step.module}, exit code {result.returncode}); see its log in data/")
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
    if "preprocessed" in later:
        print(f"Stockfish columns: {STOCKFISH}")
    for step in todo:
        mark = "always" if step.always else "done" if is_done(step, every, keys) else "to do"
        print(f"  [{mark:6s}] {step.key}")
    if args.dry_run:
        return

    if DOWNLOADED:
        from src.common.server import download
        download(DOWNLOADED)
    if START_WITH != "raw" and (CLEANUP_DIR / "cleanup.json").exists() and not cleanup_matches():
        sys.exit("the selected games on this computer come from a cleanup with other filters than src/settings.py: "
                 "start with raw, or download a state made with the current filters")
    for step in todo:
        if is_done(step, every, keys):
            print(f"--- skip {step.key}: already in {DATA.name}/ and up to date")
            continue
        print(f"--- {step.key}: python -m {step.module}")
        run_step(step)
    print("data/ is ready up to step 3. Next: bash run_models.sh (training and evaluation)")


if __name__ == "__main__":
    main()
