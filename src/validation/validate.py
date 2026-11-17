"""Validation - score every training run on the same moves.

Usage:  uv run python -m src.validation.validate

Reads every run in data/runs/<method>/<name>/ (src/training/common.py) and scores its predictions of the held-out
training games (valid) and of the test games (test), see metrics.py. Runs can cover different moves (the engine
features exist only for the analysed games), so every run is scored twice:

own      on all the moves it predicted;
common   on the moves that every run predicted: only these scores can be compared between runs.

Against the baseline (the run bucket_rate/month, else any bucket_rate run) on the common moves: log_loss_gain = its
log loss - the run's (above 0 = better than the baseline), brier_skill_vs_baseline = 1 - brier / its brier.

Choose between runs on valid; report test.

Outputs (data/validation/)
results.csv          one row per run, part (valid, test) and moves (own, common)
by_speed.csv         the common scores per speed
calibration.csv      predicted against observed rate in 20 bins per run and part (common moves)
summary.md           the tables to read
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import FEATURES_DIR, RUNS_DIR, VALIDATION_DIR
from src.training.common import KEYS, TARGET
from src.validation.metrics import calibration, scores

PARTS = ("valid", "test")
SHOWN = ["log_loss", "log_loss_gain", "brier_skill", "brier_skill_vs_baseline", "roc_auc", "pr_auc", "ece"]


def find_runs() -> dict[str, dict]:
    runs = {}
    for meta in sorted(RUNS_DIR.glob("*/*/run.json")):
        info = json.loads(meta.read_text(encoding="utf-8"))
        runs[f"{meta.parent.parent.name}/{meta.parent.name}"] = {"dir": meta.parent, **info}
    return runs


def baseline_of(runs: dict) -> str | None:
    if "bucket_rate/month" in runs:
        return "bucket_rate/month"
    return next((r for r in runs if r.startswith("bucket_rate/")), None)


def common_moves(runs: dict, part: str) -> pd.DataFrame | None:
    """One row per move that every run predicted: keys, blunder and one column of predictions per run."""
    out = None
    for run, info in runs.items():
        df = pd.read_parquet(info["dir"] / f"predictions_{part}.parquet").rename(columns={"p": run})
        out = df if out is None else out.merge(df.drop(columns=TARGET), on=KEYS, how="inner")
    return out


def speeds(part: str) -> pd.DataFrame | None:
    path = FEATURES_DIR / "positions" / f"{'train' if part == 'valid' else 'test'}.parquet"
    return pd.read_parquet(path, columns=KEYS + ["speed"]) if path.exists() else None


def run() -> None:
    runs = find_runs()
    if not runs:
        raise FileNotFoundError(f"no runs in {RUNS_DIR}: train a model first (README.md)")
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    base = baseline_of(runs)
    results, by_speed, cal = [], [], []
    for part in PARTS:
        for run_name, info in runs.items():
            df = pd.read_parquet(info["dir"] / f"predictions_{part}.parquet")
            results.append({"run": run_name, "part": part, "moves_on": "own", **scores(df[TARGET], df["p"])})
        common = common_moves(runs, part)
        sp = speeds(part)
        if sp is not None:
            common = common.merge(sp, on=KEYS, how="left")
        y = common[TARGET].to_numpy()
        b = scores(y, common[base]) if base is not None else None
        for run_name in runs:
            s = scores(y, common[run_name])
            if b is not None:
                s["log_loss_gain"] = b["log_loss"] - s["log_loss"]
                s["brier_skill_vs_baseline"] = 1 - s["brier"] / b["brier"]
            results.append({"run": run_name, "part": part, "moves_on": "common", **s})
            c = calibration(y, common[run_name].to_numpy())
            cal.append(c.assign(run=run_name, part=part))
            if "speed" in common:
                for speed, g in common.groupby("speed", observed=True):
                    by_speed.append({"run": run_name, "part": part, "speed": speed,
                                     **scores(g[TARGET], g[run_name])})

    res = pd.DataFrame(results)
    res.to_csv(VALIDATION_DIR / "results.csv", index=False)
    pd.DataFrame(by_speed).to_csv(VALIDATION_DIR / "by_speed.csv", index=False)
    pd.concat(cal, ignore_index=True).to_csv(VALIDATION_DIR / "calibration.csv", index=False)

    lines = ["# Validation of every run", "",
             f"Baseline: {base or 'none (train bucket_rate first)'}. Scores on the moves that every run predicted "
             "(common); choose on valid, report test. log_loss_gain > 0 and brier_skill_vs_baseline > 0: better "
             "than the baseline.", ""]
    for part in reversed(PARTS):
        t = res[(res["part"] == part) & (res["moves_on"] == "common")].sort_values("log_loss")
        lines += [f"## {part} ({int(t['moves'].iloc[0]):,} moves, blunder rate {100 * t['rate'].iloc[0]:.2f}%)", "",
                  "| run | " + " | ".join(SHOWN) + " |", "|:--|" + "--:|" * len(SHOWN)]
        for _, r in t.iterrows():
            lines.append(f"| {r['run']} | " + " | ".join(
                "" if pd.isna(r.get(c, np.nan)) else f"{r[c]:.4f}" for c in SHOWN) + " |")
        lines.append("")
    own = res[res["moves_on"] == "own"]
    lines += ["## Moves per run (own)", "", "| run | valid | test |", "|:--|--:|--:|"]
    for run_name in runs:
        o = own[own["run"] == run_name].set_index("part")["moves"]
        lines.append(f"| {run_name} | {int(o['valid']):,} | {int(o['test']):,} |")
    (VALIDATION_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
