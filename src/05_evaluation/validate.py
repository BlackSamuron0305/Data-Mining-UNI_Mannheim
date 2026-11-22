"""Validation - score every training run on the same moves.

Usage:  uv run python -m src.05_evaluation.validate [--with SELECTION ...] [--best RUN] [--bootstrap N]

Scores every run in data/runs/<selection>/ on its own moves and on the moves that every run predicted, against the baseline
bucket_rate/month, and writes the tables, intervals, figures and error analysis to data/validation/<selection>/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import CFG, DATA, GAMES_SUB, RUNS_DIR, SELECTION, SELECTION_DIR, SELECTION_ROOT, VALIDATION_DIR
from src.config import KEYS, TARGET
from src.common.metrics import best_cutoff, calibration, decisions, move_loss, scores, top_name
from . import bootstrap, errors, figures
from .groups import DIMENSIONS, SOURCE, aligned, bands, positions

EV = CFG["evaluation"]
PARTS = ("valid", "test")
K = EV["top_share"]
TOP = top_name(K)
SHOWN = ["log_loss", "log_loss_gain", "brier_skill_vs_baseline", "roc_auc", "pr_auc", "ece", "f1", "mcc",
         f"recall_{TOP}", f"lift_{TOP}"]


def find_runs(runs_dir: Path = RUNS_DIR, label: str = "") -> dict[str, dict]:
    runs = {}
    for meta in sorted(runs_dir.glob("*/*/run.json")):
        info = json.loads(meta.read_text(encoding="utf-8"))
        name = f"{meta.parent.parent.name}/{meta.parent.name}"
        if label:
            name = f"{label}:{name}"
        runs[name] = {"dir": meta.parent, **info}
    return runs


def test_games(selection_dir: Path) -> set[str]:
    path = selection_dir / GAMES_SUB / "games.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run the selection first")
    g = pd.read_csv(path, usecols=["game_id", "split"])
    return set(g.loc[g["split"] == "test", "game_id"])


def other_runs(specs: list[str]) -> dict[str, dict]:
    """The runs of the selections given with --with, if they have the same test games as this one."""
    out = {}
    if not specs:
        return out
    mine = test_games(SELECTION_DIR)
    for label in specs:
        theirs = test_games(SELECTION_ROOT / label)
        if theirs != mine:
            raise ValueError(f"{label} has other test games than {SELECTION} ({len(theirs ^ mine):,} differ): the "
                             "runs cannot be compared.")
        runs_dir = DATA / "runs" / label
        runs = find_runs(runs_dir, label)
        if not runs:
            raise FileNotFoundError(f"no runs in {runs_dir}")
        out.update(runs)
    return out


def baseline_of(runs: dict) -> str | None:
    if "bucket_rate/month" in runs:
        return "bucket_rate/month"
    for r in runs:
        if r.startswith("bucket_rate/"):
            return r
    return None


def predictions(info: dict, part: str) -> pd.DataFrame:
    df = pd.read_parquet(info["dir"] / f"predictions_{part}.parquet", columns=KEYS + [TARGET, "p"])
    df["p"] = df["p"].astype(np.float32)
    return df


def common_moves(runs: dict, part: str) -> pd.DataFrame | None:
    """One row per move that every run predicted: keys, blunder and one column of predictions per run."""
    out = None
    for run, info in runs.items():
        df = predictions(info, part).rename(columns={"p": run})
        if out is None:
            out = df
        else:
            out = out.merge(df.drop(columns=TARGET), on=KEYS, how="inner")
    return out


def cutoffs(runs: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Per run: the cut-off with the highest F1 and the log loss, both on its own valid moves."""
    cut = {}
    loss = {}
    for run, info in runs.items():
        v = predictions(info, "valid")
        cut[run] = best_cutoff(v[TARGET], v["p"])
        loss[run] = float(move_loss(v[TARGET], v["p"]).mean())
    return cut, loss


def chosen(valid_loss: pd.Series, base: str | None, n: int) -> list[str]:
    """The baseline and the best run of every method by valid log loss, the best n methods."""
    rest = valid_loss.drop(base, errors="ignore")
    best_of_method = {}
    for run, loss in rest.items():
        method = run.rsplit("/", 1)[0]
        if method not in best_of_method or loss < rest[best_of_method[method]]:
            best_of_method[method] = run
    best_runs = [best_of_method[m] for m in sorted(best_of_method)]
    picked = rest[best_runs].sort_values().index[:n].tolist()
    if base is not None:
        return [base] + picked
    return picked


def subgroup_scores(common: pd.DataFrame, runs: list[str], base: str | None) -> pd.DataFrame:
    """Scores of the runs per subgroup (at least evaluation.min_moves moves)."""
    rows = []
    for dim in DIMENSIONS:
        if dim not in common:
            continue
        for group, g in common.groupby(dim, observed=True):
            if len(g) < EV["min_moves"] or not 0 < g[TARGET].mean() < 1:
                continue
            y = g[TARGET].to_numpy(dtype=float)
            b = np.nan
            if base is not None:
                b = move_loss(y, g[base]).mean()
            for run in runs:
                ll = move_loss(y, g[run]).mean()
                ap = average_precision_score(y, g[run])
                rows.append({"run": run, "dimension": dim, "group": str(group), "moves": len(g),
                             "rate": y.mean(), "mean_p": float(g[run].mean()), "log_loss": ll,
                             "gain_pct": 100 * (b - ll) / b, "pr_auc": ap, "pr_auc_over_rate": ap / y.mean()})
    return pd.DataFrame(rows)


def score_part(runs: dict, part: str, base: str | None, cut: dict) -> tuple[list, list, list, pd.DataFrame]:
    """Scores of every run on one part: on its own moves and on the common moves."""
    results, by_speed, cal = [], [], []
    for run_name, info in runs.items():
        df = predictions(info, part)
        row = {"run": run_name, "part": part, "moves_on": "own"}
        row.update(scores(df[TARGET], df["p"]))
        row.update(decisions(df[TARGET], df["p"], cut[run_name], K))
        results.append(row)

    common = common_moves(runs, part)
    if part == "test":
        pos = positions(part, SOURCE, common["game_id"])
    else:
        pos = positions(part, ["speed"], common["game_id"])
    if pos is not None:
        pos = aligned(common, pos)
        if set(SOURCE) <= set(pos.columns):
            common = common.join(bands(pos))
        else:
            common = common.join(pos[["speed"]])

    y = common[TARGET].to_numpy()
    b = None
    if base is not None:
        b = scores(y, common[base])
    for run_name in runs:
        s = scores(y, common[run_name])
        if b is not None:
            s["log_loss_gain"] = b["log_loss"] - s["log_loss"]
            s["brier_skill_vs_baseline"] = 1 - s["brier"] / b["brier"]
        row = {"run": run_name, "part": part, "moves_on": "common"}
        row.update(s)
        row.update(decisions(y, common[run_name], cut[run_name], K))
        results.append(row)
        c = calibration(y, common[run_name].to_numpy(dtype=float))
        cal.append(c.assign(run=run_name, part=part))
        if "speed" in common:
            for speed, g in common.groupby("speed", observed=True):
                speed_row = {"run": run_name, "part": part, "speed": speed}
                speed_row.update(scores(g[TARGET], g[run_name]))
                by_speed.append(speed_row)
    return results, by_speed, cal, common


def fmt(v) -> str:
    if pd.isna(v):
        return ""
    return f"{v:.4f}"


def table(res: pd.DataFrame, part: str) -> list[str]:
    t = res[(res["part"] == part) & (res["moves_on"] == "common")].sort_values("log_loss")
    lines = [f"## {part} ({int(t['moves'].iloc[0]):,} moves, blunder rate {100 * t['rate'].iloc[0]:.2f}%)", "",
             "| run | " + " | ".join(SHOWN) + " |", "|:--|" + "--:|" * len(SHOWN)]
    for _, r in t.iterrows():
        values = [fmt(r.get(c, np.nan)) for c in SHOWN]
        lines.append(f"| {r['run']} | " + " | ".join(values) + " |")
    lines.append("")
    return lines


def interval_table(iv: pd.DataFrame, order: list[str]) -> list[str]:
    cols = []
    for c in ("log_loss", "log_loss_gain", "pr_auc"):
        if c in set(iv["score"]):
            cols.append(c)
    lines = ["| run | " + " | ".join(cols) + " |", "|:--|" + "--:|" * len(cols)]
    w = iv.set_index(["run", "score"])
    for run in order:
        values = []
        for c in cols:
            values.append(f"{w.loc[(run, c), 'estimate']:.4f} ({w.loc[(run, c), 'lo']:.4f} to "
                          f"{w.loc[(run, c), 'hi']:.4f})")
        lines.append(f"| {run} | " + " | ".join(values) + " |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--with", dest="others", action="append", default=[], metavar="SELECTION",
                    help="also score the runs of another selection, e.g. balanced (test part only)")
    ap.add_argument("--best", metavar="RUN", help="the run of the error analysis, e.g. gradient_boosting/context "
                    "(default: the lowest valid log loss)")
    ap.add_argument("--bootstrap", type=int, metavar="N", help=f"resamples of the test games for the intervals "
                    f"(default {EV['bootstrap']}; 0 = none)")
    args = ap.parse_args()
    run(args.others, args.best, args.bootstrap)


def run(others: list[str] | None = None, best: str | None = None, n_boot: int | None = None) -> None:
    others = others or []
    if n_boot is None:
        n_boot = EV["bootstrap"]
    runs = find_runs()
    if not runs:
        raise FileNotFoundError(f"no runs in {RUNS_DIR}: train a model first (README.md)")
    runs.update(other_runs(others))
    if best is not None and best not in runs:
        raise ValueError(f"--best {best!r} is not a run; runs: {', '.join(runs)}")

    if others:
        # the other folders have other training games: only the test part can be compared
        parts = ("test",)
        labels = others
        out_dir = VALIDATION_DIR / ("with_" + "_".join(labels))
    else:
        parts = PARTS
        out_dir = VALIDATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    base = baseline_of(runs)
    cut, own_valid_loss = cutoffs(runs)
    results, by_speed, cal = [], [], []
    for part in parts:
        r, s, c, common = score_part(runs, part, base, cut)
        results += r
        by_speed += s
        cal += c
    # from here on, common holds the common test moves (test is the last part)
    y = common[TARGET].to_numpy()

    res = pd.DataFrame(results)
    res.to_csv(out_dir / "results.csv", index=False)
    pd.DataFrame(by_speed).to_csv(out_dir / "by_speed.csv", index=False)
    pd.concat(cal, ignore_index=True).to_csv(out_dir / "calibration.csv", index=False)

    vc = res[(res["part"] == "valid") & (res["moves_on"] == "common")]
    if len(vc):
        valid_loss = vc.set_index("run")["log_loss"]
    else:
        valid_loss = pd.Series(own_valid_loss)
    drawn = chosen(valid_loss, base, EV["plot_runs"])
    by_flag = best is not None
    if best is None:
        best = valid_loss.idxmin()

    iv = None
    if n_boot > 0:
        preds = {}
        for r in runs:
            preds[r] = common[r].to_numpy()
        iv = bootstrap.intervals(y, common["game_id"].to_numpy(), preds, base, n_boot, EV["bootstrap_bins"])
        iv.to_csv(out_dir / "intervals.csv", index=False)
    sub = None
    if "phase" in common:
        sub = subgroup_scores(common, drawn, base)
        sub.to_csv(out_dir / "by_subgroup.csv", index=False)
    drawn_preds = {}
    for r in drawn:
        drawn_preds[r] = common[r].to_numpy()
    figs = figures.draw(y, drawn_preds, base, cut, sub, K, out_dir / "figures")
    del common

    own = predictions(runs[best], "test")
    pos = positions("test", SOURCE + errors.FEATURES)
    if pos is not None:
        own = own.join(aligned(own, pos))
        if set(SOURCE) <= set(own.columns):
            own = own.drop(columns="speed").join(bands(own))
    errors.analyse(best, own, cut[best], out_dir / "errors")
    del own, pos

    lines = ["# Validation of every run", "",
             f"Baseline: {base or 'none (train bucket_rate first)'}. Scores on the moves that every run predicted "
             "(common); choose on valid, report test. log_loss_gain > 0 and brier_skill_vs_baseline > 0: better "
             f"than the baseline. f1 and mcc at the run's cut-off, chosen on its own valid moves; recall_{TOP} and "
             f"lift_{TOP}: blunders caught among the {100 * K:g}% riskiest moves, and how many times the blunder "
             "rate they have.", ""]
    if others:
        lines += [f"Runs of other selections: {', '.join(others)}. Same test games as {SELECTION}, other training "
                  "games, so only the test part is scored.", ""]
    for part in reversed(parts):
        lines += table(res, part)
    if iv is not None:
        test_common = res[(res["part"] == "test") & (res["moves_on"] == "common")]
        order = test_common.sort_values("log_loss")["run"].tolist()
        lines += [f"## 95% intervals (test, {n_boot:,} resamples of whole games)", ""]
        lines += interval_table(iv, order) + [""]

    t = res[(res["run"] == best) & (res["part"] == "test") & (res["moves_on"] == "own")].iloc[0]
    how = "Chosen with --best" if by_flag else "Lowest valid log loss"
    lines += [f"## Best run: {best}", "",
              f"{how}. On its "
              f"own test moves at the cut-off {t['cutoff']:.4f}: {t['flagged']:.1%} flagged, recall {t['recall']:.1%}, "
              f"precision {t['precision']:.1%}, MCC {t['mcc']:.3f}. Error analysis: errors/summary.md.", "",
              f"Figures (figures/, {', '.join(drawn)}): {', '.join(figs)}.", ""]

    lines += ["## Moves per run (own)", "", "| run | " + " | ".join(parts) + " |", "|:--|" + "--:|" * len(parts)]
    own_res = res[res["moves_on"] == "own"]
    for run_name in runs:
        o = own_res[own_res["run"] == run_name].set_index("part")["moves"]
        counts = [f"{int(o[p]):,}" for p in parts]
        lines.append(f"| {run_name} | " + " | ".join(counts) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
