"""Hyperparameter search of a method on the held-out games, then a refit of the best settings as a normal run.

Usage:  uv run python -m src.04_data_mining.tune --method METHOD [--features context] [--sampler tpe|grid] [--trials N]
            [--objective log_loss] [--tune-moves N] [--cv K] [--params-from SELECTION] [--max-games N] [--name NAME]

One fixed set of held-out games scores every trial; --cv K afterwards checks the best against the second-best
settings on K folds. --params-from skips the search and refits the settings of the same run of another selection.
"""
import argparse
import importlib
import json
import math
import time

import pandas as pd

from src.config import RUNS_ROOT, SEED
from src.common.buckets import uniform
from src.common.metrics import scores
from .common import (FEATURE_SETS, TUNING, cap_moves, columns, folds, load, predictions, run_dir,
                                 save_run, target)

METHODS = ["naive_bayes", "logistic_regression", "decision_tree", "random_forest", "gradient_boosting", "xgboost",
           "neural_network", "svm"]
LOWER_IS_BETTER = {"log_loss", "brier", "ece"}
OBJECTIVES = ["log_loss", "pr_auc", "roc_auc", "brier", "brier_skill", "ece"]


def method_module(method):
    """The train.py of a method (fit, DEFAULTS, SPACE, GRID)."""
    return importlib.import_module(f"src.04_data_mining.{method}.train")


def suggest(trial, mod, sampler):
    """The settings of one trial, from GRID or from the ranges in SPACE."""
    params = {}
    if sampler == "grid":
        for k, values in mod.GRID.items():
            params[k] = trial.suggest_categorical(k, values)
        return params
    for k, spec in mod.SPACE.items():
        if spec[0] == "float":
            params[k] = trial.suggest_float(k, spec[1], spec[2], log=spec[3])
        elif spec[0] == "int":
            params[k] = trial.suggest_int(k, spec[1], spec[2], log=spec[3])
        else:
            params[k] = trial.suggest_categorical(k, spec[1])
    return params


def tune(method, fit_df, valid, cols, sampler, trials, objective, csv_path):
    """Run the trials; return the best settings (with the defaults) and the table of trials."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mod = method_module(method)
    if sampler == "grid":
        trials = min(trials, math.prod(len(v) for v in mod.GRID.values()))
        optuna_sampler = optuna.samplers.GridSampler(mod.GRID, seed=SEED)
    else:
        optuna_sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=min(10, max(trials // 3, 1)))
    direction = "minimize" if objective in LOWER_IS_BETTER else "maximize"
    study = optuna.create_study(direction=direction, sampler=optuna_sampler)
    y_valid = target(valid)
    rows = []

    def run_trial(trial):
        t0 = time.time()
        params = dict(mod.DEFAULTS)
        params.update(suggest(trial, mod, sampler))
        try:
            m = mod.fit(fit_df, valid, cols, **params)
            s = scores(y_valid, m.predict(valid))
        except Exception as e:
            row = {"trial": trial.number, **params, "error": repr(e), "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"trial {trial.number}: failed ({e!r})")
            raise
        row = {"trial": trial.number, **params}
        for k in OBJECTIVES:
            row[k] = s[k]
        row.update(m.info)
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        shown = []
        for k, v in params.items():
            if k in mod.SPACE or k in mod.GRID:
                shown.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
        text = ", ".join(shown)
        for k, v in m.info.items():
            text += f", {k} {v}"
        print(f"trial {trial.number:3d}: {objective} {s[objective]:.5f} ({text}), {time.time() - t0:.0f} s")
        return s[objective]

    study.optimize(run_trial, n_trials=trials, catch=(Exception,))
    best = dict(mod.DEFAULTS)
    best.update(study.best_trial.params)
    return best, pd.DataFrame(rows)


def cv_check(method, tune_df, cols, table, objective, k, csv_path):
    """Fit the best and the second-best settings of the search on k folds of whole games and compare them."""
    mod = method_module(method)
    ok = table
    if "error" in table:
        ok = table[table["error"].isna()]
    ok = ok.sort_values(objective, ascending=objective in LOWER_IS_BETTER)
    keys = [key for key in {**mod.SPACE, **mod.GRID} if key in ok.columns]

    # the two best distinct settings
    candidates = {}
    seen = []
    for _, r in ok.iterrows():
        params = dict(mod.DEFAULTS)
        for c in keys:
            params[c] = r[c]
        for c, v in params.items():
            if type(mod.DEFAULTS.get(c)) is int:
                params[c] = int(v)
        signature = sorted((c, str(v)) for c, v in params.items())
        if signature not in seen:
            seen.append(signature)
            label = "best" if len(candidates) == 0 else "second"
            candidates[label] = params
        if len(candidates) == 2:
            break

    fold_of = {}
    for g in tune_df["game_id"].unique():
        fold_of[g] = int(uniform(g, "ie500-cv") * k)
    fold = tune_df["game_id"].map(fold_of).to_numpy()
    rows = []
    for i in range(k):
        fit_i = tune_df[fold != i].reset_index(drop=True)
        hold_i = tune_df[fold == i].reset_index(drop=True)
        for label, params in candidates.items():
            t0 = time.time()
            m = mod.fit(fit_i, hold_i, cols, **params)
            s = scores(target(hold_i), m.predict(hold_i))
            row = {"setting": label, "fold": i}
            for c in keys:
                row[c] = params[c]
            for o in OBJECTIVES:
                row[o] = s[o]
            row["seconds"] = round(time.time() - t0, 1)
            rows.append(row)
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    t = pd.DataFrame(rows)
    print(f"cv check ({k} folds, {objective}):")
    print(t.groupby("setting")[objective].agg(["mean", "std"]).to_string())
    if len(candidates) == 2:
        w = t.pivot(index="fold", columns="setting", values=objective)
        if objective in LOWER_IS_BETTER:
            better = w["best"] < w["second"]
        else:
            better = w["best"] > w["second"]
        print(f"the best settings of the search are better than the second-best in {int(better.sum())} of {k} folds")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--features", choices=FEATURE_SETS, default="context")
    ap.add_argument("--sampler", choices=["tpe", "grid"], default=TUNING.get("sampler", "tpe"))
    ap.add_argument("--trials", type=int, default=None, help="default: tuning.trials of the method (grid: all)")
    ap.add_argument("--objective", choices=OBJECTIVES, default=TUNING.get("objective", "log_loss"))
    ap.add_argument("--tune-moves", type=int, default=TUNING.get("tune_moves"),
                    help="fitting moves per trial (whole games); 0 = all")
    ap.add_argument("--cv", type=int, default=TUNING.get("cv_check", 0),
                    help="folds of the check of the best settings (0 = none)")
    ap.add_argument("--params-from", metavar="SELECTION", default=None,
                    help="no trials: refit the settings of the same run of this selection (e.g. stratified)")
    ap.add_argument("--max-games", type=int, default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    t0 = time.time()
    name = args.name or f"{args.features}_tuned"
    trials = args.trials or (TUNING.get("trials") or {}).get(args.method, 20)
    if args.sampler == "grid" and args.trials is None:
        trials = 10 ** 6
    refit_moves = (TUNING.get("refit_max_moves") or {}).get(args.method)
    cols = columns(args.features)

    fit_df, valid = folds(load("train", args.features, args.max_games))
    if args.params_from:
        run_folder = RUNS_ROOT / args.params_from / args.method / name
        best = json.loads((run_folder / "run.json").read_text(encoding="utf-8"))["params"]
        table = pd.DataFrame()
        print(f"{args.method} on {args.features}: settings of {run_folder}: {best}")
    else:
        tune_df = cap_moves(fit_df, args.tune_moves or None, "ie500-tune")
        print(f"{args.method} on {args.features}: trials on {len(tune_df):,} of {len(fit_df):,} fitting moves "
              f"({tune_df['game_id'].nunique():,} games), scored by {args.objective} on {len(valid):,} held-out moves")
        csv_path = run_dir(args.method, name) / "tuning.csv"
        best, table = tune(args.method, tune_df, valid, cols, args.sampler, trials, args.objective, csv_path)
        if args.cv and args.cv > 1:
            cv_check(args.method, tune_df, cols, table, args.objective, args.cv, csv_path.with_name("cv_check.csv"))
        del tune_df
    print(f"best: {best}")

    refit_df = cap_moves(fit_df, refit_moves or None, "ie500-refit")
    del fit_df
    m = method_module(args.method).fit(refit_df, valid, cols, **best)
    test = load("test", args.features)

    if args.params_from:
        tuning = {"params_from": args.params_from, "refit_max_moves": refit_moves}
    else:
        n_ok = len(table)
        if "error" in table:
            n_ok = int(table["error"].isna().sum())
        tuning = {"sampler": args.sampler, "objective": args.objective, "trials": len(table), "trials_ok": n_ok,
                  "tune_moves": args.tune_moves, "refit_max_moves": refit_moves}
    settings = {"features": args.features, **best, "params": best, "max_train_games": args.max_games,
                "tuning": tuning}
    extra = {"fit_rows": len(refit_df)}
    extra.update(m.info)
    d = save_run(args.method, name, settings, predictions(m, {"valid": valid, "test": test}), t0, model=m,
                 extra=extra)
    if m.history:
        pd.DataFrame(m.history).to_csv(d / "history.csv", index=False)
    print(f"{d}: refitted on {len(refit_df):,} moves, valid {len(valid):,}, test {len(test):,}")


if __name__ == "__main__":
    main()
