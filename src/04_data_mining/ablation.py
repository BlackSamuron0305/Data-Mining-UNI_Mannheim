"""Feature-group ablation: one model fitted with all features, without each group, and with each group alone.

Usage:  uv run python -m src.04_data_mining.ablation [--method xgboost|gradient_boosting] [--features context|engine]
            [--max-games N]

Uses the settings of the tuned run <method>/<features>_tuned if it exists, else the defaults, and writes
data/04_data_mining/<selection>/ablation/<method>_<features>_<variant>/.
"""
import argparse
import json
import time

from src.config import RUNS_DIR
from .common import TUNING, cap_moves, columns, folds, groups_of, load, predictions, save_run
from .tune import method_module

METHOD = "ablation"


def settings_of(method, feature_set):
    """The settings to use and where they come from."""
    mod = method_module(method)
    path = RUNS_DIR / method / f"{feature_set}_tuned" / "run.json"
    if path.exists():
        params = dict(mod.DEFAULTS)
        params.update(json.loads(path.read_text(encoding="utf-8"))["params"])
        return params, str(path.parent)
    return dict(mod.DEFAULTS), "defaults"


def variants(feature_set):
    """The feature list of every variant: all, without_<group>, only_<group>."""
    full = columns(feature_set)
    out = {"all": full}
    for group, group_cols in groups_of(feature_set).items():
        if len(group_cols) < len(full):
            out[f"without_{group}"] = [c for c in full if c not in group_cols]
        out[f"only_{group}"] = group_cols
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["xgboost", "gradient_boosting"], default="xgboost")
    ap.add_argument("--features", choices=["context", "engine"], default="context")
    ap.add_argument("--max-games", type=int, default=None)
    args = ap.parse_args()

    mod = method_module(args.method)
    fit_moves = TUNING.get("ablation_max_moves")
    params, source = settings_of(args.method, args.features)
    todo = variants(args.features)
    fit_df, valid = folds(load("train", args.features, args.max_games))
    fit_df = cap_moves(fit_df, fit_moves or None, "ie500-ablation")
    test = load("test", args.features)
    print(f"{args.method} on {args.features}: {len(todo)} variants, settings from {source}, "
          f"{len(fit_df):,} fitting moves")

    for variant, cols in todo.items():
        t0 = time.time()
        m = mod.fit(fit_df, valid, cols, **params)
        name = f"{args.method}_{args.features}_{variant}"
        settings = {"model": args.method, "features": args.features, "variant": variant, "columns": cols,
                    "params": params, "params_from": source, "max_train_games": args.max_games,
                    "fit_moves": fit_moves}
        extra = {"fit_rows": len(fit_df)}
        extra.update(m.info)
        d = save_run(METHOD, name, settings, predictions(m, {"valid": valid, "test": test}), t0, model=m,
                     extra=extra)
        text = f"{d}: {len(cols)} features, {time.time() - t0:.0f} s"
        for k, v in m.info.items():
            text += f", {k} {v}"
        print(text)


if __name__ == "__main__":
    main()
