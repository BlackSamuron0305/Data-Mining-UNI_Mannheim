"""What every training method shares: the feature sets, loading the moves, the held-out games, saving a run.

Every method has its own folder, src/training/<method>/train.py, and every run of it writes one folder:

data/runs/<method>/<name>/
    run.json                      method, feature set, settings, rows, minutes
    predictions_valid.parquet     game_id, ply, blunder, p: the held-out training games
    predictions_test.parquet      the same for the test games
    model.joblib                  the fitted model (methods with a model)

src/validation/validate.py scores every run on the same moves. The held-out games (`training.valid_frac` of the
training games, chosen by a fixed random number of the game id, the same for every run) are for early stopping and
for choosing between runs; the test games are for the final comparison only.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import CFG, FEATURES_DIR, RUNS_DIR
from src.preprocessing.selection.census import SPEEDS, uniform

TRAIN = CFG["training"]
KEYS = ["game_id", "ply"]
TARGET = "blunder"
# The variables of a bucket (speed x Elo of the mover x time left), but as numbers instead of steps.
BASIC = ["speed", "mover_elo", "clock"]
CONTEXT = BASIC + [
    "base_time", "increment", "white", "move_no", "opp_elo", "elo_diff",
    "opp_clock", "clock_rel", "clock_diff", "own_last_spent", "own_spent_recent", "opp_last_spent",
    "eval_cp", "eval_wp",
    "own_last_loss", "own_loss_mean", "own_loss_recent", "own_loss_max", "own_inaccuracies", "own_mistakes",
    "own_blunders", "own_moves_since_error", "opp_last_loss", "opp_loss_recent", "opp_mistakes", "opp_blunders",
    "legal_moves", "in_check", "material_own", "material_opp", "material_diff", "material_total"]
ENGINE = (["sf_wp_best", "sf_n_scored", "sf_n_close"] + [f"sf_gap_{k}" for k in range(2, 6)]
          + [f"sf_worst_{k}" for k in range(1, 6)] + [f"sf_share_{m}" for m in CFG["features"]["marks"]])
FEATURE_SETS = {"basic": BASIC, "context": CONTEXT, "engine": CONTEXT + ENGINE}
CATEGORICAL = ["speed"]


def is_valid_game(game_ids) -> np.ndarray:
    """True for the training games held out for validation (the same for every run)."""
    return np.array([uniform(g, "ie500-valid") < TRAIN["valid_frac"] for g in game_ids], dtype=bool)


def _subset(ids: np.ndarray, n: int | None, salt: str) -> np.ndarray:
    if n is None or n >= len(ids):
        return ids
    return np.array(sorted(ids, key=lambda g: uniform(g, salt))[:n], dtype=object)


def load(part: str, feature_set: str, max_games: int | None = None) -> pd.DataFrame:
    """The moves of a part ('train' or 'test') with the keys, the bucket, the target and the features of the set.
    Training moves get the column `fold` ('train' or 'valid'). With the engine set, only the analysed moves."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}, choose from {list(FEATURE_SETS)}")
    path = FEATURES_DIR / "positions" / f"{part}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run uv run python -m src.preprocessing.features.positions")
    wanted = [c for c in FEATURE_SETS[feature_set] if c not in ENGINE]
    cols = list(dict.fromkeys(KEYS + ["bucket", TARGET] + wanted))
    filters = None
    if max_games is not None:
        ids = pq.read_table(path, columns=["game_id"]).column("game_id").unique().to_numpy(zero_copy_only=False)
        filters = [("game_id", "in", list(_subset(ids, max_games, f"ie500-subset-{part}")))]
    df = pq.read_table(path, columns=cols, filters=filters).to_pandas()
    if feature_set == "engine":
        files = sorted((FEATURES_DIR / "engine" / part).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no engine features for {part}: run uv run python -m src.preprocessing.features.engine")
        eng = pd.concat([pd.read_parquet(f, columns=KEYS + ENGINE) for f in files], ignore_index=True)
        df = df.merge(eng, on=KEYS, how="inner")
    df["speed"] = pd.Categorical(df["speed"], categories=SPEEDS)
    if part == "train":
        games = df["game_id"].unique()
        valid = dict(zip(games, is_valid_game(games)))
        df["fold"] = np.where(df["game_id"].map(valid).to_numpy(dtype=bool), "valid", "train")
    return df.reset_index(drop=True)


def features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """The feature columns of a set, booleans as numbers."""
    x = df[FEATURE_SETS[feature_set]].copy()
    for c in x.columns:
        if x[c].dtype == bool:
            x[c] = x[c].astype(np.float32)
    return x


def numeric_prep(feature_set: str, scaler=None):
    """Preprocessing for the methods that need numbers without gaps on one scale (logistic regression, naive
    Bayes, neural network): missing values filled with the median plus an indicator column, then `scaler`
    (default: standardised); speed one column per speed."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cat = [f for f in FEATURE_SETS[feature_set] if f in CATEGORICAL]
    num = [f for f in FEATURE_SETS[feature_set] if f not in CATEGORICAL]
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), scaler or StandardScaler()), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])


def tree_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """The features for scikit-learn's single trees and forests: speed as its rank (bullet 0 ... classical 3, the
    order of the time per game), missing values left as NaN (the trees route them themselves)."""
    x = features(df, feature_set)
    if "speed" in x:
        x["speed"] = x["speed"].cat.codes.astype(np.float32)
    return x


def run_dir(method: str, name: str) -> Path:
    d = RUNS_DIR / method / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run(method: str, name: str, settings: dict, predictions: dict[str, pd.DataFrame], t0: float,
             model=None, extra: dict | None = None) -> Path:
    """Write a run: its settings, its predictions per part ('valid', 'test'; columns game_id, ply, blunder, p) and
    the model."""
    d = run_dir(method, name)
    rows = {}
    for part, df in predictions.items():
        out = df[KEYS + [TARGET]].copy()
        out["p"] = np.asarray(df["p"], dtype=np.float64)
        if out["p"].isna().any() or not out["p"].between(0, 1).all():
            raise ValueError(f"{method}/{name}: predictions of {part} are missing or outside 0-1")
        out.to_parquet(d / f"predictions_{part}.parquet", index=False, compression="zstd")
        rows[part] = len(out)
    if model is not None:
        import joblib
        joblib.dump(model, d / "model.joblib", compress=3)
    meta = {"method": method, "name": name, **settings, "rows": rows, **(extra or {}),
            "minutes": round((time.time() - t0) / 60, 2),
            "finished": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (d / "run.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return d
