"""Code shared by all training methods: feature sets, loading the moves, the fitted model and saving a run.

Every run writes data/04_data_mining/<selection>/<method>/<name>/ with run.json, predictions_valid.parquet, predictions_test.parquet
and, for methods with a model, model.joblib.
"""
import importlib
import json
import time
from datetime import datetime, timezone
from functools import cache

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import BOOKKEEPING, CFG, KEYS, RUNS_DIR, STOCKFISH_PREFIX, TARGET, TRANSFORMATION_DIR
from src.common.buckets import uniform
from src.common.server import ensure

encoding = importlib.import_module("src.03_transformation.encoding")
CATEGORICAL = encoding.CATEGORICAL
scaler = encoding.scaler

TUNING = CFG["tuning"]
PREDICT_BATCH = 2_000_000    # moves per prediction call, to keep the memory down

# basic: three columns; context: every column of the table but the Stockfish columns; engine: every column
FEATURE_SETS = ["basic", "context", "engine"]
BASIC = ["speed", "mover_elo", "clock"]
NOT_FEATURES = BOOKKEEPING + ["fold"]   # fold: the held-out training games (step 3)

# Feature groups for the ablation and the importance; "game" gets every column not named elsewhere.
GROUPS = {
    "game": [],
    "rating": ["mover_elo", "opp_elo", "elo_diff"],
    "clock": ["clock", "opp_clock", "clock_rel", "clock_diff", "own_last_spent", "own_spent_recent", "opp_last_spent"],
    "eval": ["eval_cp", "eval_wp"],
    "history": ["own_last_loss", "own_loss_mean", "own_loss_recent", "own_loss_max", "own_inaccuracies",
                "own_mistakes", "own_blunders", "own_moves_since_error", "opp_last_loss", "opp_loss_recent",
                "opp_mistakes", "opp_blunders"],
    "board": ["legal_moves", "in_check", "material_own", "material_opp", "material_diff", "material_total"],
    "engine": [],
}


def table_path(part):
    """A table of step 3 (data/03_transformation/<selection>/<part>.parquet)."""
    return ensure(TRANSFORMATION_DIR / f"{part}.parquet", "step 3 (python -m src.03_transformation.transform)")


@cache
def table_columns():
    """The feature columns of the table, in its order."""
    names = pq.read_schema(table_path("train")).names
    return [c for c in names if c not in NOT_FEATURES]


def columns(cols):
    """The feature columns of a feature set (given by name) or of a list of columns."""
    if cols == "basic":
        return list(BASIC)
    if cols == "context":
        return [c for c in table_columns() if not c.startswith(STOCKFISH_PREFIX)]
    if cols == "engine":
        return table_columns()
    if isinstance(cols, str):
        raise ValueError(f"unknown feature set {cols!r}, choose from {FEATURE_SETS}")
    return list(cols)


def group_of(col):
    if col.startswith(STOCKFISH_PREFIX):
        return "engine"
    for group, group_cols in GROUPS.items():
        if col in group_cols:
            return group
    return "game"


def groups_of(cols):
    """The feature groups of the given columns (empty groups left out)."""
    out = {}
    for group in GROUPS:
        kept = [c for c in columns(cols) if group_of(c) == group]
        if kept:
            out[group] = kept
    return out


def is_valid_game(game_ids):
    """True for the training games held out for validation (the same for every run, src/03_transformation/)."""
    return encoding.is_valid_game(game_ids)


def load(part, cols, max_games=None):
    """The moves of 'train' or 'test' with keys, bucket, target and features; train moves also have `fold`."""
    path = table_path(part)
    wanted = columns(cols)
    read = KEYS + ["bucket", TARGET]
    if part == "train":
        read.append("fold")
    for c in wanted:
        if c not in read:
            read.append(c)

    games = None
    if max_games is not None:
        ids = pq.read_table(path, columns=["game_id"]).column("game_id").unique().to_pylist()
        ids = sorted(ids, key=lambda g: uniform(g, f"ie500-subset-{part}"))
        games = set(ids[:max_games])

    filters = None
    if games is not None:
        filters = [("game_id", "in", sorted(games))]
    df = pq.read_table(path, columns=read, filters=filters).to_pandas()

    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    df = encoding.add_categories(df, wanted)
    if part == "train":
        df["fold"] = pd.Categorical(df["fold"], categories=encoding.FOLDS)
    return df.reset_index(drop=True)


def folds(train):
    """The fitting moves and the held-out moves of the training games."""
    is_valid = (train["fold"] == "valid").to_numpy()
    return train[~is_valid].reset_index(drop=True), train[is_valid].reset_index(drop=True)


def cap_moves(df, max_moves, salt="ie500-cap"):
    """A random subset of whole games with about `max_moves` moves (all of df if None or smaller)."""
    if max_moves is None or len(df) <= max_moves:
        return df
    sizes = df.groupby("game_id", sort=False, observed=True).size().to_dict()
    games = sorted(sizes, key=lambda g: uniform(g, salt))
    keep = []
    total = 0
    for g in games:
        keep.append(g)
        total += sizes[g]
        if total >= max_moves:
            break
    return df[df["game_id"].isin(keep)].reset_index(drop=True)


def split_games(df, frac, salt):
    """Split df by game: the rest and about `frac` of the games."""
    second = set()
    for g in df["game_id"].unique():
        if uniform(g, salt) < frac:
            second.add(g)
    in_second = df["game_id"].isin(second).to_numpy()
    return df[~in_second].reset_index(drop=True), df[in_second].reset_index(drop=True)


def features(df, cols):
    """The feature columns, booleans as numbers (src/03_transformation/encoding.py)."""
    return encoding.features(df, columns(cols))


def numeric_prep(cols, scale=None):
    """Gaps filled, numbers scaled, categories one-hot (src/03_transformation/encoding.py)."""
    return encoding.numeric_prep(columns(cols), scale)


def tree_features(df, cols):
    """Categories as numbers for the trees (src/03_transformation/encoding.py)."""
    return encoding.tree_features(df, columns(cols))


class Fitted:
    """A fitted model plus what it needs to turn moves into blunder probabilities."""

    def __init__(self, model, columns, inputs="features", info=None, history=None):
        self.model = model
        self.columns = list(columns)
        self.inputs = inputs
        self.info = info or {}
        self.history = history or []

    def x(self, df):
        if self.inputs == "tree":
            return tree_features(df, self.columns)
        return features(df, self.columns)

    def predict(self, df):
        """Blunder probability of every move, in batches to keep the memory down."""
        batch = PREDICT_BATCH
        parts = []
        for start in range(0, len(df), batch):
            p = self.model.predict_proba(self.x(df.iloc[start:start + batch]))[:, 1]
            parts.append(np.asarray(p, dtype=np.float64))
        if not parts:
            return np.empty(0)
        return np.concatenate(parts)


def predictions(model, parts):
    """Keys, target and p of every part, for save_run."""
    out = {}
    for part, df in parts.items():
        out[part] = df[KEYS + [TARGET]].assign(p=model.predict(df))
    return out


def run_dir(method, name):
    d = RUNS_DIR / method / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run(method, name, settings, predictions, t0, model=None, extra=None):
    """Write a run: run.json, the predictions per part and the model."""
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
        joblib.dump(model, d / "model.joblib", compress=3)

    meta = {"method": method, "name": name}
    meta.update(settings)
    meta["rows"] = rows
    if extra:
        meta.update(extra)
    meta["minutes"] = round((time.time() - t0) / 60, 2)
    meta["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (d / "run.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return d


def fit_and_save(method, fit_fn, feature_set, params, max_games=None, name=None):
    """The usual run: load, fit without the held-out games, predict held-out and test games, save."""
    t0 = time.time()
    train = load("train", feature_set, max_games)
    fit, valid = folds(train)
    del train
    m = fit_fn(fit, valid, columns(feature_set), **params)
    test = load("test", feature_set)

    settings = {"features": feature_set}
    settings.update(params)
    settings["params"] = params
    settings["max_train_games"] = max_games
    extra = {"fit_rows": len(fit)}
    extra.update(m.info)
    d = save_run(method, name or feature_set, settings, predictions(m, {"valid": valid, "test": test}), t0,
                 model=m, extra=extra)
    if m.history:
        pd.DataFrame(m.history).to_csv(d / "history.csv", index=False)

    text = f"{d}: fitted on {len(fit):,} moves, valid {len(valid):,}, test {len(test):,}"
    for k, v in m.info.items():
        text += f", {k} {v}"
    print(text)
    return d, m


def target(df):
    """The target as 0/1."""
    return df[TARGET].to_numpy(dtype=np.int8)
