"""Code shared by all training methods: feature sets, loading the moves, the fitted model and saving a run.

Every run writes data/runs/<method>/<name>/ with run.json, predictions_valid.parquet, predictions_test.parquet
and, for methods with a model, model.joblib.
"""
import json
import time
from datetime import datetime, timezone
from functools import cache

import joblib
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.config import CFG, FEATURES_DIR, KEYS, RUNS_DIR, TARGET
from src.shared.buckets import SPEEDS, uniform

TRAIN = CFG["training"]
TUNING = CFG["tuning"]

BASIC = ["speed", "mover_elo", "clock"]
CONTEXT = BASIC + [
    "base_time", "increment", "white", "move_no", "hour", "event_type", "eco", "opp_elo", "elo_diff",
    "opp_clock", "clock_rel", "clock_diff", "own_last_spent", "own_spent_recent", "opp_last_spent",
    "eval_cp", "eval_wp",
    "own_last_loss", "own_loss_mean", "own_loss_recent", "own_loss_max", "own_inaccuracies", "own_mistakes",
    "own_blunders", "own_moves_since_error", "opp_last_loss", "opp_loss_recent", "opp_mistakes", "opp_blunders",
    "legal_moves", "in_check", "material_own", "material_opp", "material_diff", "material_total"]
ENGINE = (["sf_wp_best", "sf_n_scored", "sf_n_close"]
          + [f"sf_gap_{k}" for k in range(2, 6)]
          + [f"sf_worst_{k}" for k in range(1, 6)]
          + [f"sf_share_{m}" for m in CFG["features"]["marks"]])
FEATURE_SETS = {"basic": BASIC, "context": CONTEXT, "engine": CONTEXT + ENGINE}

CATEGORICAL = ["speed", "event_type", "eco"]
CATEGORIES = {"speed": SPEEDS, "event_type": ["lobby", "arena", "swiss"], "eco": list("ABCDE")}

# Feature groups for the ablation and the importance; "game" gets every context feature not named elsewhere.
GROUPS = {
    "game": [],
    "rating": ["mover_elo", "opp_elo", "elo_diff"],
    "clock": ["clock", "opp_clock", "clock_rel", "clock_diff", "own_last_spent", "own_spent_recent", "opp_last_spent"],
    "eval": ["eval_cp", "eval_wp"],
    "history": ["own_last_loss", "own_loss_mean", "own_loss_recent", "own_loss_max", "own_inaccuracies",
                "own_mistakes", "own_blunders", "own_moves_since_error", "opp_last_loss", "opp_loss_recent",
                "opp_mistakes", "opp_blunders"],
    "board": ["legal_moves", "in_check", "material_own", "material_opp", "material_diff", "material_total"],
    "engine": ENGINE,
}
for _col in CONTEXT:
    if not any(_col in group_cols for group_cols in GROUPS.values()):
        GROUPS["game"].append(_col)


def columns(cols):
    """The feature columns of a feature set (given by name) or of a list of columns."""
    if isinstance(cols, str):
        if cols not in FEATURE_SETS:
            raise ValueError(f"unknown feature set {cols!r}, choose from {list(FEATURE_SETS)}")
        return list(FEATURE_SETS[cols])
    return list(cols)


def groups_of(cols):
    """The feature groups restricted to the given columns (empty groups left out)."""
    have = columns(cols)
    out = {}
    for group, group_cols in GROUPS.items():
        kept = [c for c in group_cols if c in have]
        if kept:
            out[group] = kept
    return out


def is_valid_game(game_ids):
    """True for the training games held out for validation (the same for every run)."""
    result = []
    for g in game_ids:
        result.append(uniform(g, "ie500-valid") < TRAIN["valid_frac"])
    return np.array(result, dtype=bool)


@cache
def categories(col):
    """The categories of a categorical feature: CATEGORIES, else the sorted values of the training games."""
    if col in CATEGORIES:
        return list(CATEGORIES[col])
    values = pq.read_table(FEATURES_DIR / "positions" / "train.parquet", columns=[col]).column(col)
    return sorted(v for v in pc.unique(values).to_pylist() if v is not None)


def load_engine(part, cols):
    files = sorted((FEATURES_DIR / "engine" / part).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no engine features for {part}: run uv run python -m src.03_transformation.engine")
    tables = [pd.read_parquet(f, columns=KEYS + cols) for f in files]
    eng = pd.concat(tables, ignore_index=True)
    return eng.astype({c: np.float32 for c in cols})


def load(part, cols, max_games=None):
    """The moves of 'train' or 'test' with keys, bucket, target and features; train moves also get `fold`."""
    path = FEATURES_DIR / "positions" / f"{part}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run uv run python -m src.02_preprocessing.positions")
    wanted = columns(cols)
    engine_cols = [c for c in wanted if c in ENGINE]
    read = KEYS + ["bucket", TARGET]
    for c in wanted:
        if c not in ENGINE and c not in read:
            read.append(c)

    games = None
    if max_games is not None:
        ids = pq.read_table(path, columns=["game_id"]).column("game_id").unique().to_pylist()
        ids = sorted(ids, key=lambda g: uniform(g, f"ie500-subset-{part}"))
        games = set(ids[:max_games])

    eng = None
    if engine_cols:
        eng = load_engine(part, engine_cols)
        analysed = set(eng["game_id"].unique())
        if games is None:
            games = analysed
        else:
            games = games & analysed

    filters = None
    if games is not None:
        filters = [("game_id", "in", sorted(games))]
    df = pq.read_table(path, columns=read, filters=filters).to_pandas()
    if eng is not None:
        df = df.merge(eng, on=KEYS, how="inner")

    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    for c in wanted:
        if c in CATEGORICAL:
            df[c] = pd.Categorical(df[c], categories=categories(c))

    if part == "train":
        game_ids = df["game_id"].unique()
        valid = dict(zip(game_ids, is_valid_game(game_ids)))
        is_valid = df["game_id"].map(valid).to_numpy(dtype=bool)
        df["fold"] = pd.Categorical(np.where(is_valid, "valid", "train"), categories=["train", "valid"])
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
    """The feature columns, booleans as numbers, categories kept as categories."""
    x = df[columns(cols)].copy()
    for c in x.columns:
        if x[c].dtype == bool:
            x[c] = x[c].astype(np.float32)
    return x


def numeric_prep(cols, scaler=None):
    """Median fill plus missing indicators, then scaling; categorical features one-hot encoded."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import MissingIndicator, SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if scaler is None:
        scaler = StandardScaler()
    cat = [c for c in columns(cols) if c in CATEGORICAL]
    num = [c for c in columns(cols) if c not in CATEGORICAL]
    return ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), scaler), num),
        ("missing", MissingIndicator(features="missing-only", error_on_new=False), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])


def tree_features(df, cols):
    """Features for scikit-learn trees: a category becomes its rank, missing values stay NaN."""
    x = features(df, cols)
    for c in x.columns:
        if isinstance(x[c].dtype, pd.CategoricalDtype):
            codes = x[c].cat.codes.to_numpy()
            x[c] = np.where(codes < 0, np.nan, codes).astype(np.float32)
    return x


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
        batch = TRAIN.get("predict_batch") or 2_000_000
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
