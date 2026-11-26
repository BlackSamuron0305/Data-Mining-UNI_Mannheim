"""Step 3: the tables the models read (transform.py) and what each model learns from the training games
(encoding.py)."""
import importlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.common.buckets import SPEEDS, uniform

transform = importlib.import_module("src.03_transformation.transform")
folds = numeric = trees = importlib.import_module("src.03_transformation.encoding")


def moves(n_games=200, plies=10, speeds=("blitz", "bullet")):
    rng = np.random.default_rng(0)
    n = n_games * plies
    return pd.DataFrame({
        "game_id": np.repeat([f"g{i:04d}" for i in range(n_games)], plies),
        "ply": np.tile(np.arange(plies, dtype=np.int16), n_games),
        "bucket": rng.integers(0, 756, n).astype(np.int16), "group": rng.integers(0, 28, n).astype(np.int16),
        "blunder": rng.random(n) < 0.06,
        "speed": np.repeat(rng.choice(list(speeds), n_games), plies),
        "eco": np.where(np.tile(np.arange(plies), n_games) >= 6, "B", None),
        "in_check": rng.random(n) < 0.1,
        "clock": rng.random(n) * 300,                                     # float64
        "eval_wp": np.where(rng.random(n) < 0.05, np.nan, rng.random(n)).astype(np.float32),
    })


def test_transform_types_and_categories():
    out = transform.transform(moves(), "test")
    assert out["in_check"].dtype == np.float32 and set(out["in_check"].unique()) <= {0.0, 1.0}
    assert out["clock"].dtype == np.float32 and out["eval_wp"].dtype == np.float32
    assert out["blunder"].dtype == bool                                  # the target is left as it is
    assert list(out["speed"].cat.categories) == SPEEDS                   # all four, also those not in the data
    assert list(out["eco"].cat.categories) == list("ABCDE") and out["eco"].isna().sum() > 0
    assert "fold" not in out


def test_fold_holds_out_whole_games():
    out = transform.transform(moves(n_games=3000), "train")
    per_game = out.groupby("game_id", observed=True)["fold"].nunique()
    assert (per_game == 1).all()                                         # a game is held out as a whole
    held = out.drop_duplicates("game_id")["fold"].eq("valid").mean()
    assert abs(held - folds.VALID_FRAC) < 0.02
    want = np.array([uniform(g, "ie500-valid") < folds.VALID_FRAC for g in out["game_id"]])
    assert np.array_equal(out["fold"].eq("valid").to_numpy(), want)      # the same games as before step 3 existed


def test_run_part_in_batches_with_missing_categories(tmp_path, monkeypatch):
    final = tmp_path / "final"
    final.mkdir()
    df = pd.concat([moves(100, speeds=("blitz",)), moves(100, speeds=("classical",)).assign(
        game_id=lambda d: "h" + d["game_id"])], ignore_index=True)
    df.to_parquet(final / "train.parquet", row_group_size=500)           # several batches, each with one speed
    monkeypatch.setattr(transform, "PREPROCESSING_DIR", final)
    monkeypatch.setattr(transform, "TRANSFORMATION_DIR", tmp_path / "out")
    monkeypatch.setattr(transform, "BATCH", 500)
    info = transform.run_part("train")
    back = pq.read_table(tmp_path / "out" / "train.parquet").to_pandas()
    assert info["moves"] == len(df) == len(back) and info["games"] == 200
    assert set(back["speed"]) == {"blitz", "classical"} and list(back["speed"].cat.categories) == SPEEDS
    assert "fold" in back.columns


def test_numeric_prep_learns_from_the_training_games_only():
    train = transform.transform(moves(), "test")
    test = train.copy()
    test["speed"] = pd.Categorical(["rapid"] * len(test), categories=SPEEDS)   # a value training never saw
    test["clock"] = np.float32(np.nan)
    cols = ["speed", "eco", "in_check", "clock", "eval_wp"]
    prep = numeric.numeric_prep(cols).fit(train[cols])
    x_train, x_test = prep.transform(train[cols]), prep.transform(test[cols])
    assert x_train.shape[1] == x_test.shape[1]
    assert not np.isnan(x_test).any()                                    # gaps filled with the training median
    num = x_train[:, :3]                                                 # in_check, clock, eval_wp scaled
    assert np.allclose(num.mean(0), 0, atol=1e-6) and np.allclose(num.std(0), 1, atol=1e-3)


def test_quantile_scaler_is_bell_shaped():
    x = np.random.default_rng(1).exponential(size=(20_000, 1))
    z = numeric.scaler("quantile").fit_transform(x)
    assert abs(np.median(z)) < 0.05 and 0.9 < z.std() < 1.1


def test_tree_features_keep_gaps_and_number_the_categories():
    df = transform.transform(moves(), "test")
    x = trees.tree_features(df, ["speed", "eco", "clock"])
    assert set(np.unique(x["speed"])) <= {0.0, 1.0}                       # blitz, bullet: their positions in SPEEDS
    assert x["eco"].isna().sum() == df["eco"].isna().sum()               # a gap stays a gap
    assert x["clock"].dtype == np.float32
