"""Step 2, reduction: how much a column tells about blunders, and the rules that remove columns."""
import importlib

import numpy as np
import pandas as pd
import pytest

inspection = importlib.import_module("src.02_preprocessing.common.inspection")
reduction = importlib.import_module("src.02_preprocessing.common.reduction")
with_reduce = importlib.import_module("src.02_preprocessing.with_stockfish.reduce")


def test_mutual_information():
    rng = np.random.default_rng(0)
    y = rng.random(50_000) < 0.1
    assert inspection.mutual_information(pd.Series(rng.random(50_000)), y) < 0.001          # no relation
    assert inspection.mutual_information(pd.Series(y.astype(float)), y) > 0.4               # the target itself
    words = pd.Series(np.where(y, "yes", "no"))
    assert inspection.mutual_information(words, y) > 0.4                                     # a text column
    curve = pd.Series(np.where(y, rng.choice([-1.0, 1.0], 50_000), 0) + rng.normal(0, .1, 50_000))
    assert abs(curve.corr(pd.Series(y, dtype=float), method="spearman")) < 0.05             # no straight line...
    assert inspection.mutual_information(curve, y) > 0.1                                    # ...but information
    gaps = pd.Series(np.where(y, np.nan, 1.0))
    assert inspection.mutual_information(gaps, y) > 0.4                                     # a gap is a value


def inspection_dir(tmp_path, info: dict, pairs: list):
    pd.DataFrame({"column": list(info), "mi_millibits": list(info.values())}).to_csv(tmp_path / "columns.csv",
                                                                                     index=False)
    pd.DataFrame(pairs, columns=["column_a", "column_b", "corr"]).to_csv(tmp_path / "pairs.csv", index=False)
    return tmp_path


def removed(table):
    return dict(zip(table["column"], table["rule"]))


def test_rules(tmp_path):
    folder = inspection_dir(tmp_path, {"strong": 20, "twin": 15, "copy": 18, "faint": 0.1, "derived": 9,
                                       "protected": 0.01, "a": 5, "b": 4},
                            [("twin", "copy", 0.95), ("a", "b", -0.92)])
    got = removed(reduction.decide(folder, {"derived": "follows from strong"}, {"protected": "needed"}))
    # the weaker of a pair goes; a weak column goes unless kept; the domain list goes whatever the numbers say
    assert got == {"derived": "domain", "twin": "redundant", "b": "redundant", "faint": "weak"}


def test_kept_columns_survive_every_rule(tmp_path):
    folder = inspection_dir(tmp_path, {"x": 10, "y": 9, "z": 0.01}, [("x", "y", -0.95)])
    got = reduction.decide(folder, {}, {"x": "reason", "y": "reason", "z": "reason"})
    assert got.empty


def test_a_column_already_removed_saves_its_partner(tmp_path):
    folder = inspection_dir(tmp_path, {"x": 10, "y": 9, "z": 8}, [("x", "y", 0.99), ("y", "z", 0.95)])
    assert removed(reduction.decide(folder, {}, {})) == {"y": "redundant"}


def test_only_candidates_can_go(tmp_path):
    # with Stockfish: a Stockfish column that repeats a kept column goes, even if it tells more
    folder = inspection_dir(tmp_path, {"eval_wp": 10, "sf_wp_best": 12, "sf_weak": 0.01, "weak_other": 0.01},
                            [("eval_wp", "sf_wp_best", 0.99)])
    got = removed(reduction.decide(folder, {}, {}, candidates=["sf_wp_best", "sf_weak"]))
    assert got == {"sf_wp_best": "redundant", "sf_weak": "weak"}


def write_clean(pre_dir, table, columns):
    folder = pre_dir / table / "clean"
    folder.mkdir(parents=True)
    for part in ("train", "test"):
        pd.DataFrame({c: [1.0, 2.0] for c in columns}).to_parquet(folder / f"{part}.parquet")


def test_reduce_tables_writes_the_final_tables(tmp_path, monkeypatch):
    monkeypatch.setattr(reduction, "PRE_DIR", tmp_path)
    write_clean(tmp_path, "without_stockfish", ["game_id", "a", "b"])
    decisions = pd.DataFrame([{"column": "b", "rule": "weak", "basis": "inspection", "reason": "r"}])
    reduction.reduce_tables("without_stockfish", lambda: (decisions, {"a": "kept"}))
    final = tmp_path / "without_stockfish" / "final"
    assert list(pd.read_parquet(final / "train.parquet").columns) == ["game_id", "a"]
    assert list(pd.read_csv(final / "removed.csv")["column"]) == ["b"]
    assert list(pd.read_csv(final / "kept.csv")["column"]) == ["a"]


def test_reduction_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setattr(reduction, "PRE_DIR", tmp_path)
    monkeypatch.setitem(reduction.RULES, "apply", False)
    write_clean(tmp_path, "without_stockfish", ["game_id", "a", "b"])
    reduction.reduce_tables("without_stockfish", lambda: pytest.fail("no decisions without a reduction"))
    final = tmp_path / "without_stockfish" / "final"
    assert list(pd.read_parquet(final / "test.parquet").columns) == ["game_id", "a", "b"]
    assert pd.read_csv(final / "removed.csv").empty


def test_the_balanced_run_takes_the_decisions_of_the_stratified_run(tmp_path, monkeypatch):
    monkeypatch.setattr(reduction, "PRE_DIR", tmp_path / "stratified_balanced")
    monkeypatch.setattr(reduction, "PREPROCESSING_ROOT", tmp_path)
    monkeypatch.setattr(reduction, "SELECTION", "stratified_balanced")
    write_clean(tmp_path / "stratified_balanced", "without_stockfish", ["game_id", "a", "b"])
    stratified = tmp_path / "stratified" / "without_stockfish" / "final"
    stratified.mkdir(parents=True)
    pd.DataFrame([{"column": "a", "rule": "weak", "basis": "inspection", "reason": "r"}]) \
        .to_csv(stratified / "removed.csv", index=False)
    reduction.reduce_tables("without_stockfish", lambda: pytest.fail("the balanced run decides nothing itself"))
    final = tmp_path / "stratified_balanced" / "without_stockfish" / "final"
    assert list(pd.read_parquet(final / "train.parquet").columns) == ["game_id", "b"]


def test_with_stockfish_repeats_the_table_without(tmp_path, monkeypatch):
    without, with_ = tmp_path / "without_stockfish", tmp_path / "with_stockfish"
    (without / "final").mkdir(parents=True)
    pd.DataFrame([{"column": "material_own", "rule": "domain", "basis": "domain knowledge", "reason": "follows"},
                  {"column": "hour", "rule": "weak", "basis": "inspection", "reason": "tells nothing"}]) \
        .to_csv(without / "final" / "removed.csv", index=False)
    (with_ / "inspection").mkdir(parents=True)
    inspection_dir(with_ / "inspection", {"eval_wp": 10, "material_own": 9, "hour": 0.01, "sf_wp_best": 12,
                                          "sf_gap_2": 8, "sf_best_move": 1},
                   [("eval_wp", "sf_wp_best", 0.99), ("material_own", "sf_gap_2", 0.95)])
    monkeypatch.setattr(with_reduce, "WITHOUT_DIR", without)
    monkeypatch.setattr(with_reduce, "WITH_DIR", with_)
    got, _ = with_reduce.decisions()
    rules = removed(got)
    assert rules == {"material_own": "domain", "hour": "weak", "sf_best_move": "domain", "sf_wp_best": "redundant"}
    assert "sf_gap_2" not in rules                       # its partner material_own goes anyway


def test_inspection_samples_whole_games(tmp_path, monkeypatch):
    df = pd.DataFrame({"game_id": np.repeat([f"g{i}" for i in range(10)], 4), "ply": np.tile(range(4), 10)})
    df.to_parquet(tmp_path / "train.parquet")
    monkeypatch.setattr(inspection, "GAMES", 3)
    got = inspection.sample(tmp_path / "train.parquet")
    assert got["game_id"].nunique() == 3 and len(got) == 12


def test_reasons_carry_the_evidence_and_the_notes(tmp_path):
    folder = inspection_dir(tmp_path, {"material_own": 9.8, "material_total": 9.6, "hour": 0.01, "strong": 30},
                            [("material_own", "material_total", 0.99)])
    names = ["material_own", "material_total", "hour", "strong"]
    corr = pd.DataFrame(np.eye(4), index=names, columns=names)
    corr.loc["material_own", "material_total"] = corr.loc["material_total", "material_own"] = 0.99
    corr.to_csv(folder / "correlation.csv")
    got = reduction.decide(folder, {"material_own": "follows from material_total"}, {"material_total": "kept"},
                           notes={"hour": "UTC, not the player's own time", "material_own": "not used for domain"})
    reasons = dict(zip(got["column"], got["reason"]))
    assert reasons["material_own"] == ("follows from material_total; measured: Spearman 0.99 with material_total, "
                                       "9.80 millibits about blunders")
    assert reasons["hour"].startswith("tells 0.01 millibits") and reasons["hour"].endswith("UTC, not the player's own time")
