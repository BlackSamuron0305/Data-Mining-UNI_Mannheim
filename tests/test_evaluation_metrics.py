"""Unit and end-to-end tests for evaluation metrics, operating points, figures, and LaTeX tables."""
from __future__ import annotations

import importlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.common.metrics import at_cutoff, best_cutoff, operating_points

common = importlib.import_module("src.04_data_mining.common")
dt_train = importlib.import_module("src.04_data_mining.decision_tree.train")
Fitted = common.Fitted
ShrunkTree = dt_train.ShrunkTree
groups = importlib.import_module("src.05_evaluation.groups")
validate = importlib.import_module("src.05_evaluation.validate")


def test_at_cutoff_accuracy_and_f05_and_paradox():
    # tp=2, fp=1, fn=1, tn=6 -> acc=0.8, prec=2/3, rec=2/3, f1=2/3, f05=2/3
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.2, 0.7, 0.3, 0.2, 0.1, 0.1, 0.05, 0.01])
    res = at_cutoff(y, p, 0.5)
    assert np.isclose(res["accuracy"], 0.8)
    assert np.isclose(res["precision"], 2 / 3)
    assert np.isclose(res["recall"], 2 / 3)
    assert np.isclose(res["f1"], 2 / 3)
    assert np.isclose(res["f05"], 2 / 3)
    assert np.isclose(res["balanced_accuracy"], (2 / 3 + 6 / 7) / 2)

    # Asymmetric check: tp=3, fp=1, fn=3, tn=13 -> prec=0.75, rec=0.50, f1=0.60, f05=15/22
    y2 = np.array([1] * 6 + [0] * 14)
    p2 = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05] + [0.75] + [0.1] * 13)
    res2 = at_cutoff(y2, p2, 0.5)
    assert np.isclose(res2["accuracy"], 16 / 20)
    assert np.isclose(res2["precision"], 0.75)
    assert np.isclose(res2["recall"], 0.50)
    assert np.isclose(res2["f1"], 0.60)
    assert np.isclose(res2["f05"], 15 / 22)

    # Accuracy Paradox: 5% positives, predict all negative (cut=1.0)
    y_imb = np.array([1] * 5 + [0] * 95)
    p_imb = np.full(100, 0.05)
    paradox = at_cutoff(y_imb, p_imb, 1.0)
    assert np.isclose(paradox["accuracy"], 0.95)
    assert paradox["precision"] == 0.0
    assert paradox["recall"] == 0.0
    assert paradox["f1"] == 0.0
    assert paradox["f05"] == 0.0
    assert paradox["mcc"] == 0.0
    assert np.isclose(paradox["balanced_accuracy"], 0.5)


def test_best_cutoff_objectives_and_validation():
    # High precision tail at >= 0.8 (3 TP, 0 FP -> prec=1.0) and noisier middle (4 TP, 4 FP at >= 0.4)
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.95, 0.90, 0.85, 0.45, 0.44, 0.43, 0.42, 0.41, 0.10, 0.08, 0.05, 0.02])

    cut_f1 = best_cutoff(y, p, objective="f1")
    cut_f05 = best_cutoff(y, p, objective="f0.5")
    cut_f05_alias = best_cutoff(y, p, objective="f05")
    cut_prec = best_cutoff(y, p, objective="precision_target", target_precision=0.75)

    assert np.isclose(cut_f05, cut_f05_alias)
    assert cut_f05 >= cut_f1
    m_prec = at_cutoff(y, p, cut_prec)
    assert m_prec["precision"] >= 0.75

    with pytest.raises(ValueError):
        best_cutoff(y, p, objective="unknown")
    with pytest.raises(ValueError):
        best_cutoff(y, p, objective="precision_target")
    with pytest.raises(ValueError):
        best_cutoff(y, p, objective="precision_target", target_precision=1.5)


def test_operating_points_schema_and_no_leakage():
    rng = np.random.default_rng(7)
    y_valid = (rng.random(400) < 0.06).astype(float)
    p_valid = np.clip(0.02 + 0.45 * y_valid + 0.15 * rng.random(400), 0.001, 0.95)

    y_test = (rng.random(500) < 0.06).astype(float)
    p_test = np.clip(0.02 + 0.45 * y_test + 0.15 * rng.random(500), 0.001, 0.95)

    op = operating_points(y_valid, p_valid, y_test, p_test)
    expected_cols = [
        "policy", "cutoff", "flagged", "accuracy", "balanced_accuracy",
        "precision", "recall", "f1", "f05", "mcc", "lift",
    ]
    assert list(op.columns) == expected_cols
    expected_policies = [
        "majority_never_blunder",
        "valid_max_f1",
        "valid_max_f05",
        "valid_prec_35",
        "valid_prec_50",
        "top_1pct",
        "top_2pct",
        "top_3pct",
        "top_5pct",
        "top_10pct",
    ]
    assert list(op["policy"]) == expected_policies
    assert np.isclose(op.loc[op["policy"] == "valid_max_f1", "cutoff"].iloc[0], best_cutoff(y_valid, p_valid, "f1"))
    assert np.isnan(op.loc[op["policy"] == "majority_never_blunder", "lift"].iloc[0])
    non_maj = op[op["policy"] != "majority_never_blunder"]
    assert non_maj.drop(columns=["policy"]).notna().all().all()


def _make_run(run_dir, keys_valid, y_valid, p_valid, keys_test, y_test, p_test, method, name):
    d = run_dir / method / name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "game_id": keys_valid["game_id"],
        "ply": keys_valid["ply"],
        "blunder": y_valid,
        "p": p_valid,
    }).to_parquet(d / "predictions_valid.parquet", index=False)
    pd.DataFrame({
        "game_id": keys_test["game_id"],
        "ply": keys_test["ply"],
        "blunder": y_test,
        "p": p_test,
    }).to_parquet(d / "predictions_test.parquet", index=False)
    (d / "run.json").write_text(json.dumps({"method": method, "name": name}), encoding="utf-8")
    return d


def test_validate_end_to_end_smoke(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)

    sel_root = tmp_path / "01_selection"
    runs_root = tmp_path / "04_data_mining"
    eval_dir = tmp_path / "05_evaluation" / "stratified"
    trans_dir = tmp_path / "03_transformation" / "stratified"
    trans_dir.mkdir(parents=True, exist_ok=True)

    # 30 valid games + 60 test games, 20 moves each (1,800 moves total; 1,200 test moves)
    n_valid_games, n_test_games, moves_per_game = 30, 60, 20
    valid_gids = [f"vg{i:03d}" for i in range(n_valid_games)]
    test_gids = [f"tg{i:03d}" for i in range(n_test_games)]

    for sel in ("stratified", "stratified_balanced"):
        gdir = sel_root / sel
        gdir.mkdir(parents=True, exist_ok=True)
        gdf = pd.DataFrame({
            "game_id": valid_gids + test_gids,
            "split": ["train"] * n_valid_games + ["test"] * n_test_games,
        })
        gdf.to_csv(gdir / "games.csv", index=False)

    def _build_moves(gids):
        rows = []
        for gi, gid in enumerate(gids):
            spd = "blitz" if gi % 2 == 0 else "rapid"
            elo = 1000 + (gi % 6) * 200
            for ply in range(1, moves_per_game + 1):
                c_rel = max(0.05, 1.0 - 0.04 * ply)
                wp = float(np.clip(0.2 + 0.6 * rng.random(), 0.05, 0.95))
                opp_loss = float(0.25 if rng.random() < 0.12 else 0.02)
                risk = 0.02 + 0.25 * (opp_loss > 0.15) + 0.12 * (c_rel < 0.25)
                y = int(rng.random() < risk)
                rows.append({
                    "game_id": gid,
                    "ply": ply,
                    "blunder": y,
                    "bucket": 0,
                    "group": f"{spd}_{elo}",
                    "speed": spd,
                    "mover_elo": float(elo),
                    "elo_diff": float(rng.normal(0, 60)),
                    "move_no": float((ply + 1) // 2),
                    "clock": float(180.0 * c_rel),
                    "clock_rel": float(c_rel),
                    "eval_wp": wp,
                    "own_last_loss": float(rng.exponential(0.03)),
                    "own_loss_recent": float(rng.exponential(0.02)),
                    "own_mistakes": float(rng.integers(0, 3)),
                    "opp_last_loss": opp_loss,
                    "legal_moves": float(rng.integers(10, 45)),
                    "in_check": float(rng.random() < 0.08),
                    "material_total": float(rng.integers(20, 78)),
                })
        df = pd.DataFrame(rows)
        df["speed"] = pd.Categorical(df["speed"], categories=["bullet", "blitz", "rapid", "classical"])
        return df

    valid_df = _build_moves(valid_gids)
    test_df = _build_moves(test_gids)
    valid_df.to_parquet(trans_dir / "train.parquet", index=False)
    test_df.to_parquet(trans_dir / "test.parquet", index=False)

    yv = valid_df["blunder"].to_numpy(dtype=float)
    yt = test_df["blunder"].to_numpy(dtype=float)
    base_rate = float(yv.mean())

    def _scores(df, y, quality):
        sig = 0.04 + 0.28 * (df["opp_last_loss"].to_numpy() > 0.15) + 0.15 * y
        noise = rng.normal(0, 0.05 / quality, len(y))
        return np.clip(quality * sig + (1 - quality) * base_rate + noise, 0.002, 0.92)

    runs_s = runs_root / "stratified"
    runs_b = runs_root / "stratified_balanced"
    pv_base = np.full(len(yv), base_rate)
    pt_base = np.full(len(yt), base_rate)
    _make_run(runs_s, valid_df, yv, pv_base, test_df, yt, pt_base, "bucket_rate", "month")
    _make_run(runs_b, valid_df, yv, pv_base, test_df, yt, pt_base, "bucket_rate", "month")

    for method, name, q in (
        ("naive_bayes", "context", 0.55),
        ("gradient_boosting", "context", 0.90),
        ("decision_tree", "engine_depth4", 0.72),
    ):
        p_v = _scores(valid_df, yv, q)
        p_t = _scores(test_df, yt, q)
        d = _make_run(runs_s, valid_df, yv, p_v, test_df, yt, p_t, method, name)
        if method == "decision_tree":
            cols = ["opp_last_loss", "clock_rel", "eval_wp", "mover_elo"]
            clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
            clf.fit(valid_df[cols], yv)
            fitted = Fitted(ShrunkTree(clf, base_rate), cols, inputs="tree")
            joblib.dump(fitted, d / "model.joblib")
        if method in ("naive_bayes", "gradient_boosting"):
            _make_run(runs_b, valid_df, yv, p_v, test_df, yt, np.clip(p_t * 0.99, 0.002, 0.92), method, name)

    for abl_name, q in (
        ("xgboost_engine_all", 0.91),
        ("xgboost_engine_without_engine", 0.75),
        ("xgboost_engine_only_engine", 0.65),
    ):
        _make_run(
            runs_s, valid_df, yv, _scores(valid_df, yv, q), test_df, yt, _scores(test_df, yt, q), "ablation", abl_name
        )

    monkeypatch.setattr(validate, "SELECTION", "stratified")
    monkeypatch.setattr(validate, "SELECTION_ROOT", sel_root)
    monkeypatch.setattr(validate, "SELECTION_DIR", sel_root / "stratified")
    monkeypatch.setattr(validate, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(validate, "RUNS_DIR", runs_s)
    monkeypatch.setattr(validate, "VALIDATION_DIR", eval_dir)
    monkeypatch.setitem(validate.EV, "min_moves", 20)
    monkeypatch.setattr(groups, "TRANSFORMATION_DIR", trans_dir)

    validate.run(n_boot=20)
    assert (eval_dir / "results.csv").exists()
    assert (eval_dir / "operating_points.csv").exists()
    assert (eval_dir / "intervals.csv").exists()
    assert (eval_dir / "summary.md").exists()

    validate.run(others=["stratified_balanced"], n_boot=20)
    with_dir = eval_dir / "with_stratified_balanced"
    assert (with_dir / "results.csv").exists()
    assert (with_dir / "operating_points.csv").exists()
    assert (with_dir / "intervals.csv").exists()

    expected_figs = [
        "roc_pr", "calibration", "gains", "subgroups_gain", "subgroups_pr_auc",
        "operating_points_metrics", "stratified_vs_balanced", "tree_depth4", "error_profile",
    ]
    for fig_name in expected_figs:
        for ext in ("pdf", "png"):
            fp = with_dir / "figures" / f"{fig_name}.{ext}"
            assert fp.exists(), f"missing figure {fp}"
            assert fp.stat().st_size > 0

    expected_tables = [
        "tab_results.tex", "tab_results_slide.tex", "tab_metric_justification.tex",
        "tab_operating_points.tex", "tab_ablation.tex",
    ]
    for tab_name in expected_tables:
        tp = with_dir / "tables" / tab_name
        assert tp.exists(), f"missing table {tp}"
        assert "\\begin{tabular}" in tp.read_text(encoding="utf-8")
