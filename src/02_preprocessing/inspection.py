"""Step 2.4: look at the columns of the moves to decide which ones can go: missing values, near-constant columns,
how much every column tells about blunders, and which columns repeat each other. Only training games are looked at.

Usage:  uv run python -m src.02_preprocessing.inspection   (after preprocess.py; the stratified run only)
Output: data/02_preprocessing/<selection>/inspection/: columns.csv, correlation.csv, pairs.csv, categories.csv,
        inspection.json; look at it in report.ipynb
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import BOOKKEEPING, CFG, PREPROCESSING_DIR, PREPROCESSING_WORK, TARGET
from src.common.buckets import uniform

GAMES = CFG["reduction"]["inspection_games"]
HIGH = CFG["reduction"]["redundant"]
BINS = 20


def sample(path) -> pd.DataFrame:
    """Every move of GAMES training games, chosen at random by game (all games if there are fewer)."""
    ids = pq.read_table(path, columns=["game_id"]).column("game_id").unique().to_pylist()
    if len(ids) > GAMES:
        ids = sorted(ids, key=lambda g: uniform(g, "ie500-inspection"))[:GAMES]
    return pq.read_table(path, filters=[("game_id", "in", ids)]).to_pandas()


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in BOOKKEEPING]


def is_number(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


def mutual_information(x: pd.Series, y: np.ndarray) -> float:
    """How much the column tells about blunders, in bits: the mutual information between the column, cut into up to
    20 bins of equal size (a missing value is a bin of its own), and the target. Unlike a correlation it also sees
    relations that are not a straight line, and it works for text columns too."""
    if not is_number(x) or x.nunique() <= BINS:
        codes = pd.factorize(x.astype(str))[0]
    else:
        codes = pd.qcut(x.astype(float), BINS, labels=False, duplicates="drop").fillna(-1).to_numpy(dtype=int)
    counts = pd.crosstab(codes, y).to_numpy(dtype=float)
    joint = counts / counts.sum()
    expected = joint.sum(1, keepdims=True) * joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum(joint * np.log2(joint / expected)))


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per feature: kind, missing share, distinct values, share of the most common value, information about
    blunders (millibits) and the correlation with the target."""
    y = df[TARGET].astype(int).to_numpy()
    rows = []
    for col in feature_columns(df):
        s = df[col]
        top = s.value_counts(normalize=True, dropna=False)
        row = {"column": col, "kind": "number" if is_number(s) else "category",
               "missing_pct": 100 * s.isna().mean(), "distinct": s.nunique(dropna=True),
               "top_value_pct": 100 * float(top.iloc[0]) if len(top) else np.nan,
               "mi_millibits": 1000 * mutual_information(s, y), "corr_target": np.nan}
        if is_number(s):
            row["corr_target"] = s.astype(float).corr(pd.Series(y, index=s.index, dtype=float), method="spearman")
        rows.append(row)
    return pd.DataFrame(rows)


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between every two number columns (rank based, so also curved relations count)."""
    cols = [c for c in feature_columns(df) if is_number(df[c])]
    return df[cols].astype(float).corr(method="spearman")


def high_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    """Pairs of columns with an absolute correlation of at least HIGH."""
    rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= HIGH:
                rows.append({"column_a": a, "column_b": b, "corr": r})
    return pd.DataFrame(rows, columns=["column_a", "column_b", "corr"]).sort_values("corr", key=abs, ascending=False)


def category_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Blunder rate per value of every category column."""
    rows = []
    for col in feature_columns(df):
        if is_number(df[col]):
            continue
        rates = df.groupby(df[col].fillna("missing"))[TARGET].agg(["mean", "size"])
        for value, r in rates.iterrows():
            rows.append({"column": col, "value": value, "moves": int(r["size"]), "blunder_pct": 100 * r["mean"]})
    return pd.DataFrame(rows)


def inspect_table(path, out_dir) -> None:
    """Write columns.csv, correlation.csv, pairs.csv, categories.csv and inspection.json into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = sample(path)
    corr = correlations(df)
    column_summary(df).to_csv(out_dir / "columns.csv", index=False)
    corr.to_csv(out_dir / "correlation.csv")
    high_pairs(corr).to_csv(out_dir / "pairs.csv", index=False)
    category_rates(df).to_csv(out_dir / "categories.csv", index=False)
    meta = {"table": path.name, "moves": len(df), "games": int(df["game_id"].nunique()), "high": HIGH,
            "blunder_pct": 100 * float(df[TARGET].mean())}
    (out_dir / "inspection.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    inspect_table(PREPROCESSING_WORK / "clean_train.parquet", PREPROCESSING_DIR / "inspection")
