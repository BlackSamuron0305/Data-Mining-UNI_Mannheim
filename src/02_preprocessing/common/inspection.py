"""Look at the columns of a move table to decide which ones can go: missing values, near-constant columns,
correlations between columns and with the target. Only training games are looked at."""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import SEED
from .paths import BOOKKEEPING, TARGET

SAMPLE_MOVES = 1_000_000
HIGH = 0.9


def sample(path) -> pd.DataFrame:
    """A random sample of SAMPLE_MOVES moves of the training table."""
    table = pq.read_table(path)
    rng = np.random.default_rng(SEED)
    n = min(SAMPLE_MOVES, table.num_rows)
    rows = np.sort(rng.choice(table.num_rows, n, replace=False))
    return table.take(rows).to_pandas()


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in BOOKKEEPING]


def is_number(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per feature: kind, missing share, distinct values, share of the most common value, and the
    correlation with the target."""
    y = df[TARGET].astype(float)
    rows = []
    for col in feature_columns(df):
        s = df[col]
        top = s.value_counts(normalize=True, dropna=False)
        row = {"column": col, "kind": "number" if is_number(s) else "category",
               "missing_pct": 100 * s.isna().mean(), "distinct": s.nunique(dropna=True),
               "top_value_pct": 100 * float(top.iloc[0]) if len(top) else np.nan, "corr_target": np.nan}
        if is_number(s):
            row["corr_target"] = s.astype(float).corr(y, method="spearman")
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
    meta = {"table": str(path), "moves": len(df), "games": int(df["game_id"].nunique()), "high": HIGH,
            "blunder_pct": 100 * float(df[TARGET].mean())}
    (out_dir / "inspection.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
