"""Write move tables with some columns removed."""
import pandas as pd
import pyarrow.parquet as pq


def write_without(src, dst, drop) -> int:
    """Copy the table src to dst without the columns in drop. Returns the number of rows."""
    names = pq.read_schema(src).names
    keep = [c for c in names if c not in drop]
    table = pq.read_table(src, columns=keep)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst, compression="zstd")
    return table.num_rows


def decision_table(decisions: dict) -> pd.DataFrame:
    """One row per removed column: its reason and on what the decision rests."""
    rows = []
    for col, (basis, reason) in decisions.items():
        rows.append({"column": col, "basis": basis, "reason": reason})
    return pd.DataFrame(rows, columns=["column", "basis", "reason"])
