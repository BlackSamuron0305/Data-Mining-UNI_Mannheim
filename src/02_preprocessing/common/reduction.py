"""Write the final move tables without the columns chosen in reduce.py."""
from .paths import PARTS
from .tables import decision_table, write_without


def reduce_tables(clean_dir, final_dir, remove: dict) -> None:
    for part in PARTS:
        n = write_without(clean_dir / f"{part}.parquet", final_dir / f"{part}.parquet", remove)
        print(f"{part}: {n:,} moves, {len(remove)} columns removed")
    decision_table(remove).to_csv(final_dir / "removed.csv", index=False)
