"""Remove the false predictors from the extracted moves.

Usage:  uv run python -m src.02_preprocessing.without_stockfish.preprocess   (after common.extract)
Output: data/preprocessing/<selection>/without_stockfish/clean/train.parquet, test.parquet, false_predictors.csv
"""
import logging

from src.config import setup_log
from ..common.false_predictors import FALSE_PREDICTORS
from ..common.paths import EXTRACTED_DIR, PARTS, WITHOUT_DIR
from ..common.tables import decision_table, write_without

log = logging.getLogger("without_stockfish")


def run() -> None:
    WITHOUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_log(WITHOUT_DIR / "preprocess.log")
    for part in PARTS:
        src = EXTRACTED_DIR / f"{part}.parquet"
        if not src.exists():
            raise FileNotFoundError(f"{src} is missing: run uv run python -m src.02_preprocessing.common.extract first")
        n = write_without(src, WITHOUT_DIR / "clean" / f"{part}.parquet", FALSE_PREDICTORS)
        log.info("%s: %s moves, %d false predictors removed", part, f"{n:,}", len(FALSE_PREDICTORS))
    decision_table(FALSE_PREDICTORS).to_csv(WITHOUT_DIR / "false_predictors.csv", index=False)


if __name__ == "__main__":
    run()
