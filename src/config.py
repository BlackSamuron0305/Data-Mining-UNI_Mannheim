"""Settings and paths shared by every script: the fixed settings of the method (src/settings.py) and what can be
chosen in config.yaml (where the data comes from, the number of processes)."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import yaml

from src.settings import SETTINGS

ROOT = Path(__file__).resolve().parents[1]
CHOICES = ("data", "workers")   # the only settings read from config.yaml
_yaml = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8")) or {}
CFG: dict = {**SETTINGS, **{k: _yaml[k] for k in CHOICES}}
SEED: int = CFG["seed"]
WORKERS: int = int(os.environ.get("IE500_WORKERS") or min(CFG["workers"], os.cpu_count() or 1))
# IE500_MONTH picks another month for the download, the cleanup and the blunder rates, e.g. 2026-08
MONTH: str = os.environ.get("IE500_MONTH") or CFG["month"]
MONTH_NAMES = {"2026-08": "august", "2026-09": "september"}

# data/ has one folder per step of src/; IE500_SELECTION picks the selection that the steps after the
# selection use, and each of them has a folder per selection
DATA = ROOT / "data"
RAW_ROOT = DATA / "01_selection" / "raw"
CLEANUP_DIR = DATA / "01_selection" / "preparation" / "cleanup" / MONTH_NAMES[MONTH]
BLUNDER_RATES_DIR = DATA / "01_selection" / "preparation" / "blunder_rates" / MONTH_NAMES[MONTH]
SELECTIONS = ["random", "stratified", "stratified_balanced"]
SELECTION = os.environ.get("IE500_SELECTION") or "stratified"
SELECTION_ROOT = DATA / "01_selection" / "sampling"
SELECTION_DIR = SELECTION_ROOT / SELECTION
COMPARISON_DIR = SELECTION_ROOT / "comparison"
PREPROCESSING_ROOT = DATA / "02_preprocessing"
PREPROCESSING_DIR = PREPROCESSING_ROOT / SELECTION
FINAL_DIRS = {"without_stockfish": PREPROCESSING_DIR / "without_stockfish" / "final",
              "with_stockfish": PREPROCESSING_DIR / "with_stockfish" / "final"}
TRANSFORMATION_ROOT = DATA / "03_transformation"
TRANSFORMATION_DIR = TRANSFORMATION_ROOT / SELECTION
TRANSFORMED_DIRS = {"without_stockfish": TRANSFORMATION_DIR / "without_stockfish",
                    "with_stockfish": TRANSFORMATION_DIR / "with_stockfish"}
RUNS_ROOT = DATA / "04_data_mining"
RUNS_DIR = RUNS_ROOT / SELECTION
VALIDATION_DIR = DATA / "05_evaluation" / SELECTION
GAMES_SUB = "games"
TABLES_SUB = "tables"
LOGS_SUB = "logs"
KEYS = ["game_id", "ply"]   # one row per move
TARGET = "blunder"
# the columns of a move table that are not features: its keys, its bucket and group, and the target
BOOKKEEPING = KEYS + ["bucket", "group", TARGET]

S3_NAMES = ("LICHESS_S3_ENDPOINT", "LICHESS_S3_BUCKET", "LICHESS_S3_ACCESS_KEY", "LICHESS_S3_SECRET_KEY")


def dotenv() -> dict[str, str]:
    """The settings in the file .env at the root of the repository (empty if there is none)."""
    out = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#"):
                out[key.strip()] = value.strip()
    return out


def env(name: str) -> str | None:
    """An environment variable, else its value in .env, else None."""
    return os.environ.get(name) or dotenv().get(name) or None


# Where the data comes from (data: in config.yaml): the state downloaded from our server, the later steps computed
STATES = ["raw", "selected", "preprocessed", "transformed"]
START_WITH: str = CFG["data"]["start_with"]
ALSO_DOWNLOAD: list[str] = list(CFG["data"].get("also_download") or [])
RAW_FROM: str = CFG["data"]["raw_from"]
STOCKFISH: str = CFG["data"]["stockfish"]
if START_WITH not in STATES:
    raise ValueError(f"data.start_with must be one of {', '.join(STATES)}, not {START_WITH!r}")
for _state in ALSO_DOWNLOAD:
    if _state not in STATES[1:]:
        raise ValueError(f"data.also_download takes {', '.join(STATES[1:])}, not {_state!r}")
if RAW_FROM not in ("lichess", "server"):
    raise ValueError(f"data.raw_from must be lichess or server, not {RAW_FROM!r}")
if STOCKFISH not in ("download", "compute"):
    raise ValueError(f"data.stockfish must be download or compute, not {STOCKFISH!r}")
# the states that come from the server, the start first
DOWNLOADED: list[str] = list(dict.fromkeys(([START_WITH] if START_WITH != "raw" else []) + ALSO_DOWNLOAD))


def s3_settings() -> dict[str, str]:
    """Endpoint, bucket and keys of our server, from the environment or else from .env. Without keys the requests
    go unsigned, which works for a public bucket."""
    out = dotenv()
    for k in S3_NAMES:
        if os.environ.get(k):
            out[k] = os.environ[k]
    missing = [k for k in S3_NAMES[:2] if not out.get(k)]
    if missing:
        raise RuntimeError(f"{', '.join(missing)} not set: copy .env.example to .env")
    settings = {}
    for k in S3_NAMES:
        settings[k] = out.get(k) or ""
    return settings


def setup_log(path: Path) -> None:
    """Log to the screen and to `path` (overwritten)."""
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(message)s", "%H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(path, mode="w", encoding="utf-8")]
    for h in handlers:
        h.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    for name in ("boto3", "botocore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
