"""Settings and paths shared by every script. The settings live in config.yaml at the root of the repository."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG: dict = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
SEED: int = CFG["seed"]
WORKERS: int = int(os.environ.get("IE500_WORKERS") or min(CFG["workers"], os.cpu_count() or 1))
MONTH: str = CFG["source"]["month"]

# data/ holds the selected games, cache/ the working files of the scripts; neither is tracked by git
# (data/README.md). The environment variables IE500_DATA and IE500_CACHE point them elsewhere, e.g. for a test run.
DATA = Path(os.environ.get("IE500_DATA") or ROOT / "data")
CACHE = Path(os.environ.get("IE500_CACHE") or ROOT / "cache")
CENSUS_DIR = CACHE / "census"
STRATIFIED_DIR = DATA / "stratified_selection"
# subfolders of a selection in data/: the games (PGN and one row per game), the tables of counts and chances,
# the records of the runs (settings, statistics, logs); summary.md stays at the top (data/README.md)
GAMES_SUB, TABLES_SUB, LOGS_SUB = "games", "tables", "logs"

S3_NAMES = ("LICHESS_S3_ENDPOINT", "LICHESS_S3_BUCKET", "LICHESS_S3_ACCESS_KEY", "LICHESS_S3_SECRET_KEY")


def s3_settings() -> dict[str, str]:
    """Endpoint, bucket and keys of the storage with the raw Lichess files: the environment variables
    LICHESS_S3_*, else the file .env at the root of the repository."""
    out = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and not key.strip().startswith("#"):
                out[key.strip()] = value.strip()
    out.update({k: os.environ[k] for k in S3_NAMES if os.environ.get(k)})
    missing = [k for k in S3_NAMES if not out.get(k)]
    if missing:
        raise RuntimeError(f"{', '.join(missing)} not set: copy .env.example to .env and fill in the keys")
    return {k: out[k] for k in S3_NAMES}


def setup_log(path: Path) -> None:
    """Log to the screen and to `path` (overwritten)."""
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(message)s", "%H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(path, mode="w", encoding="utf-8")]
    for h in handlers:
        h.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    for name in ("boto3", "botocore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
