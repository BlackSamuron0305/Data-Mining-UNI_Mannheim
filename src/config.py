"""Settings (from config.yaml) and paths shared by every script."""
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

# IE500_DATA and IE500_CACHE point data/ and cache/ elsewhere; IE500_SELECTION picks the selection that
# the later steps use
DATA = Path(os.environ.get("IE500_DATA") or ROOT / "data")
CACHE = Path(os.environ.get("IE500_CACHE") or ROOT / "cache")
CLEANUP_DIR = CACHE / "cleanup"
BLUNDER_RATES_DIR = DATA / "blunder_rates"
SELECTIONS = ["random", "stratified", "balanced"]
SELECTION = os.environ.get("IE500_SELECTION") or "stratified"
SELECTION_ROOT = DATA / "selection"
SELECTION_DIR = SELECTION_ROOT / SELECTION
COMPARISON_DIR = SELECTION_ROOT / "comparison"
FEATURES_DIR = DATA / "features"
RUNS_DIR = DATA / "runs"
VALIDATION_DIR = DATA / "validation"
GAMES_SUB = "games"
TABLES_SUB = "tables"
LOGS_SUB = "logs"
KEYS = ["game_id", "ply"]   # one row per move
TARGET = "blunder"

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


def s3_settings() -> dict[str, str]:
    """Endpoint, bucket and keys of the S3 storage, from the environment or else from .env."""
    out = dotenv()
    for k in S3_NAMES:
        if os.environ.get(k):
            out[k] = os.environ[k]
    missing = [k for k in S3_NAMES if not out.get(k)]
    if missing:
        raise RuntimeError(f"{', '.join(missing)} not set: copy .env.example to .env and fill in the keys")
    settings = {}
    for k in S3_NAMES:
        settings[k] = out[k]
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
