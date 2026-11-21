"""Our prepared data on the team's server (S3), so that nobody needs the 30 GB month or 15 hours of Stockfish.

Usage:  uv run python -m src.lichess.share download [cleanup] [stockfish]   (default: both)
        uv run python -m src.lichess.share upload   [cleanup] [stockfish]   (needs a key that may write)
        uv run python -m src.lichess.share list

On the server, under Data-Mining-Project/<month folder>/:
  raw/        the month as Lichess publishes it
  cleanup/    the counts of the month and the pool of games (src/01_selection/preparation/cleanup/)
  stockfish/  train.parquet, test.parquet, engine.json: Stockfish for every move of the selected games
"""
import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError

from src.config import CACHE, DATA, MONTH, ROOT, s3_settings

PROJECT = "Data-Mining-Project"
MONTH_FOLDERS = {"2026-09": "September 2026", "2026-08": "August 2026"}
PACKAGES = ["cleanup", "stockfish"]
PARTS = ["train", "test"]
KEYS = ["game_id", "ply"]
CLEANUP_DIR = CACHE / "cleanup"
STOCKFISH_DIR = DATA / "preprocessing" / "with_stockfish" / "stockfish"
OLD_ENGINE_DIRS = [ROOT / "data" / "features" / "engine", ROOT / "data_balanced" / "features" / "engine"]
TRANSFER = TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20, max_concurrency=16)


def client():
    s = s3_settings()
    s3 = boto3.client("s3", endpoint_url=s["LICHESS_S3_ENDPOINT"], aws_access_key_id=s["LICHESS_S3_ACCESS_KEY"],
                      aws_secret_access_key=s["LICHESS_S3_SECRET_KEY"], region_name="us-east-1",
                      config=Config(s3={"addressing_style": "path"}, request_checksum_calculation="when_required",
                                    response_checksum_validation="when_required", retries={"max_attempts": 10}))
    return s3, s["LICHESS_S3_BUCKET"]


def folder(month: str = MONTH) -> str:
    return f"{PROJECT}/{MONTH_FOLDERS[month]}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def manifest(s3, bucket: str) -> dict:
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=f"{folder()}/manifest.json")["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return {"files": {}}
        raise


def local_path(name: str) -> Path | None:
    """Where a file of the server belongs on this computer, or None if the name is not one we expect."""
    package, _, rest = name.partition("/")
    if not rest or Path(rest).name != rest or Path(rest).is_absolute():
        return None
    if package == "cleanup":
        base, dst = CLEANUP_DIR, CLEANUP_DIR / rest
    elif package == "stockfish" and rest.removesuffix(".parquet") in PARTS:
        base, dst = STOCKFISH_DIR, STOCKFISH_DIR / rest.removesuffix(".parquet") / "stockfish.parquet"
    elif package == "stockfish" and rest == "engine.json":
        base, dst = STOCKFISH_DIR, STOCKFISH_DIR / rest
    else:
        return None
    if not dst.resolve().is_relative_to(base.resolve()):
        return None
    return dst


def merged_stockfish(part: str, tmp: Path) -> Path:
    """One file with the Stockfish rows of a part, each move once, from the engine runs on this computer."""
    tables = []
    for base in OLD_ENGINE_DIRS + [STOCKFISH_DIR]:
        for f in sorted((base / part).glob("*.parquet")):
            tables.append(pq.read_table(f))
    if not tables:
        sys.exit(f"no Stockfish files of the {part} games on this computer")
    df = pa.concat_tables(tables, promote_options="default").to_pandas()
    df = df.drop_duplicates(KEYS).sort_values(KEYS, kind="stable").reset_index(drop=True)
    out = tmp / f"{part}.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out, compression="zstd", row_group_size=1_000_000)
    return out


def upload(packages: list[str], tmp: Path) -> None:
    s3, bucket = client()
    man = manifest(s3, bucket)
    files = {}
    if "cleanup" in packages:
        for path in sorted(CLEANUP_DIR.glob("*")):
            if path.is_file():
                files[f"cleanup/{path.name}"] = path
    if "stockfish" in packages:
        for part in PARTS:
            print(f"stockfish {part}: merging the engine files")
            files[f"stockfish/{part}.parquet"] = merged_stockfish(part, tmp)
        for base in OLD_ENGINE_DIRS + [STOCKFISH_DIR]:
            meta = base / "train" / "engine.json"
            if meta.exists():
                files["stockfish/engine.json"] = meta
                break
    total = sum(p.stat().st_size for p in files.values())
    print(f"uploading {len(files)} files, {total / 1e9:.2f} GB, to {bucket}/{folder()}/")
    for name, path in files.items():
        s3.upload_file(str(path), bucket, f"{folder()}/{name}", Config=TRANSFER)
        man["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        print(f"  {name}: {path.stat().st_size / 1e6:.1f} MB")
    man["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    body = json.dumps(man, indent=2).encode()
    s3.put_object(Bucket=bucket, Key=f"{folder()}/manifest.json", Body=body, ContentLength=len(body))
    print("done, manifest.json updated")


def download(packages: list[str]) -> None:
    s3, bucket = client()
    man = manifest(s3, bucket)
    names = [n for n in man["files"] if n.split("/")[0] in packages]
    if not names:
        sys.exit(f"nothing of {', '.join(packages)} under {bucket}/{folder()}/ yet")
    for name in names:
        dst = local_path(name)
        if dst is None:
            sys.exit(f"unexpected file name in manifest.json: {name}")
        info = man["files"][name]
        if dst.exists() and dst.stat().st_size == info["bytes"] and sha256(dst) == info["sha256"]:
            print(f"  {name}: already there")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, f"{folder()}/{name}", str(dst), Config=TRANSFER)
        if dst.stat().st_size != info["bytes"] or sha256(dst) != info["sha256"]:
            sys.exit(f"{name}: the download is incomplete or damaged; try again")
        print(f"  {name}: {info['bytes'] / 1e6:.1f} MB")
    print(f"{len(names)} files checked and in place")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["download", "upload", "list"])
    ap.add_argument("packages", nargs="*", default=[], help="cleanup, stockfish (default: both)")
    args = ap.parse_args()
    packages = args.packages or PACKAGES
    unknown = [p for p in packages if p not in PACKAGES]
    if unknown:
        ap.error(f"unknown package {', '.join(unknown)}; choose from {', '.join(PACKAGES)}")
    if args.action == "list":
        s3, bucket = client()
        man = manifest(s3, bucket)
        for name, info in man["files"].items():
            print(f"{info['bytes'] / 1e6:10.1f} MB  {name}")
        print(f"{len(man['files'])} files, updated {man.get('updated', '-')}")
    elif args.action == "upload":
        with tempfile.TemporaryDirectory() as tmp:
            upload(packages, Path(tmp))
    else:
        download(packages)


if __name__ == "__main__":
    main()
