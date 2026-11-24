"""Our prepared data on the team's server: list, download and upload the states of the pipeline.

Usage:  uv run python -m src.common.server list
        uv run python -m src.common.server download [selected|preprocessed|transformed|stockfish ...]
        uv run python -m src.common.server upload selected|preprocessed|transformed|stockfish ...

A state holds what its step wrote below data/ (data/README.md), plus the few files of step 1 that steps 4 and 5
read. `download` without a name takes the states of data: in config.yaml. `upload` needs a key that may write.
"""
import argparse
import fnmatch
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
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

from src.config import DATA, DOWNLOADED, KEYS, MONTH, PREPROCESSING_ROOT, s3_settings

PROJECT = "Data-Mining-Project"
MONTH_FOLDERS = {"2026-09": "September 2026", "2026-08": "August 2026"}
# what steps 4 and 5 read of step 1: the blunder rates of the month (baseline) and the test games (validation)
LATER_NEEDS = ["01_selection/preparation/blunder_rates/*/rates.csv", "01_selection/sampling/*/games/games.csv"]
# every state: the files below data/ that it holds, and those it leaves out (the pool and the large intermediate
# tables can be built again from the state before)
STATES = {
    "selected": (["01_selection/preparation/*", "01_selection/sampling/*"],
                 ["*/pool.parquet", "*/pool_buckets.parquet"]),
    "preprocessed": (["02_preprocessing/*"] + LATER_NEEDS, ["*/extracted/*", "*/clean/*", "*/stockfish/*"]),
    "transformed": (["03_transformation/*"] + LATER_NEEDS, []),
}
PACKAGES = list(STATES) + ["stockfish"]
PARTS = ["train", "test"]
# Stockfish columns of the stratified run and the balanced run: one file per part on the server, each move once.
# They are downloaded into the stratified run's folder, which the balanced run reads as well
# (src/02_preprocessing/with_stockfish/preprocess.py).
ENGINE_DIRS = [PREPROCESSING_ROOT / s / "with_stockfish" / "stockfish" for s in ("stratified", "stratified_balanced")]
# small pieces, two at a time: the server's proxy cuts off any request that takes longer than about 60 seconds
UPLOAD = TransferConfig(multipart_threshold=16 << 20, multipart_chunksize=16 << 20, max_concurrency=2)
DOWNLOAD = TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20, max_concurrency=16)


def client(**options):
    """S3 client and bucket of our server; unsigned requests if .env has no keys."""
    s = s3_settings()
    config = {"s3": {"addressing_style": "path"}, "request_checksum_calculation": "when_required",
              "response_checksum_validation": "when_required", "retries": {"max_attempts": 10}, **options}
    keys = {}
    if s["LICHESS_S3_ACCESS_KEY"] and s["LICHESS_S3_SECRET_KEY"]:
        keys = {"aws_access_key_id": s["LICHESS_S3_ACCESS_KEY"], "aws_secret_access_key": s["LICHESS_S3_SECRET_KEY"]}
    else:
        config["signature_version"] = UNSIGNED
    s3 = boto3.client("s3", endpoint_url=s["LICHESS_S3_ENDPOINT"], region_name="us-east-1", config=Config(**config),
                      **keys)
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


def rel(path: Path) -> str:
    """The path below data/, e.g. 02_preprocessing/stratified/..."""
    return path.resolve().relative_to(DATA.resolve()).as_posix()


def in_state(relpath: str, state: str) -> bool:
    include, exclude = STATES[state]
    if not any(fnmatch.fnmatch(relpath, p) for p in include):
        return False
    return not any(fnmatch.fnmatch(relpath, p) for p in exclude + ["*.part"])


# Where a file from the server goes on this computer (None if we do not expect it)
def local_path(name: str) -> Path | None:
    package, _, rest = name.partition("/")
    if package == "stockfish":
        if rest.removesuffix(".parquet") in PARTS:
            return ENGINE_DIRS[0] / rest.removesuffix(".parquet") / "stockfish.parquet"
        if rest == "engine.json":
            return ENGINE_DIRS[0] / "engine.json"
        return None
    if package not in STATES or Path(rest).is_absolute() or ".." in Path(rest).parts or not in_state(rest, package):
        return None
    dst = DATA / rest
    if not dst.resolve().is_relative_to(DATA.resolve()):
        return None
    return dst


# One Stockfish file per part from the engine runs of the stratified run and the balanced run, each move once
def merged_stockfish(part: str, tmp: Path) -> Path:
    tables = []
    for base in ENGINE_DIRS:
        for f in sorted((base / part).glob("*.parquet")):
            tables.append(pq.read_table(f))
    if not tables:
        sys.exit(f"no Stockfish files of the {part} games on this computer")
    df = pa.concat_tables(tables, promote_options="default").to_pandas()
    df = df.drop_duplicates(KEYS).sort_values(KEYS, kind="stable").reset_index(drop=True)
    out = tmp / f"{part}.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out, compression="zstd", row_group_size=1_000_000)
    return out


def state_files(state: str) -> dict[str, Path]:
    files = {}
    for path in sorted(DATA.rglob("*")):
        if path.is_file() and in_state(rel(path), state):
            files[f"{state}/{rel(path)}"] = path
    return files


def upload(packages: list[str], tmp: Path) -> None:
    s3, bucket = client()
    man = manifest(s3, bucket)
    files = {}
    for package in packages:
        if package == "stockfish":
            for part in PARTS:
                print(f"stockfish {part}: merging the engine files")
                files[f"stockfish/{part}.parquet"] = merged_stockfish(part, tmp)
            for base in ENGINE_DIRS:
                for meta in (base / "engine.json", base / "train" / "engine.json"):
                    if meta.exists() and "stockfish/engine.json" not in files:
                        files["stockfish/engine.json"] = meta
            continue
        found = state_files(package)
        if not found:
            sys.exit(f"nothing of the {package} state on this computer: compute it first (src/pipeline.py)")
        # a new upload of a state replaces the old one
        for name in [n for n in man["files"] if n.startswith(f"{package}/") and n not in found]:
            del man["files"][name]
        files.update(found)
    total = sum(p.stat().st_size for p in files.values())
    print(f"uploading {len(files)} files, {total / 1e9:.2f} GB, to {bucket}/{folder()}/")
    for name, path in files.items():
        s3.upload_file(str(path), bucket, f"{folder()}/{name}", Config=UPLOAD)
        man["files"][name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        print(f"  {name}: {path.stat().st_size / 1e6:.1f} MB")
    man["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    body = json.dumps(man, indent=2).encode()
    s3.put_object(Bucket=bucket, Key=f"{folder()}/manifest.json", Body=body, ContentLength=len(body))
    print("done, manifest.json updated")


def download(packages: list[str]) -> None:
    s3, bucket = client()
    man = manifest(s3, bucket)
    for package in packages:
        names = [n for n in man["files"] if n.startswith(f"{package}/")]
        if not names:
            sys.exit(f"the {package} state is not on our server ({bucket}/{folder()}/) yet")
        print(f"{package}: {len(names)} files, {sum(man['files'][n]['bytes'] for n in names) / 1e9:.2f} GB")
        for name in names:
            dst = local_path(name)
            if dst is None:
                sys.exit(f"unexpected file name in manifest.json: {name}")
            info = man["files"][name]
            if dst.exists() and dst.stat().st_size == info["bytes"] and sha256(dst) == info["sha256"]:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, f"{folder()}/{name}", str(dst), Config=DOWNLOAD)
            if dst.stat().st_size != info["bytes"] or sha256(dst) != info["sha256"]:
                sys.exit(f"{name}: the download is incomplete or damaged; try again")
            print(f"  {name}: {info['bytes'] / 1e6:.1f} MB")
        print(f"{package}: every file checked and in place")


def ensure(path: Path, step: str) -> Path:
    """The file at path. If it is missing and belongs to a state that config.yaml takes from our server (data:), that
    state is downloaded first; otherwise stop and name the step that makes it."""
    if path.exists():
        return path
    r = rel(path)
    for state in DOWNLOADED:
        if in_state(r, state):
            print(f"{r} is missing: downloading the {state} state from our server")
            download([state])
            if path.exists():
                return path
    sys.exit(f"{r} is missing: run {step} first, or start later (data.start_with in config.yaml)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["download", "upload", "list"])
    ap.add_argument("packages", nargs="*", default=[], help=", ".join(PACKAGES))
    args = ap.parse_args()
    unknown = [p for p in args.packages if p not in PACKAGES]
    if unknown:
        ap.error(f"unknown package {', '.join(unknown)}; choose from {', '.join(PACKAGES)}")
    if args.action == "list":
        s3, bucket = client()
        man = manifest(s3, bucket)
        for package in PACKAGES:
            names = [n for n in man["files"] if n.startswith(f"{package}/")]
            size = sum(man["files"][n]["bytes"] for n in names)
            print(f"{package:13s} {len(names):5d} files {size / 1e9:8.2f} GB")
        print(f"{bucket}/{folder()}/, updated {man.get('updated', '-')}")
    elif args.action == "upload":
        if not args.packages:
            ap.error(f"name what to upload: {', '.join(PACKAGES)}")
        with tempfile.TemporaryDirectory() as tmp:
            upload(args.packages, Path(tmp))
    else:
        packages = args.packages or DOWNLOADED
        if not packages:
            ap.error("data.start_with is raw, so nothing is downloaded: name a state or change config.yaml")
        download(packages)


if __name__ == "__main__":
    main()
