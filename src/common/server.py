"""Our prepared data on the team's server: list, download and upload what the steps wrote below data/.

Usage:  uv run python -m src.common.server list
        uv run python -m src.common.server download [01_selection|02_preprocessing|03_transformation|<folder> ...]
        uv run python -m src.common.server upload 01_selection|02_preprocessing|03_transformation|<folder> ...

The server holds a copy of data/ (Data-Mining-Project/<month>/ in our bucket), without the work/ folders and the
logs. A name may be a step or a folder inside it, e.g. 02_preprocessing/stockfish. `download` without a name takes
the steps of data: in config.yaml. `upload` needs a key that may write.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

from src.config import DATA, DOWNLOADED, MONTH, START_WITH, STATES, STEPS, STOCKFISH, STOCKFISH_DIR, s3_settings

PROJECT = "Data-Mining-Project"
MONTH_FOLDERS = {"2026-09": "September 2026", "2026-08": "August 2026"}
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
    """The path below data/, e.g. 02_preprocessing/stratified/train.parquet"""
    return path.resolve().relative_to(DATA.resolve()).as_posix()


def is_shared(name: str) -> bool:
    """A file below data/ that belongs on the server: in a step's folder, not in work/ or logs/, not half written."""
    parts = name.split("/")
    return (parts[0] in STEPS and "work" not in parts and "logs" not in parts
            and not name.endswith((".part", ".log")))


def under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix.rstrip("/") + "/")


def check_names(names: list[str]) -> None:
    for n in names:
        if n.split("/")[0] not in STEPS:
            sys.exit(f"unknown name {n!r}: a step ({', '.join(STEPS)}) or a folder inside it")


def local_files(prefix: str) -> dict[str, Path]:
    """The files on this computer below data/<prefix> that belong on the server."""
    files = {}
    root = DATA / prefix
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for path in paths:
        if path.is_file() and is_shared(rel(path)):
            files[rel(path)] = path
    return files


def upload(names: list[str]) -> None:
    s3, bucket = client()
    man = manifest(s3, bucket)
    files = {}
    for prefix in names:
        found = local_files(prefix)
        if not found:
            sys.exit(f"nothing of {prefix} on this computer: compute it first (src/pipeline.py)")
        # a new upload of a folder replaces the old one
        for name in [n for n in man["files"] if under(n, prefix) and n not in found]:
            s3.delete_object(Bucket=bucket, Key=f"{folder()}/{name}")
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


def fetch(s3, bucket: str, man: dict, name: str) -> None:
    """Download one file of the manifest to data/<name>, unless it is already there and the same."""
    info = man["files"][name]
    dst = DATA / name
    if not dst.resolve().is_relative_to(DATA.resolve()) or not is_shared(name):
        sys.exit(f"unexpected file name in manifest.json: {name}")
    if dst.exists() and dst.stat().st_size == info["bytes"] and sha256(dst) == info["sha256"]:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, f"{folder()}/{name}", str(dst), Config=DOWNLOAD)
    if dst.stat().st_size != info["bytes"] or sha256(dst) != info["sha256"]:
        sys.exit(f"{name}: the download is incomplete or damaged; try again")
    print(f"  {name}: {info['bytes'] / 1e6:.1f} MB")


def download(names: list[str]) -> None:
    s3, bucket = client()
    man = manifest(s3, bucket)
    for prefix in names:
        found = [n for n in man["files"] if under(n, prefix)]
        if not found:
            sys.exit(f"{prefix} is not on our server ({bucket}/{folder()}/) yet")
        print(f"{prefix}: {len(found)} files, {sum(man['files'][n]['bytes'] for n in found) / 1e9:.2f} GB")
        for name in found:
            fetch(s3, bucket, man, name)
        print(f"{prefix}: every file checked and in place")


def from_server(name: str) -> bool:
    """config.yaml takes this file from our server instead of computing it: its step is not computed here, or it is
    a Stockfish result and data.stockfish is download."""
    step = name.split("/")[0]
    if START_WITH != "raw" and step in STEPS and STEPS.index(step) <= STATES.index(START_WITH) - 1:
        return True
    return STOCKFISH == "download" and under(name, rel(STOCKFISH_DIR))


def ensure(path: Path, step: str) -> Path:
    """The file at path. If it is missing and config.yaml takes it from our server (data:), it is downloaded first;
    otherwise stop and name the step that makes it."""
    if path.exists():
        return path
    name = rel(path)
    if from_server(name):
        s3, bucket = client()
        man = manifest(s3, bucket)
        if name in man["files"]:
            print(f"{name} is missing: downloading it from our server")
            fetch(s3, bucket, man, name)
            return path
    sys.exit(f"{name} is missing: run {step} first, or start later (data.start_with in config.yaml)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["download", "upload", "list"])
    ap.add_argument("names", nargs="*", default=[], help=", ".join(STEPS) + " or a folder inside one")
    args = ap.parse_args()
    check_names(args.names)
    if args.action == "list":
        s3, bucket = client()
        man = manifest(s3, bucket)
        folders = {}
        for name, info in man["files"].items():
            parts = name.split("/")
            key = "/".join(parts[:2]) if len(parts) > 2 else parts[0]
            n, b = folders.get(key, (0, 0))
            folders[key] = (n + 1, b + info["bytes"])
        for key, (n, b) in sorted(folders.items()):
            print(f"{key:40s} {n:5d} files {b / 1e9:8.2f} GB")
        print(f"{bucket}/{folder()}/, updated {man.get('updated', '-')}")
    elif args.action == "upload":
        if not args.names:
            ap.error(f"name what to upload: {', '.join(STEPS)} or a folder inside one")
        upload(args.names)
    else:
        names = args.names or DOWNLOADED
        if not names:
            ap.error("data.start_with is raw, so nothing is downloaded: name a step or change config.yaml")
        download(names)


if __name__ == "__main__":
    main()
