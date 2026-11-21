"""Read a raw Lichess monthly file one zstd frame at a time, from a local folder or from the team's S3 storage,
or download it into the local folder.

The local folder is LICHESS_LOCAL_DIR (environment or .env), else source.local_dir of config.yaml, else cache/raw/.
"""
from __future__ import annotations

import hashlib
import io
import logging
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from urllib3.exceptions import HTTPError

from src.config import CFG, ROOT, env, s3_settings

log = logging.getLogger("storage")
ZSTD_MAGIC = 0xFD2FB528
SKIPPABLE_MAGIC = 0x184D2A50
RETRIES = 8
BUFFER = 8 << 20
DOWNLOAD = TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20, max_concurrency=16)
SHA256_URL = "https://database.lichess.org/standard/sha256sums.txt"


def month_key(month: str) -> str:
    return CFG["source"]["key"].format(month=month)


def local_dir() -> Path:
    """The folder for the monthly files on this computer."""
    folder = Path(env("LICHESS_LOCAL_DIR") or CFG["source"].get("local_dir") or "cache/raw")
    if not folder.is_absolute():
        folder = ROOT / folder
    return folder


def local_file(month: str) -> Path | None:
    """The monthly file in the local folder, or None."""
    path = local_dir() / month_key(month)
    if path.is_file():
        return path
    return None


def lichess_sha256(name: str) -> str | None:
    """The sha256 that Lichess publishes for a monthly file (None if Lichess cannot be reached)."""
    try:
        with urllib.request.urlopen(SHA256_URL, timeout=30) as response:
            text = response.read().decode("utf-8")
    except OSError as err:
        log.warning("could not read %s (%s)", SHA256_URL, err)
        return None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == name:
            return parts[0]
    return None


def download(month: str) -> Path:
    """Copy a monthly file from S3 into the local folder, in many parts at once, and check it."""
    name = month_key(month)
    dst = local_dir() / name
    s = s3_settings()
    s3 = boto3.client("s3", endpoint_url=s["LICHESS_S3_ENDPOINT"], aws_access_key_id=s["LICHESS_S3_ACCESS_KEY"],
                      aws_secret_access_key=s["LICHESS_S3_SECRET_KEY"], region_name="us-east-1",
                      config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 10}))
    size = s3.head_object(Bucket=s["LICHESS_S3_BUCKET"], Key=name)["ContentLength"]
    if dst.is_file() and dst.stat().st_size == size:
        log.info("%s is already there", dst)
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s (%.1f GB) to %s", name, size / 1e9, dst.parent)
    t0 = time.time()
    # boto3 writes to a temporary name and renames it at the end, so a stopped download leaves no half file
    s3.download_file(s["LICHESS_S3_BUCKET"], name, str(dst), Config=DOWNLOAD)
    log.info("done in %.1f minutes, checking the sha256", (time.time() - t0) / 60)
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        while chunk := f.read(BUFFER):
            h.update(chunk)
    expected = lichess_sha256(name)
    if expected is None:
        log.warning("sha256 %s (not compared: no sum from Lichess)", h.hexdigest())
    elif h.hexdigest() != expected:
        dst.unlink()
        raise ValueError(f"{name}: the sha256 differs from Lichess's, file deleted; try again")
    else:
        log.info("sha256 matches Lichess's")
    return dst


class S3File(io.RawIOBase):
    """A file on S3, read front to back; a dropped connection continues at the same byte."""

    def __init__(self, key: str):
        s = s3_settings()
        self.endpoint = s["LICHESS_S3_ENDPOINT"]
        self.bucket = s["LICHESS_S3_BUCKET"]
        self.key = key
        self.s3 = boto3.client("s3", endpoint_url=self.endpoint, aws_access_key_id=s["LICHESS_S3_ACCESS_KEY"],
                               aws_secret_access_key=s["LICHESS_S3_SECRET_KEY"], region_name="us-east-1",
                               config=Config(s3={"addressing_style": "path"}, connect_timeout=30, read_timeout=120))
        self.size = self.s3.head_object(Bucket=self.bucket, Key=key)["ContentLength"]
        self.pos = 0
        self.body = None

    def readable(self) -> bool:
        return True

    def readinto(self, buf) -> int:
        if self.pos >= self.size:
            return 0
        for attempt in range(1, RETRIES + 1):
            try:
                if self.body is None:
                    response = self.s3.get_object(Bucket=self.bucket, Key=self.key, Range=f"bytes={self.pos}-")
                    self.body = response["Body"]
                data = self.body.read(len(buf))
                if not data:
                    raise ConnectionError("the connection closed early")
                buf[:len(data)] = data
                self.pos += len(data)
                return len(data)
            except (BotoCoreError, ClientError, HTTPError, OSError) as err:
                self.body = None
                if attempt == RETRIES:
                    raise
                log.warning("download stopped at byte %s (%s); retry %d of %d", f"{self.pos:,}", err, attempt,
                            RETRIES - 1)
                time.sleep(min(2 ** attempt, 60))
        return 0


class MonthFile:
    """A monthly file; `frames()` yields its zstd frames in order and then sets `sha256`."""

    def __init__(self, month: str):
        local = local_file(month)
        if local is not None:
            self.raw = io.FileIO(local, "rb")
            self.key = local.name
            self.size = local.stat().st_size
            self.location = str(local)
        else:
            self.raw = S3File(month_key(month))
            self.key = self.raw.key
            self.size = self.raw.size
            self.location = f"{self.raw.endpoint}/{self.raw.bucket}/{self.key}"
        self.sha256 = None

    def frames(self, limit: int | None = None) -> Iterator[bytes]:
        """The zstd frames of the file (skippable frames left out); only the first `limit` if given."""
        f = io.BufferedReader(self.raw, BUFFER)
        sha = hashlib.sha256()
        pos = 0
        count = 0

        def read(n: int) -> bytes:
            nonlocal pos
            data = f.read(n)
            if len(data) < n:
                raise ValueError(f"{self.key} ends inside a frame at byte {pos:,}: incomplete file")
            sha.update(data)
            pos += n
            return data

        while limit is None or count < limit:
            if not f.peek(1):
                if pos != self.size:
                    raise ValueError(f"{self.key}: read {pos:,} of {self.size:,} bytes")
                self.sha256 = sha.hexdigest()
                return
            magic_bytes = read(4)
            magic = int.from_bytes(magic_bytes, "little")
            if magic & 0xFFFFFFF0 == SKIPPABLE_MAGIC:
                skip = int.from_bytes(read(4), "little")
                read(skip)
                continue
            if magic != ZSTD_MAGIC:
                raise ValueError(f"{self.key}: no zstd frame at byte {pos - 4:,}")

            # frame header (zstd format, RFC 8878)
            header = read(1)
            flags = header[0]
            single_segment = flags >> 5 & 1
            window_size = 1 - single_segment
            dict_id_size = (0, 1, 2, 4)[flags & 3]
            content_size_size = (single_segment, 2, 4, 8)[flags >> 6]
            parts = [magic_bytes, header, read(window_size + dict_id_size + content_size_size)]

            # blocks: a 3-byte header (last block flag, type, size) and the block
            while True:
                block_header = read(3)
                b = int.from_bytes(block_header, "little")
                block_type = b >> 1 & 3
                if block_type == 1:
                    block_size = 1
                else:
                    block_size = b >> 3
                parts.append(block_header)
                parts.append(read(block_size))
                if b & 1:
                    break
            if flags >> 2 & 1:
                parts.append(read(4))
            count += 1
            yield b"".join(parts)
