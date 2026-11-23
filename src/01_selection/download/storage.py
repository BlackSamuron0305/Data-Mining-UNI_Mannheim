"""The raw month: download it, or read it frame by frame from this computer or our server (data.raw_from in
config.yaml). Used by download/september.py, download/august.py and the cleanup."""
from __future__ import annotations

import hashlib
import io
import logging
import time
import urllib.request
from collections.abc import Iterator
from compression import zstd
from pathlib import Path

from boto3.s3.transfer import TransferConfig
from botocore.exceptions import BotoCoreError, ClientError
from urllib3.exceptions import HTTPError

from src.common.server import client
from src.config import MONTH_NAMES, RAW_ROOT, RAW_FROM, ROOT, env

log = logging.getLogger("storage")
ZSTD_MAGIC = 0xFD2FB528         # the first 4 bytes of every zstd frame
SKIPPABLE_MAGIC = 0x184D2A50    # ... of a frame without content (the last 4 bits may vary)
MAX_FRAME = 1 << 28             # Lichess's frames have a few MB; a frame not complete after 256 MB means a damaged file
RETRIES = 8
BUFFER = 8 << 20
DOWNLOAD = TransferConfig(multipart_threshold=64 << 20, multipart_chunksize=64 << 20, max_concurrency=16)
LICHESS_URL = "https://database.lichess.org/standard/"
KEY = "lichess_db_standard_rated_{month}.pgn.zst"   # the name of a month's file, at Lichess and on our server
SHA256_URL = LICHESS_URL + "sha256sums.txt"


def month_key(month: str) -> str:
    return KEY.format(month=month)


# Folder for a raw month on this computer: data/01_selection/raw/<month>/, or LICHESS_LOCAL_DIR (.env) if set
def local_dir(month: str) -> Path:
    other = env("LICHESS_LOCAL_DIR")
    if not other:
        return RAW_ROOT / MONTH_NAMES[month]
    folder = Path(other)
    if not folder.is_absolute():
        folder = ROOT / folder
    return folder


# The raw month on this computer, or None
def local_file(month: str) -> Path | None:
    path = local_dir(month) / month_key(month)
    if path.is_file():
        return path
    return None


# Checksum that Lichess publishes for a file (None if Lichess cannot be reached)
def lichess_sha256(name: str) -> str | None:
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


# Download a raw month from Lichess (slower than our server, but needs no key)
def download_from_lichess(name: str, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(LICHESS_URL + name, timeout=60) as response, open(tmp, "wb") as f:
        while chunk := response.read(BUFFER):
            f.write(chunk)
    tmp.replace(dst)


# Compare the file with Lichess's checksum and delete it if they differ
def check_sha256(name: str, dst: Path) -> None:
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


# Download a raw month: from Lichess, or from our server with data.raw_from: server
def download(month: str) -> Path:
    name = month_key(month)
    dst = local_dir(month) / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if RAW_FROM == "lichess":
        if dst.is_file():
            log.info("%s is already there", dst)
            return dst
        log.info("downloading %s from Lichess to %s", name, dst.parent)
        t0 = time.time()
        download_from_lichess(name, dst)
        log.info("done in %.1f minutes, checking the sha256", (time.time() - t0) / 60)
        check_sha256(name, dst)
        return dst
    s3, bucket = client()
    size = s3.head_object(Bucket=bucket, Key=name)["ContentLength"]
    if dst.is_file() and dst.stat().st_size == size:
        log.info("%s is already there", dst)
        return dst
    log.info("downloading %s (%.1f GB) to %s", name, size / 1e9, dst.parent)
    t0 = time.time()
    # a stopped download leaves no half file
    s3.download_file(bucket, name, str(dst), Config=DOWNLOAD)
    log.info("done in %.1f minutes, checking the sha256", (time.time() - t0) / 60)
    check_sha256(name, dst)
    return dst


# A file on our server, read from start to end; continues where it stopped if the connection drops
class S3File(io.RawIOBase):

    def __init__(self, key: str):
        self.s3, self.bucket = client(connect_timeout=30, read_timeout=120)
        self.endpoint = self.s3.meta.endpoint_url
        self.key = key
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


# A raw month (a file on this computer, else downloaded from Lichess or read from our server); frames() gives its
# pieces in order
class MonthFile:

    def __init__(self, month: str):
        local = local_file(month)
        if local is None and RAW_FROM == "lichess":
            local = download(month)
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
        """The compressed frames of the file in order (only the first `limit` if given).

        Lichess compresses a month as thousands of independent zstd frames of a few MB each, so every frame can be
        decompressed on its own and the cleanup can share them out to several processes. We read the file in
        chunks and cut off one whole frame at a time; zstd itself tells how long a frame is."""
        f = io.BufferedReader(self.raw, BUFFER)
        sha = hashlib.sha256()
        data = bytearray()      # read, but not given out yet
        pos = 0                 # where `data` starts in the file
        count = 0
        while limit is None or count < limit:
            size = frame_size(data, pos)
            while size is None:
                chunk = f.read(BUFFER)
                if not chunk:
                    if data:
                        raise ValueError(f"{self.key} ends inside a frame at byte {pos:,}: incomplete file")
                    if pos != self.size:
                        raise ValueError(f"{self.key}: read {pos:,} of {self.size:,} bytes")
                    self.sha256 = sha.hexdigest()
                    return
                sha.update(chunk)
                data += chunk
                size = frame_size(data, pos)
            frame = bytes(data[:size])
            del data[:size]
            pos += size
            if not is_skippable(frame):
                count += 1
                yield frame


def is_skippable(data) -> bool:
    """A frame without content (zstd allows such frames for metadata); there are none in Lichess's files so far."""
    return int.from_bytes(data[:4], "little") & 0xFFFFFFF0 == SKIPPABLE_MAGIC


def frame_size(data, pos: int) -> int | None:
    """The length of the frame at the start of `data`, or None if `data` does not hold the whole frame yet."""
    if len(data) < 8:
        return None
    if is_skippable(data):
        size = 8 + int.from_bytes(data[4:8], "little")
        return size if len(data) >= size else None
    if int.from_bytes(data[:4], "little") != ZSTD_MAGIC:
        raise ValueError(f"no zstd frame at byte {pos:,}")
    try:
        return zstd.get_frame_size(data)
    except zstd.ZstdError:
        if len(data) > MAX_FRAME:
            raise ValueError(f"no complete zstd frame in {MAX_FRAME:,} bytes from byte {pos:,}: damaged file")
        return None
