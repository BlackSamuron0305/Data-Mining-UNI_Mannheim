"""The raw Lichess monthly files on the team's S3 storage, read front to back, one zstd frame at a time.

The files are the unchanged .pgn.zst files of database.lichess.org. Such a file is a sequence of independent
zstd frames of about 14,000 games each, so every frame can be decompressed and parsed on its own. The frames
are cut out of the download by reading the frame and block headers as the bytes arrive; nothing is written to
disk. A dropped connection is resumed at the byte where it stopped, and the sha256 of all bytes is computed on
the way, so that it can be compared with Lichess's published sum.
"""
from __future__ import annotations

import hashlib
import io
import logging
import time
from collections.abc import Iterator

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from urllib3.exceptions import HTTPError

from src.config import CFG, s3_settings

log = logging.getLogger("storage")
ZSTD_MAGIC = 0xFD2FB528
SKIPPABLE = 0x184D2A50           # the magic numbers 0x184D2A50 to 0x184D2A5F mark skippable frames
RETRIES = 8
BUFFER = 8 << 20                 # bytes per read from the connection


def month_key(month: str) -> str:
    return CFG["source"]["key"].format(month=month)


class _Download(io.RawIOBase):
    """One file of the storage, readable only forward; a dropped connection is opened again at the same byte."""

    def __init__(self, key: str):
        s = s3_settings()
        self.endpoint, self.bucket, self.key = s["LICHESS_S3_ENDPOINT"], s["LICHESS_S3_BUCKET"], key
        self.s3 = boto3.client("s3", endpoint_url=self.endpoint, aws_access_key_id=s["LICHESS_S3_ACCESS_KEY"],
                               aws_secret_access_key=s["LICHESS_S3_SECRET_KEY"], region_name="us-east-1",
                               config=Config(s3={"addressing_style": "path"}, connect_timeout=30, read_timeout=120))
        self.size = self.s3.head_object(Bucket=self.bucket, Key=key)["ContentLength"]
        self.pos, self.body = 0, None

    def readable(self) -> bool:
        return True

    def readinto(self, buf) -> int:
        if self.pos >= self.size:
            return 0
        for attempt in range(1, RETRIES + 1):
            try:
                if self.body is None:
                    self.body = self.s3.get_object(Bucket=self.bucket, Key=self.key, Range=f"bytes={self.pos}-")["Body"]
                data = self.body.read(len(buf))
                if not data:
                    raise ConnectionError("the connection closed early")
                buf[: len(data)] = data
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
    """A monthly file of the storage; `frames()` yields its zstd frames in order."""

    def __init__(self, month: str):
        self._raw = _Download(month_key(month))
        self.key, self.size = self._raw.key, self._raw.size
        self.location = f"{self._raw.endpoint}/{self._raw.bucket}/{self.key}"
        self.sha256: str | None = None          # set once the whole file has been read

    def frames(self, limit: int | None = None) -> Iterator[bytes]:
        """The zstd frames of the file (skippable frames are passed over); only the first `limit` if given."""
        f, sha, pos, count = io.BufferedReader(self._raw, BUFFER), hashlib.sha256(), 0, 0

        def read(n: int) -> bytes:
            nonlocal pos
            b = f.read(n)
            if len(b) < n:
                raise ValueError(f"{self.key} ends inside a frame at byte {pos:,}: incomplete file")
            sha.update(b)
            pos += n
            return b

        while limit is None or count < limit:
            if not f.peek(1):
                if pos != self.size:
                    raise ValueError(f"{self.key}: read {pos:,} of {self.size:,} bytes")
                self.sha256 = sha.hexdigest()
                return
            head = read(4)
            magic = int.from_bytes(head, "little")
            if magic & 0xFFFFFFF0 == SKIPPABLE:
                read(int.from_bytes(read(4), "little"))
                continue
            if magic != ZSTD_MAGIC:
                raise ValueError(f"{self.key}: no zstd frame at byte {pos - 4:,}")
            fhd = read(1)
            d = fhd[0]                                    # frame header: window, dictionary id, content size
            single = d >> 5 & 1
            parts = [head, fhd, read((1 - single) + (0, 1, 2, 4)[d & 3] + (single, 2, 4, 8)[d >> 6])]
            while True:                                   # blocks: 3-byte header (last flag, type, size), body
                hdr = read(3)
                b = int.from_bytes(hdr, "little")
                parts += [hdr, read(1 if (b >> 1 & 3) == 1 else b >> 3)]
                if b & 1:
                    break
            if d >> 2 & 1:
                parts.append(read(4))                     # content checksum
            count += 1
            yield b"".join(parts)
