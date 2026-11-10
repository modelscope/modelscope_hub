"""Repeatable source ranges and compact SHA256 manifests."""

from __future__ import annotations

import hashlib
import os
import stat
from array import array
from pathlib import Path
from typing import Any

from ._multipart_control import UploadExecution
from ._multipart_protocol import MultipartError, MultipartPlan
from .utils._sha256_state import SHA256State, prefix_header


def _integrity(message: str) -> MultipartError:
    error = MultipartError(message)
    error.error_code = "E2020"
    return error


class UploadSource:
    def __init__(self, data: str | Path | bytes, size: int, sha256: str, chunk_size: int) -> None:
        self.data = data
        self.size = size
        self.sha256 = sha256
        self.chunk_size = chunk_size
        self.identity = None if isinstance(data, bytes) else self._identity(os.stat(data))
        self.verify()

    @staticmethod
    def _identity(info: os.stat_result) -> tuple[int, ...]:
        if not stat.S_ISREG(info.st_mode):
            raise _integrity("Multipart source must be a regular file.")
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns

    def verify(self) -> None:
        if isinstance(self.data, bytes):
            if len(self.data) != self.size:
                raise _integrity("Multipart source size changed.")
        else:
            current = self._identity(os.stat(self.data))
            if current != self.identity or current[2] != self.size:
                raise _integrity("Multipart source changed; the original upload cannot continue.")

    def reader(self, offset: int, length: int, expected: bytes | None) -> RangeReader:
        self.verify()
        return RangeReader(self, offset, length, expected)

    def verify_content(self, execution: UploadExecution) -> None:
        digest = hashlib.sha256()
        with execution.scanner_slot(), self.reader(0, self.size, None) as stream:
            while chunk := stream.read(self.chunk_size):
                execution.check()
                digest.update(chunk)
            stream.verify()
        if digest.hexdigest() != self.sha256:
            raise _integrity("Multipart source content does not match its declared SHA256.")


class RangeReader:
    def __init__(self, source: UploadSource, offset: int, length: int, expected: bytes | None) -> None:
        if offset < 0 or length < 0 or offset + length > source.size:
            raise _integrity("Invalid upload source range.")
        self.source = source
        self.offset = offset
        self.length = length
        self.expected = expected
        self.bytes_read = 0
        self.hasher = hashlib.sha256()
        self._stream: Any = None
        if not isinstance(source.data, bytes):
            self._stream = open(source.data, "rb")
            try:
                if source._identity(os.fstat(self._stream.fileno())) != source.identity:
                    raise _integrity("Multipart source was replaced before reading.")
                self._stream.seek(offset)
            except BaseException:
                self._stream.close()
                raise

    def __len__(self) -> int:
        return self.length

    def tell(self) -> int:
        return self.bytes_read

    def read(self, size: int = -1) -> bytes:
        remaining = self.length - self.bytes_read
        if remaining <= 0:
            return b""
        count = min(remaining, self.source.chunk_size, size if size >= 0 else self.source.chunk_size)
        if isinstance(self.source.data, bytes):
            start = self.offset + self.bytes_read
            chunk = self.source.data[start : start + count]
        else:
            chunk = self._stream.read(count)
        if not chunk and count:
            raise _integrity("Multipart source ended before the requested range.")
        self.bytes_read += len(chunk)
        self.hasher.update(chunk)
        return chunk

    def verify(self) -> None:
        if self.bytes_read != self.length or (self.expected is not None and self.hasher.digest() != self.expected):
            raise _integrity("Actual upload bytes do not match the scanned source.")
        self.source.verify()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()

    def __enter__(self) -> RangeReader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class HashManifest:
    def __init__(self, source: UploadSource, plan: MultipartPlan, execution: UploadExecution) -> None:
        self.words: array = array("I")
        self.digests = bytearray()
        self.part_size = plan.part_size
        state = SHA256State()
        complete = hashlib.sha256()
        with execution.scanner_slot():
            for number in range(1, plan.total_parts + 1):
                execution.check()
                offset = (number - 1) * plan.part_size
                count, words = state.export_state()
                if count != offset or len(words) != 8:
                    raise _integrity("SHA256 prefix does not match the source offset.")
                self.words.extend(words)
                with source.reader(offset, plan.length(number), None) as stream:
                    while chunk := stream.read(source.chunk_size):
                        execution.check()
                        state.update(chunk)
                        complete.update(chunk)
                    stream.verify()
                    self.digests.extend(stream.hasher.digest())
        if complete.hexdigest() != source.sha256 or state.digest() != complete.digest():
            raise _integrity("Scanned multipart source does not match its declared SHA256.")
        source.verify()

    def digest(self, number: int) -> bytes:
        offset = (number - 1) * 32
        return bytes(self.digests[offset : offset + 32])

    def header(self, number: int) -> str:
        start = (number - 1) * 8
        return prefix_header((number - 1) * self.part_size, tuple(self.words[start : start + 8]))
