"""Repeatable source ranges and compact SHA256 manifests."""

from __future__ import annotations

import hashlib
import os
import stat
from array import array
from bisect import bisect_left
from collections.abc import Iterable
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
        with execution.scanner_slot(), self.reader(0, self.size, None) as stream:
            while stream.read(self.chunk_size):
                execution.check()
            stream.verify()
            if stream.hasher.hexdigest() != self.sha256:
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
    """Verified sparse part hashes and unpadded prefix states for one source plan."""

    def __init__(
        self,
        source: UploadSource,
        plan: MultipartPlan,
        execution: UploadExecution,
        *,
        required_parts: Iterable[int] | None = None,
    ) -> None:
        requested = tuple(range(1, plan.total_parts + 1)) if required_parts is None else tuple(required_parts)
        if not requested or any(type(n) is not int or not 1 <= n <= plan.total_parts for n in requested):
            raise MultipartError("Multipart manifest requires valid part numbers.")
        self._parts = tuple(sorted(set(requested)))
        selected = set(self._parts)
        self._source = source
        self._identity = source.identity
        self._sha256 = source.sha256
        self._size = source.size
        self._total_parts = plan.total_parts
        self.part_size = plan.part_size
        self.words: array = array("I")
        self.digests = bytearray()
        prefix_end = (self._parts[-1] - 1) * plan.part_size
        state = SHA256State()
        with execution.scanner_slot(), source.reader(0, source.size, None) as stream:
            for number in range(1, plan.total_parts + 1):
                execution.check()
                offset = (number - 1) * plan.part_size
                part_hash = None
                if number in selected:
                    count, words = state.export_state()
                    if count != offset or len(words) != 8:
                        raise _integrity("SHA256 prefix does not match the source offset.")
                    # Finalize copies for verification, never the state encoded in the header.
                    if state.digest() != stream.hasher.copy().digest():
                        raise _integrity("SHA256 prefix does not match the scanned source.")
                    self.words.extend(words)
                    part_hash = hashlib.sha256()
                remaining = plan.length(number)
                while remaining:
                    execution.check()
                    chunk = stream.read(min(source.chunk_size, remaining))
                    if offset < prefix_end:
                        state.update(chunk)
                    if part_hash is not None:
                        part_hash.update(chunk)
                    remaining -= len(chunk)
                if part_hash is not None:
                    self.digests.extend(part_hash.digest())
            stream.verify()
            if stream.hasher.hexdigest() != source.sha256:
                raise _integrity("Scanned multipart source does not match its declared SHA256.")
        source.verify()

    @property
    def coverage(self) -> tuple[int, ...]:
        return self._parts

    def matches(self, source: UploadSource, plan: MultipartPlan) -> bool:
        return (
            source is self._source
            and source.identity == self._identity
            and source.sha256 == plan.sha256 == self._sha256
            and source.size == plan.size == self._size
            and plan.part_size == self.part_size
            and plan.total_parts == self._total_parts
        )

    def _index(self, number: int) -> int:
        if type(number) is not int:
            raise MultipartError("Multipart manifest part number must be an integer.")
        index = bisect_left(self._parts, number)
        if index == len(self._parts) or self._parts[index] != number:
            raise MultipartError("Multipart manifest does not cover the requested part.")
        return index

    def covers(self, parts: Iterable[int]) -> bool:
        try:
            for number in parts:
                self._index(number)
        except MultipartError:
            return False
        return True

    def digest(self, number: int) -> bytes:
        offset = self._index(number) * 32
        return bytes(self.digests[offset : offset + 32])

    def header(self, number: int) -> str:
        start = self._index(number) * 8
        return prefix_header((number - 1) * self.part_size, tuple(self.words[start : start + 8]))
