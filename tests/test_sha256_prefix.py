"""Offline SHA256 prefix correctness and sparse scanning contracts."""

from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
import random
import socket
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import modelscope_hub._multipart_source as source_module
from modelscope_hub._legacy_api import LegacyClient
from modelscope_hub._multipart_config import MultipartSettings
from modelscope_hub._multipart_control import UploadExecution
from modelscope_hub._multipart_protocol import MultipartError, MultipartPlan
from modelscope_hub._multipart_source import HashManifest, RangeReader, UploadSource
from modelscope_hub.utils._sha256_state import SHA256State, prefix_header


def run_offline_tests(arguments: list[str]) -> int:
    """Run selected checks with temporary SDK state and a hard socket prohibition."""

    def blocked(*args, **kwargs):
        raise AssertionError("Real network access is forbidden in offline hash tests")

    with tempfile.TemporaryDirectory(prefix="hash-offline-", dir="temp") as directory:
        root = str(Path(directory).resolve())
        environment = {
            "HOME": root,
            "MODELSCOPE_HOME": root + "/sdk",
            "MODELSCOPE_CACHE": root + "/cache",
            "MODELSCOPE_RUN_REMOTE_TESTS": "false",
            "MODELSCOPE_API_TOKEN": "offline-test-identity",
        }
        with (
            patch.dict(os.environ, environment),
            patch.object(socket.socket, "connect", blocked),
            patch.object(socket.socket, "connect_ex", blocked),
            patch.object(socket, "create_connection", blocked),
        ):
            return pytest.main([*arguments, "--basetemp", root + "/pytest"])


@pytest.fixture
def execution():
    client = LegacyClient("offline-test-identity", "https://example.test", user_agent="modelscope_hub/test")
    settings = replace(MultipartSettings.load(client.endpoint), mode="required", chunk_size=97)
    with UploadExecution(settings, client, 1, False) as value:
        yield value
    client._session.close()
    client._read_session.close()


def fixture_source(payload: bytes, data=None, part_size: int = 128):
    sha = hashlib.sha256(payload).hexdigest()
    source = UploadSource(payload if data is None else data, len(payload), sha, 97)
    plan = MultipartPlan(
        "opaque-token", sha, len(payload), part_size, (len(payload) + part_size - 1) // part_size, 1, frozenset()
    )
    return source, plan


def resumed_digest(header: str, suffix: bytes) -> bytes:
    """Decode the wire fields, then verify the exported state by continuation."""
    fields = json.loads(base64.b64decode(header, validate=True))
    assert all(isinstance(value, str) for value in fields.values())
    assert fields["hash_type"] == "sha256" and fields["num"] == "0" and fields["md_len"] == "32"
    bits = int(fields["Nl"]) + (int(fields["Nh"]) << 32)
    assert bits % 512 == 0
    state = SHA256State()
    state._h = [int(fields[f"h{i}"]) for i in range(8)]
    state._count = bits // 8
    state.update(suffix)
    return state.digest()


@pytest.mark.parametrize("payload", [b"", b"abc", b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"])
def test_known_answers(payload):
    state = SHA256State()
    state.update(payload)
    assert state.digest() == hashlib.sha256(payload).digest()
    assert state.digest() == state.copy().digest()


@pytest.mark.parametrize("length", [55, 56, 63, 64, 65, 127, 128, 129, 2049])
@pytest.mark.parametrize("chunk", [1, 31, 64, 97])
def test_streaming_boundaries(length, chunk):
    rng = random.Random(length)
    payload = bytes(rng.randrange(256) for _ in range(length))
    state = SHA256State()
    for offset in range(0, length, chunk):
        state.update(payload[offset : offset + chunk])
    assert state.digest() == hashlib.sha256(payload).digest()


@pytest.mark.parametrize("parts", [None, [1], [2], [5], [2, 5], [5, 2, 2]])
@pytest.mark.parametrize("path_source", [False, True])
def test_sparse_manifest_and_exact_work(execution, tmp_path, monkeypatch, parts, path_source):
    payload = bytes(range(256)) * 2 + b"tail!"
    path = tmp_path / "source.bin"
    path.write_bytes(payload)
    source, plan = fixture_source(payload, path if path_source else None)
    counts = {"file": 0, "padding": 0, "read": 0, "readers": 0, "hashers": 0}

    class CountingState(SHA256State):
        def update(self, data):
            counts["file"] += len(data)
            return super().update(data)

        def copy(self):
            result = super().copy()
            original = result.update

            def padding(data):
                counts["padding"] += len(data)
                original(data)

            result.update = padding
            return result

    original_read = RangeReader.read
    original_reader = source.reader
    original_sha = hashlib.sha256

    def read(reader, size=-1):
        result = original_read(reader, size)
        counts["read"] += len(result)
        return result

    def reader(*args, **kwargs):
        counts["readers"] += 1
        return original_reader(*args, **kwargs)

    def sha(*args, **kwargs):
        counts["hashers"] += 1
        return original_sha(*args, **kwargs)

    monkeypatch.setattr(source_module, "SHA256State", CountingState)
    monkeypatch.setattr(RangeReader, "read", read)
    monkeypatch.setattr(source, "reader", reader)
    monkeypatch.setattr(source_module.hashlib, "sha256", sha)
    manifest = HashManifest(source, plan, execution, required_parts=parts)
    selected = tuple(range(1, plan.total_parts + 1)) if parts is None else tuple(sorted(set(parts)))
    assert manifest.coverage == selected
    assert counts == {
        "file": (max(selected) - 1) * plan.part_size,
        "padding": 64 * len(selected),
        "read": len(payload),
        "readers": 1,
        "hashers": len(selected) + 1,
    }
    assert manifest.matches(source, plan)
    assert manifest.matches(source, replace(plan, token="refreshed"))
    assert not manifest.matches(source, replace(plan, part_size=64))
    assert not manifest.matches(source, replace(plan, total_parts=plan.total_parts + 1))
    assert not manifest.matches(fixture_source(payload)[0], plan)
    assert manifest.covers(selected)
    for number in selected:
        offset = (number - 1) * plan.part_size
        assert manifest.digest(number) == original_sha(payload[offset : offset + plan.length(number)]).digest()
        assert resumed_digest(manifest.header(number), payload[offset:]) == original_sha(payload).digest()
    for number in {0, True, plan.total_parts + 1, "1"}:
        assert not manifest.covers([number])
        with pytest.raises(MultipartError):
            manifest.digest(number)
    for number in set(range(1, plan.total_parts + 1)) - set(selected):
        with pytest.raises(MultipartError, match="cover"):
            manifest.header(number)


@pytest.mark.parametrize("parts", [[], [True], [False], [0], [9], [1.0], ["1"]])
def test_invalid_selection_never_scans(execution, monkeypatch, parts):
    source, plan = fixture_source(b"x" * 300)
    monkeypatch.setattr(source, "reader", lambda *args: pytest.fail("Invalid selection started a scan"))
    with pytest.raises(MultipartError):
        HashManifest(source, plan, execution, required_parts=parts)


def test_corrupt_prefix_fails_cross_check(execution, monkeypatch):
    source, plan = fixture_source(b"x" * 300)

    class CorruptState(SHA256State):
        def update(self, data):
            super().update(data)
            self._h[0] ^= 1

    monkeypatch.setattr(source_module, "SHA256State", CorruptState)
    with pytest.raises(MultipartError, match="prefix"):
        HashManifest(source, plan, execution, required_parts=[2])


@pytest.mark.parametrize("mutation", ["same_size", "replace", "shorten"])
def test_file_identity_changes_fail(execution, tmp_path, mutation):
    payload = b"original" * 80
    path = tmp_path / "source.bin"
    path.write_bytes(payload)
    source, plan = fixture_source(payload, path)
    if mutation == "replace":
        replacement = tmp_path / "replacement.bin"
        replacement.write_bytes(payload)
        replacement.replace(path)
    else:
        path.write_bytes(b"changed!" * (80 if mutation == "same_size" else 40))
    with pytest.raises(MultipartError):
        HashManifest(source, plan, execution, required_parts=[1])


def test_first_part_still_verifies_the_entire_tail(execution):
    payload = b"x" * 512 + b"original"
    changed = payload[:-8] + b"modified"
    source, plan = fixture_source(payload, changed)
    with pytest.raises(MultipartError, match="declared"):
        HashManifest(source, plan, execution, required_parts=[1])


def test_source_io_and_reader_release(execution, tmp_path, monkeypatch):
    path = tmp_path / "source.bin"
    path.write_bytes(b"x" * 300)
    source, plan = fixture_source(path.read_bytes(), path)
    streams = []
    original = source.reader

    def reader(*args):
        result = original(*args)
        streams.append(result)
        return result

    monkeypatch.setattr(source, "reader", reader)
    monkeypatch.setattr(RangeReader, "read", lambda *args: (_ for _ in ()).throw(OSError(errno.EIO, "disk error")))
    with pytest.raises(OSError):
        HashManifest(source, plan, execution, required_parts=[1])
    assert streams[0]._stream.closed


def test_single_part_uses_iv_without_file_compression(execution):
    payload = b"x" * 65
    source, plan = fixture_source(payload)
    manifest = HashManifest(source, plan, execution)
    assert manifest.coverage == (1,)
    assert tuple(manifest.words) == SHA256State().export_state()[1]
    assert manifest.digest(1) == hashlib.sha256(payload).digest()
    assert resumed_digest(manifest.header(1), payload) == hashlib.sha256(payload).digest()


@pytest.mark.parametrize("words", [(1,) * 7, (True,) * 8, (-1,) * 8, (2**32,) * 8])
def test_invalid_prefix_words(words):
    with pytest.raises(MultipartError):
        prefix_header(0, words)


def test_verify_content_uses_one_hasher(execution, monkeypatch):
    source, _ = fixture_source(b"x" * 300)
    original = hashlib.sha256
    calls = []

    def sha(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(source_module.hashlib, "sha256", sha)
    source.verify_content(execution)
    assert len(calls) == 1
    source.sha256 = "0" * 64
    with pytest.raises(MultipartError):
        source.verify_content(execution)


@pytest.mark.parametrize("offset", [0, 64, 512 * 1024**2 - 64, 512 * 1024**2, 4 * 1024**3 + 64])
def test_synthetic_large_bit_counts(offset):
    # These exercise encoding/bit carry, not a real multi-gigabyte hash scan.
    words = SHA256State().export_state()[1]
    fields = json.loads(base64.b64decode(prefix_header(offset, words)))
    assert int(fields["Nl"]) == (offset * 8) & 0xFFFFFFFF
    assert int(fields["Nh"]) == (offset * 8) >> 32
    assert fields["num"] == "0"


@pytest.mark.parametrize("offset", [-64, 1, 2**61])
def test_invalid_prefix_boundary(offset):
    with pytest.raises(MultipartError):
        prefix_header(offset, SHA256State().export_state()[1])


def test_scanner_cancelled_before_read(monkeypatch):
    source, plan = fixture_source(b"x" * 300)
    execution = SimpleNamespace(scanner_slot=lambda: (_ for _ in ()).throw(MultipartError("cancelled")))
    monkeypatch.setattr(source, "reader", lambda *args: pytest.fail("Cancelled scanner read source"))
    with pytest.raises(MultipartError):
        HashManifest(source, plan, execution)
