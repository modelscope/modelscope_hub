"""Offline transport-level multipart recovery and pipeline regressions."""

from __future__ import annotations

import errno
import hashlib
import importlib
import io
import json
import socket
import threading
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

import modelscope_hub._multipart_source as source_module
import modelscope_hub._multipart_upload as multipart_module
import modelscope_hub._upload as upload_module
from modelscope_hub import HubApi, constants
from modelscope_hub._multipart_config import MultipartSettings
from modelscope_hub._multipart_control import UploadExecution
from modelscope_hub._multipart_protocol import _ERROR_ROWS, MultipartError, MultipartPlan
from modelscope_hub._multipart_source import RangeReader, UploadSource
from modelscope_hub._multipart_state import UploadState
from modelscope_hub._multipart_upload import MultipartUploader
from modelscope_hub._upload import BatchTracker, _stop_on_error
from modelscope_hub.config import HubConfig
from modelscope_hub.errors import StorageError
from tests.test_sha256_prefix import resumed_digest

PART = 100 * 1024


def response(request, data, *, status=200, code=200, success=True):
    result = requests.Response()
    result.status_code = status
    result.url = request.url
    result.request = request
    result._content = json.dumps({"Success": success, "Code": code, "Data": data, "RequestId": "offline"}).encode()
    result._content_consumed = True
    result.raw = io.BytesIO(result._content)
    return result


def fault(request, name, action, number=None, status=422):
    row = next(row for row in _ERROR_ROWS if row[0] == name)
    detail = {"errorCode": row[0], "recoveryAction": row[1], "action": action}
    if number is not None:
        detail["partNumber"] = number
    return response(request, detail, status=status, code=10030001001 + _ERROR_ROWS.index(row), success=False)


class Mirror:
    """Intercept adapters only; real sessions, prepared requests and readers still run."""

    def __init__(self, monkeypatch, tmp_path):
        self.states = {}
        self.requests = []
        self.parts = []
        self.commits = []
        self.hook = None
        self.lock = threading.RLock()
        self.manifests = []
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("MODELSCOPE_HOME", str(tmp_path / "sdk"))
        monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "cache"))
        monkeypatch.setenv("MODELSCOPE_API_TOKEN", "offline-test-identity")
        monkeypatch.setattr(constants, "UPLOAD_MULTIPART_MODE", "required")
        monkeypatch.setattr(constants, "UPLOAD_MULTIPART_THRESHOLD_BYTES", 1)
        monkeypatch.setattr(constants, "UPLOAD_MULTIPART_PART_WORKERS", 1)
        monkeypatch.setattr(upload_module, "UPLOAD_FAILED_FILE_MAX_RETRY_ROUNDS", 1)
        monkeypatch.setattr(multipart_module, "UPLOAD_BLOB_MAX_ATTEMPTS", 3)
        monkeypatch.setattr(multipart_module.random, "random", lambda: 0.0)
        original_manifest = multipart_module.HashManifest

        def manifest(*args, **kwargs):
            result = original_manifest(*args, **kwargs)
            self.manifests.append(result)
            return result

        monkeypatch.setattr(multipart_module, "HashManifest", manifest)

        def send(adapter, request, **kwargs):
            return self.send(adapter, request, **kwargs)

        monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)

        def forbidden(*args, **kwargs):
            raise AssertionError("Unexpected real network access")

        monkeypatch.setattr(socket.socket, "connect", forbidden)
        monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
        monkeypatch.setattr(socket, "create_connection", forbidden)
        config = HubConfig(
            endpoint="https://example.test",
            token="offline-test-identity",
            config_dir=tmp_path / "sdk",
            cache_dir=tmp_path / "cache",
        )
        self.api = HubApi(config=config, user_agent="invoked_by/cli invoked_by/compat")

    def add(self, payload, *, uploaded=(), initialized=True, done=False):
        sha = hashlib.sha256(payload).hexdigest()
        self.states[sha] = {
            "payload": payload,
            "uploaded": set(uploaded),
            "part_size": PART,
            "token": "opaque-token",
            "initialized": initialized,
            "done": done,
        }
        return sha

    def plan(self, sha):
        state = self.states[sha]
        size, p = len(state["payload"]), state["part_size"]
        total = (size + p - 1) // p
        uploaded = [{"partNumber": n, "size": min(p, size - (n - 1) * p)} for n in sorted(state["uploaded"])]
        return {
            "resumableMultipart": True,
            "state": "UPLOADING",
            "sha256": sha,
            "size": size,
            "uploadToken": state["token"],
            "partSize": p,
            "totalParts": total,
            "maxParallelism": 2,
            "uploadedParts": uploaded,
            "uploadedBytes": sum(x["size"] for x in uploaded),
        }

    def send(self, adapter, request, **kwargs):
        parsed = urlsplit(request.url)
        assert parsed.hostname == "example.test"
        query = parse_qs(parsed.query)
        action = query.get("action", [""])[0]
        number = int(query["part_number"][0]) if "part_number" in query else None
        with self.lock:
            self.requests.append((action, number, request, kwargs))
            if parsed.path.endswith("/info/lfs/objects/batch"):
                body = json.loads(request.body)
                objects = []
                for item in body["objects"]:
                    state = self.states[item["oid"]]
                    obj = {"oid": item["oid"], "size": item["size"], "actions": {}}
                    if not state["done"]:
                        obj["actions"]["upload"] = {
                            "href": f"https://example.test{parsed.path.split('/info/')[0]}/blobs/{item['oid']}/{item['size']}"
                        }
                    objects.append(obj)
                return response(request, {"objects": objects})
            if "/commit/" in parsed.path:
                if self.hook:
                    override = self.hook(request, "commit", None, None)
                    if override is not None:
                        return override
                self.commits.extend(json.loads(request.body)["actions"])
                return response(request, {"commit_id": "offline-commit"})
            if parsed.path.endswith(("/models", "/datasets")) and request.method == "POST":
                return response(request, {"id": "owner/repo"})
            if "/blobs/" not in parsed.path:
                raise AssertionError(f"Unmatched offline request path: {parsed.path}")
            sha = parsed.path.split("/")[-2]
            state = self.states[sha]
            body = request.body
            payload = b""
            if hasattr(body, "read"):
                payload = b"".join(iter(lambda: body.read(65536), b""))
            elif body:
                payload = body if isinstance(body, bytes) else body.encode()
            assert request.headers["User-Agent"].startswith("modelscope_hub/")
            assert "invoked_by/cli" in request.headers["User-Agent"]
            assert "invoked_by/compat" in request.headers["User-Agent"]
            assert request.headers["Cookie"] == "m_session_id=offline-test-identity"
            assert request.headers["Authorization"] == "Bearer offline-test-identity"
            assert "Transfer-Encoding" not in request.headers
            assert int(request.headers["Content-Length"]) == len(payload)
            if action:
                retry = adapter.max_retries
                assert all(
                    getattr(retry, field) == 0 for field in ("total", "connect", "read", "status", "redirect", "other")
                )
            if action == "part":
                p = state["part_size"]
                offset = (number - 1) * p
                assert payload == state["payload"][offset : offset + p]
                assert request.headers["X-Part-SHA256"] == hashlib.sha256(payload).hexdigest()
                assert resumed_digest(request.headers["X-Oss-Hash-Ctx"], state["payload"][offset:]) == bytes.fromhex(
                    sha
                )
                assert request.headers["X-Upload-Token"] == state["token"]
                self.parts.append((sha, number, payload))
            elif action:
                assert payload == b"" and "X-Part-SHA256" not in request.headers
                assert "X-Oss-Hash-Ctx" not in request.headers
            if self.hook:
                override = self.hook(request, action, number, sha)
                if override is not None:
                    return override
            if action == "status":
                assert "part_size" not in query
                if state["done"]:
                    return response(request, {})
                if not state["initialized"]:
                    return fault(request, "UPLOAD_NOT_FOUND", action, status=404)
            elif action == "init":
                assert "X-Upload-Token" not in request.headers
                state["initialized"] = True
                state["uploaded"].clear()
            elif action == "part":
                state["uploaded"].add(number)
            elif action == "complete":
                assert state["uploaded"] == set(range(1, self.plan(sha)["totalParts"] + 1))
                state["done"] = True
                return response(request, {})
            elif not action:
                assert payload == state["payload"]
                state["done"] = True
                return response(request, {})
            else:
                raise AssertionError(f"Unexpected multipart action: {action}")
            return response(request, self.plan(sha))

    def uploader(self, sha, *, cache=False):
        settings = replace(MultipartSettings.load(self.api.legacy.endpoint), part_workers=1)
        execution = UploadExecution(settings, self.api.legacy, 1, cache)
        value = MultipartUploader(
            execution,
            home=self.api._config.config_dir,
            repo_id="owner/repo",
            repo_type="model",
            sha256=sha,
            size=len(self.states[sha]["payload"]),
            data=self.states[sha]["payload"],
        )
        url = f"https://example.test/api/v1/repos/models/owner/repo/blobs/{sha}/{value.size}"
        return execution, value, url


@pytest.fixture
def mirror(monkeypatch, tmp_path):
    value = Mirror(monkeypatch, tmp_path)
    yield value
    if value.api._legacy is not None:
        value.api._legacy._session.close()
        value.api._legacy._read_session.close()
    if value.api._openapi is not None:
        value.api._openapi._session.close()


def payload(tag=b"x"):
    return tag * (2 * PART) + b"odd-tail!"


@pytest.mark.parametrize("folder", [False, True])
@pytest.mark.parametrize("repo_type", ["model", "dataset"])
def test_facade_real_chain_and_verified_sparse_payload(mirror, tmp_path, folder, repo_type):
    data = payload()
    sha = mirror.add(data, uploaded={1, 2})
    if folder:
        path = tmp_path / "inputs"
        path.mkdir()
        (path / "weights.bin").write_bytes(data)
        mirror.api.upload_folder("owner/repo", repo_type, path, use_cache=False, max_workers=1, disable_tqdm=True)
    else:
        mirror.api.upload_file("owner/repo", repo_type, data, "weights.bin", disable_tqdm=True)
    assert [(oid, n) for oid, n, _ in mirror.parts] == [(sha, 3)]
    assert [m.coverage for m in mirror.manifests] == [(3,)]
    assert mirror.commits[0]["sha256"] == sha


@pytest.mark.parametrize("name", ["PART_INTEGRITY", "HASH_CONTEXT_MISMATCH"])
def test_dirty_repair_and_coverage_expansion(mirror, name):
    data = payload()
    sha = mirror.add(data, uploaded={1, 2})
    seen = []

    def hook(request, action, number, oid):
        trigger = action == "part" if name == "PART_INTEGRITY" else action == "complete"
        if trigger and not seen:
            seen.append(1)
            return fault(request, name, action, number if action == "part" else None)
        return None

    mirror.hook = hook
    mirror.api.upload_file("owner/repo", "model", data, "weights.bin", disable_tqdm=True)
    assert mirror.states[sha]["done"]
    counts = Counter(n for _, n, _ in mirror.parts)
    assert counts == (Counter({3: 2}) if name == "PART_INTEGRITY" else Counter({3: 2, 1: 1, 2: 1}))
    assert [m.coverage for m in mirror.manifests] == ([(3,)] if name == "PART_INTEGRITY" else [(3,), (1, 2, 3)])


def test_context_fault_then_refresh_preserves_dirty_and_budget(mirror):
    data = payload()
    sha = mirror.add(data, uploaded={1, 2})
    flags = set()

    def hook(request, action, number, oid):
        if action == "complete" and "fault" not in flags:
            flags.add("fault")
            return fault(request, "HASH_CONTEXT_MISMATCH", action)
        if action == "status" and "fault" in flags and "refresh" not in flags:
            flags.add("refresh")
            mirror.states[sha]["token"] = "refreshed-token"
            return fault(request, "TOKEN_REFRESH_REQUIRED", action, status=409)
        return None

    mirror.hook = hook
    execution, uploader, url = mirror.uploader(sha)
    with execution:
        uploader.run(url)
    assert uploader._token_refreshes == 1 and uploader.state.generation == 0
    assert uploader.state.dirty == [] and uploader._rounds == 2
    assert [m.coverage for m in mirror.manifests] == [(3,), (1, 2, 3)]
    assert not any(a == "init" for a, *_ in mirror.requests)
    assert any(a == "status" and "X-Upload-Token" not in r.headers for a, _, r, _ in mirror.requests)


def test_part_size_change_maps_dirty_and_discards_old_manifest(mirror):
    data = payload()
    sha = mirror.add(data, uploaded={1, 2})
    flags = set()

    def hook(request, action, number, oid):
        if action == "complete" and not flags:
            flags.add("fault")
            return fault(request, "HASH_CONTEXT_MISMATCH", action)
        if action == "status" and flags == {"fault"}:
            flags.add("changed")
            mirror.states[sha].update(part_size=2 * PART, uploaded={1}, token="new-plan-token")
        return None

    mirror.hook = hook
    execution, uploader, url = mirror.uploader(sha)
    with execution:
        uploader.run(url)
    assert [(m.part_size, m.coverage) for m in mirror.manifests] == [(PART, (3,)), (2 * PART, (1, 2))]
    assert uploader.state.dirty == [] and mirror.states[sha]["done"]


@pytest.mark.parametrize("name", ["FILE_SHA256_MISMATCH", "UPLOAD_SESSION_GONE", "COMPLETION_UNVERIFIED"])
def test_reinitialize_verifies_source_and_rescans(mirror, name):
    data = payload()
    sha = mirror.add(data, uploaded={1, 2})
    flags = set()

    def hook(request, action, number, oid):
        if action == "complete" and not flags:
            flags.add(1)
            return fault(request, name, action)
        return None

    mirror.hook = hook
    execution, uploader, url = mirror.uploader(sha)
    with execution:
        uploader.run(url)
    assert uploader.state.generation == 1 and uploader._initializations == 1
    assert [m.coverage for m in mirror.manifests] == [(3,), (1, 2, 3)]


@pytest.mark.parametrize("stage", ["init", "part", "complete"])
def test_unknown_transport_result_is_reconciled_not_blindly_replayed(mirror, stage):
    data = payload()
    sha = mirror.add(data, uploaded={1, 2}, initialized=stage != "init")
    flags = set()

    def hook(request, action, number, oid):
        if action == stage and not flags:
            flags.add(1)
            state = mirror.states[sha]
            if stage == "init":
                state["initialized"] = True
                state["uploaded"].clear()
            elif stage == "part":
                state["uploaded"].add(number)
            else:
                state["done"] = True
            raise requests.ConnectionError("Offline lost response")
        return None

    mirror.hook = hook
    execution, uploader, url = mirror.uploader(sha)
    with execution:
        result = uploader.run(url)
    assert result.ready
    actions = [a for a, *_ in mirror.requests]
    assert actions.count(stage) == 1
    if stage == "init":
        assert any(a == "status" and "X-Upload-Token" not in r.headers for a, _, r, _ in mirror.requests[2:])


def test_completion_only_021_has_no_scan_or_new_init(mirror, monkeypatch):
    sha = mirror.add(payload(), uploaded={1, 2, 3})
    execution, uploader, url = mirror.uploader(sha, cache=True)
    with uploader.store.lease(url, execution.check):
        uploader.store.save(UploadState(url, "opaque-token", PART, last_action="complete"))
    flags = set()

    def hook(request, action, number, oid):
        if action == "complete" and not flags:
            flags.add(1)
            return fault(request, "SOURCE_CLEANUP_FAILED", action)
        return None

    mirror.hook = hook
    monkeypatch.setattr(execution, "sleep", lambda seconds: None)
    with execution:
        assert uploader.run(None).ready
    assert not mirror.manifests and not mirror.parts
    assert [a for a, *_ in mirror.requests] == ["status", "complete", "complete"]


def test_existing_blob_without_pending_has_no_multipart_requests(mirror):
    data = payload()
    mirror.add(data, done=True)
    mirror.api.upload_file("owner/repo", "model", data, "weights.bin", disable_tqdm=True)
    assert not mirror.parts and not mirror.manifests
    assert not any(a in {"status", "init", "complete"} for a, *_ in mirror.requests)


@pytest.mark.parametrize(
    "name,status",
    [("AUTHENTICATION_REQUIRED", 401), ("REPOSITORY_UNAVAILABLE", 403), (None, 404), ("FEATURE_UNAVAILABLE", 400)],
)
def test_terminal_status_never_initializes(mirror, name, status):
    sha = mirror.add(payload())
    mirror.hook = lambda request, action, number, oid: (
        fault(request, name, action, status=status)
        if name
        else response(request, {}, status=status, code=status, success=False)
    )
    execution, uploader, url = mirror.uploader(sha)
    with execution, pytest.raises(MultipartError):
        uploader.run(url)
    assert not mirror.manifests and not mirror.parts and not mirror.commits
    assert [a for a, *_ in mirror.requests] == ["status"]


def test_manifest_subset_reuse_union_rebuild_and_atomic_failure(mirror):
    sha = mirror.add(payload())
    execution, uploader, url = mirror.uploader(sha)
    with execution:
        uploader.source = UploadSource(uploader._data, uploader.size, sha, PART)
        uploader._adopt(MultipartPlan.parse(mirror.plan(sha), sha, uploader.size))
        first = uploader._manifest_for({2})
        assert uploader._manifest_for({2}) is first
        uploader._adopt(replace(uploader._plan, token="refreshed"))
        assert uploader._manifest_for({2}) is first
        second = uploader._manifest_for({1})
        assert second.coverage == (1, 2) and second is not first
        uploader.source.sha256 = "0" * 64
        with pytest.raises(MultipartError):
            uploader._manifest_for({3})
        assert uploader.manifest is second


def test_source_io_is_terminal_without_part_or_outer_retry(mirror, monkeypatch):
    data = payload()
    sha = mirror.add(data)
    original = RangeReader.read

    def read(stream, size=-1):
        raise OSError(errno.EIO, "offline disk read failure")

    monkeypatch.setattr(RangeReader, "read", read)
    execution, uploader, url = mirror.uploader(sha)
    with execution, pytest.raises(MultipartError) as caught:
        uploader.run(url)
    assert caught.value.recovery_handled and not caught.value.retryable
    assert not mirror.parts
    monkeypatch.setattr(RangeReader, "read", original)


def test_corrupt_prefix_cannot_send_part(mirror, monkeypatch):
    data = payload()
    mirror.add(data, uploaded={1})

    class Corrupt(source_module.SHA256State):
        def export_state(self):
            self._h[0] ^= 1
            return super().export_state()

    monkeypatch.setattr(source_module, "SHA256State", Corrupt)
    with pytest.raises(MultipartError):
        mirror.api.upload_file("owner/repo", "model", data, "weights.bin", disable_tqdm=True)
    assert not mirror.parts and not mirror.commits


def test_off_and_metadata_keep_existing_request_paths(mirror, monkeypatch):
    data = payload()
    mirror.add(data)
    monkeypatch.setattr(constants, "UPLOAD_MULTIPART_MODE", "off")
    mirror.api.upload_file("owner/repo", "model", data, "weights.bin", disable_tqdm=True)
    mirror.api.upload_file("owner/repo", "model", b"metadata", "README.md", disable_tqdm=True)
    assert not mirror.manifests and not mirror.parts
    assert not any(a in {"status", "init", "complete"} for a, *_ in mirror.requests)
    assert [op["type"] for op in mirror.commits] == ["lfs", "normal"]


def test_replica_errors_follow_their_own_owner_and_suppress_sync(mirror, tmp_path, monkeypatch):
    root = tmp_path / "inputs"
    root.mkdir()
    first, second = payload(b"a"), payload(b"b")
    first_sha = mirror.add(first)
    second_sha = mirror.add(second)
    for name, data in (("a.bin", first), ("b.bin", second), ("dup-a.bin", first), ("dup-b.bin", second)):
        (root / name).write_bytes(data)
    (root / "README.md").write_text("metadata")
    failures = {}
    original = mirror.api.uploader._retry_failed_files_react

    def observe(*args, **kwargs):
        rows = kwargs.get("failed_files", args[0] if args else [])
        failures.update({path: error.name for (path, _), error in rows})
        return original(*args, **kwargs)

    monkeypatch.setattr(mirror.api.uploader, "_retry_failed_files_react", observe)
    mirror.hook = lambda request, action, number, oid: (
        fault(request, "FEATURE_UNAVAILABLE" if oid == first_sha else "AUTHENTICATION_REQUIRED", action, status=400)
        if action == "status"
        else None
    )
    with pytest.raises(StorageError, match="4 file"):
        mirror.api.upload_folder(
            "owner/repo", "model", root, use_cache=False, max_workers=1, sync_remote_repo=True, disable_tqdm=True
        )
    assert failures["dup-a.bin"] == failures["a.bin"] == "FEATURE_UNAVAILABLE"
    assert failures["dup-b.bin"] == failures["b.bin"] == "AUTHENTICATION_REQUIRED"
    assert {op["path"] for op in mirror.commits} == {"README.md"}
    assert Counter(r.url.split("/")[-2] for a, _, r, _ in mirror.requests if a == "status") == {
        first_sha: 1,
        second_sha: 1,
    }


def test_cancelled_file_future_does_not_trap_batch_consumer(mirror):
    settings = MultipartSettings.load(mirror.api.legacy.endpoint)
    with UploadExecution(settings, mirror.api.legacy, 1, False) as execution:
        tracker = BatchTracker(1, [1])
        queued = Future()
        execution.register(queued)
        execution.client.token = "changed-identity"
        errors = []

        def wait():
            try:
                tracker.wait_for_batch(0, check=execution.check)
            except MultipartError as error:
                errors.append(error)

        consumer = threading.Thread(target=wait, daemon=True)
        consumer.start()
        consumer.join(2)
        assert not consumer.is_alive() and errors
        assert queued.cancelled()


def test_interrupt_stops_before_executor_drain(mirror):
    settings = MultipartSettings.load(mirror.api.legacy.endpoint)
    execution = UploadExecution(settings, mirror.api.legacy, 1, False)
    entered, release = threading.Event(), threading.Event()
    outcomes = []
    original_stop = execution.stop

    def stop():
        original_stop()
        release.set()

    execution.stop = stop

    def in_flight():
        entered.set()
        assert release.wait(2)

    with pytest.raises(KeyboardInterrupt):
        with ThreadPoolExecutor(max_workers=1) as pool, _stop_on_error(execution):
            execution.register(pool.submit(in_flight))
            assert entered.wait(2)
            execution.register(pool.submit(lambda: outcomes.append("unexpected queued work")))
            raise KeyboardInterrupt
    assert execution._cancelled.is_set() and not outcomes
    execution.close()


def test_cancelled_in_flight_init_persists_returned_token(mirror):
    sha = mirror.add(payload(), initialized=False)
    execution, uploader, url = mirror.uploader(sha, cache=True)

    def hook(request, action, number, oid):
        if action == "init":
            mirror.states[sha]["initialized"] = True
            execution.stop()
            return response(request, mirror.plan(sha))
        return None

    mirror.hook = hook
    with execution, pytest.raises(MultipartError, match="cancelled"):
        uploader.run(url)
    saved = uploader.store._read(uploader.store._path)
    assert saved.token == "opaque-token" and saved.generation == 1
    assert not saved.ready and not mirror.parts


def test_source_constructor_io_is_inside_terminal_boundary(mirror, tmp_path):
    sha = mirror.add(payload())
    execution, uploader, url = mirror.uploader(sha)
    uploader._data = tmp_path / "missing-source.bin"
    with execution, pytest.raises(MultipartError) as caught:
        uploader.run(url)
    assert caught.value.recovery_handled and isinstance(caught.value.__cause__, FileNotFoundError)
    assert not mirror.requests


def test_three_failed_commits_stop_and_preserve_tracker(mirror, tmp_path, monkeypatch):
    root = tmp_path / "inputs"
    root.mkdir()
    for number in range(4):
        data = bytes([number]) * PART
        mirror.add(data)
        (root / f"weights-{number}.bin").write_bytes(data)
    monkeypatch.setattr(upload_module, "UPLOAD_COMMIT_BATCH_MAX_OPERATIONS", 1)
    monkeypatch.setattr(upload_module.time, "sleep", lambda seconds: None)
    attempts = []

    def hook(request, action, number, oid):
        if action == "commit":
            attempts.append(1)
            return response(request, {}, status=503, code=503, success=False)
        return None

    mirror.hook = hook
    tracker = tmp_path / "tracker.json"
    with pytest.raises(RuntimeError, match="3 consecutive"):
        mirror.api.upload_folder(
            "owner/repo", "model", root, use_cache=True, tracker_path=tracker, max_workers=1, disable_tqdm=True
        )
    per_batch = upload_module.UploadManager._commit_with_retry.__kwdefaults__["max_attempts"]
    assert len(attempts) == 3 * per_batch and tracker.exists() and not mirror.commits
    assert "COMMITTED" not in tracker.read_text()


def test_http_200_false_success_cannot_complete_or_commit(mirror):
    sha = mirror.add(payload())
    mirror.hook = lambda request, action, number, oid: response(request, {}, success=False)
    execution, uploader, url = mirror.uploader(sha)
    with execution, pytest.raises(MultipartError):
        uploader.run(url)
    assert not mirror.parts and not mirror.commits


def test_scan_publication_precedes_any_part(mirror):
    data = payload()
    mirror.add(data, uploaded={1, 2})

    def hook(request, action, number, oid):
        if action == "part":
            # Mirror records a manifest only after its complete verified constructor returns.
            assert len(mirror.manifests) == 1
        return None

    mirror.hook = hook
    mirror.api.upload_file("owner/repo", "model", data, "weights.bin", disable_tqdm=True)


def test_all_local_modules_and_upload_shims_import_offline(mirror, monkeypatch):
    root = Path(source_module.__file__).parent
    imported, optional_missing = [], []
    for path in root.rglob("*.py"):
        name = "modelscope_hub." + ".".join(path.relative_to(root).with_suffix("").parts)
        name = name.removesuffix(".__init__")
        try:
            importlib.import_module(name)
            imported.append(name)
        except ModuleNotFoundError as error:
            if error.name in {"kernels", "huggingface_hub"}:
                optional_missing.append((name, error.name))
                continue
            raise
    shim_root = root.parents[2] / "modelscope"
    monkeypatch.syspath_prepend(str(shim_root))
    for name in ("modelscope.hub.api", "modelscope.hub.snapshot_download", "modelscope.hub.file_download"):
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(shim_root)
    print(f"Imported {len(imported)} local modules and 3 local shims; optional missing: {optional_missing}")
