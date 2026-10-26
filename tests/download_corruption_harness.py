"""Deterministic HTTP corruption injection using the unmodified SDK download pipeline."""

from __future__ import annotations

import contextlib
import hashlib
import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from modelscope_hub import DownloadRetryPolicy, HubApi
from modelscope_hub._download import ProgressCallback
from modelscope_hub._download_state import safe_error

GROUPS = {
    1: "persistent_empty_200",
    2: "persistent_short_object",
    3: "persistent_short_chunked",
    4: "persistent_bit_flips",
    5: "persistent_wrong_object",
    6: "persistent_overlong_chunked",
    7: "transient_empty_200",
    8: "transient_hash_mismatch",
    9: "truncated_connection_then_resume",
    10: "wrong_range_start_then_restart",
}


@dataclass(frozen=True)
class FaultCase:
    """One unique repository file, never an alias for another corruption case."""

    path: str
    content: bytes
    group: int = 0
    lfs: bool = True

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()

    @property
    def cut(self) -> int:
        return len(self.content) // 3 + 13

    @property
    def metadata(self) -> dict:
        return {"Path": self.path, "Type": "blob", "Size": len(self.content), "Sha256": self.sha256}


def synthetic_cases() -> list[FaultCase]:
    """Provide ten distinct files per group plus one hundred normal controls."""
    cases = []
    for index in range(100):
        group = index // 10 + 1
        path = f"group-{group:02}/file-{index:03}.bin"
        content = hashlib.shake_256(path.encode()).digest(32768 + index * 17)
        cases.append(FaultCase(path, content, group, lfs=group >= 9 or index % 2 == 0))
        path = f"controls/normal-{index:03}.bin"
        content = hashlib.shake_256(path.encode()).digest(32768 + index * 19)
        cases.append(FaultCase(path, content, lfs=index % 2 == 0))
    return cases


class FaultHTTPServer:
    """Serve valid HTTP framing, truncated sockets, redirects and incorrect ranges."""

    def __init__(self, cases: list[FaultCase]) -> None:
        self.cases = {case.path: case for case in cases}
        self.lock = threading.Lock()
        self.attempts: dict[tuple[str, str], int] = {}
        self.wire: list[dict] = []
        self.credential_requests = 0
        state = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                if self.headers.get("Authorization") or self.headers.get("Cookie"):
                    with state.lock:
                        state.credential_requests += 1
                    self._send(403, b"Credentials must not be sent to the local fault server")
                    return
                url = urlparse(self.path)
                path = (
                    unquote(url.path.removeprefix("/cdn/"))
                    if url.path.startswith("/cdn/")
                    else parse_qs(url.query).get("FilePath", [""])[0]
                )
                case = state.cases.get(path)
                if case is None:
                    self._send(404, b"Unknown test file")
                    return
                stage = self.headers.get("X-Test-Stage", "first")
                with state.lock:
                    state.wire.append({"stage": stage, "path": path, "cdn": url.path.startswith("/cdn/")})
                if case.lfs and not url.path.startswith("/cdn/"):
                    self._send(302, b"", {"Location": f"/cdn/{quote(path)}"})
                    return
                with state.lock:
                    key = (stage, path)
                    attempt = state.attempts.get(key, 0) + 1
                    state.attempts[key] = attempt
                inject = self.headers.get("X-Test-Inject") == "1"
                active = inject and (case.group <= 6 or attempt == 1)
                body = case.content
                group = case.group if active else 0
                if group in (1, 7):
                    self._send(200, b"")
                elif group == 2:
                    self._send(200, body[: case.cut])
                elif group == 3:
                    self._chunked(body[: case.cut])
                elif group in (4, 8):
                    damaged = bytearray(body)
                    for index in (0, len(body) // 2, len(body) - 1):
                        damaged[index] ^= 1
                    self._send(200, bytes(damaged))
                elif group == 5:
                    self._send(200, bytes(byte ^ 0xA5 for byte in body))
                elif group == 6:
                    self._chunked(body + b"EXCESS-CONTENT" * 128)
                elif group == 9:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body[: case.cut])
                    self.wfile.flush()
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_WR)
                elif group == 10:
                    requested = self.headers.get("Range", "bytes=0-")
                    start = int(requested.removeprefix("bytes=").partition("-")[0]) + 1
                    self._send(206, body[start:], {"Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}"})
                else:
                    requested = self.headers.get("Range")
                    if requested and case.lfs:
                        start_text, _, end_text = requested.removeprefix("bytes=").partition("-")
                        start = int(start_text)
                        end = min(int(end_text), len(body) - 1) if end_text else len(body) - 1
                        if start >= len(body):
                            self._send(416, b"", {"Content-Range": f"bytes */{len(body)}"})
                        else:
                            self._send(
                                206, body[start : end + 1], {"Content-Range": f"bytes {start}-{end}/{len(body)}"}
                            )
                    elif case.lfs:
                        self._send(200, body, {"X-Linked-ETag": case.sha256})
                    else:
                        self._chunked(body)

            def _send(self, status: int, body: bytes, headers: dict | None = None) -> None:
                self.send_response(status)
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _chunked(self, body: bytes) -> None:
                self.send_response(200)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for start in range(0, len(body), 4096):
                    chunk = body[start : start + 4096]
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")

        class QuietServer(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address) -> None:
                # Integrity rejection legitimately closes an unread response body.
                pass

        self.server = QuietServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)

    def __enter__(self) -> FaultHTTPServer:
        self.thread.start()
        return self

    def __exit__(self, *args) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class FaultRouter:
    """Inject at the Response boundary, preserving all SDK orchestration and checks."""

    def __init__(self, server: FaultHTTPServer, *, live: bool = False) -> None:
        self.server = server
        self.live = live
        self.stage = "first"
        self.inject = True
        self.records: list[dict] = []
        self.lock = threading.Lock()
        self.local_api = HubApi(endpoint=server.url, token="")
        self.local_api.legacy._read_session.trust_env = False

    def attach(self, api: HubApi) -> None:
        original = api.legacy.download_stream

        def stream(repo_id, repo_type, file_path, revision="master", headers=None):
            case = self.server.cases[file_path]
            use_local = not self.live or self.inject and case.group != 0
            record = {
                "stage": self.stage,
                "path": file_path,
                "range": (headers or {}).get("Range"),
                "transport": "fault_http" if use_local else "modelscope.cn",
            }
            with self.lock:
                self.records.append(record)
            started = time.monotonic()
            try:
                if use_local:
                    control = api.legacy._read_state.control
                    local_headers = {
                        **(headers or {}),
                        "X-Test-Stage": self.stage,
                        "X-Test-Inject": "1" if self.inject and case.group else "0",
                    }
                    # Never forward auth from the real API facade to localhost.
                    local_headers = {
                        key: value
                        for key, value in local_headers.items()
                        if key.lower() not in ("authorization", "cookie")
                    }
                    with self.local_api.legacy.read_context(control):
                        response = self.local_api.legacy.download_stream(
                            repo_id, repo_type, file_path, revision, headers=local_headers
                        )
                else:
                    response = original(repo_id, repo_type, file_path, revision, headers=headers)
                record.update(
                    status=response.status_code,
                    redirect_hops=len(response.history),
                    content_length=response.headers.get("Content-Length"),
                    content_range=response.headers.get("Content-Range"),
                )
                return response
            except Exception as exc:
                record["error"] = safe_error(exc)
                raise
            finally:
                record["header_seconds"] = round(time.monotonic() - started, 4)

        api.legacy.download_stream = stream

    def for_stage(self, stage: str) -> list[dict]:
        return [record for record in self.records if record["stage"] == stage]

    def close(self) -> None:
        self.local_api.legacy._read_session.close()
        self.local_api.legacy._session.close()


def verify_file(target: Path, metadata: dict) -> None:
    assert target.is_file(), str(target)
    assert target.stat().st_size == int(metadata["Size"]), str(target)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == metadata["Sha256"].lower(), str(target)


def assert_single_json(root: Path, cases: list[FaultCase]) -> None:
    data_paths = {root / "files" / case.path for case in cases}
    generated = {path for path in root.rglob("*") if path.is_file() and path not in data_paths}
    assert {path for path in generated if path.suffix in (".json", ".jsonl")} == {root / ".ms_download_state.json"}
    assert not any(path.name.endswith((".tmp", ".incomplete", ".parallel_tmp")) for path in generated)


def exercise_stages(
    root: Path,
    cases: list[FaultCase],
    api_factory: Callable[[], HubApi],
    router: FaultRouter,
    *,
    fresh_manifest: Callable[[], list[dict]] | None = None,
    announce: Callable[[str], None] = lambda message: None,
) -> dict:
    """Exercise first failure, restart, no-reset recovery, explicit repair and reuse."""
    assert len(cases) == len({case.path for case in cases}) == 200
    assert sum(case.group > 0 for case in cases) == 100
    assert all(sum(case.group == group for case in cases) == 10 for group in GROUPS)
    output = root / "files"
    tracker = root / ".ms_download_state.json"
    assert not tracker.exists() and not output.exists(), "The corruption experiment needs a fresh isolated directory"
    persistent = {case.path for case in cases if 1 <= case.group <= 6}
    metadata = [case.metadata for case in cases]
    for case in cases:
        if case.group == 10:
            target = output / case.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.with_suffix(target.suffix + ".incomplete").write_bytes(case.content[: case.cut])
    policy = DownloadRetryPolicy()
    evidence: dict = {"policy": asdict(policy), "fault_files": 100, "control_files": 100, "stages": {}, "cases": []}
    report = None
    try:
        for stage in ("first", "restart", "source_repaired_without_reset", "explicit_repair", "reuse"):
            router.stage = stage
            router.inject = stage in ("first", "restart")
            api = api_factory()
            router.attach(api)
            started = time.monotonic()
            transfer_times: dict[str, list[float]] = {}
            timing_lock = threading.Lock()

            class CaseProgress(ProgressCallback):
                def __init__(self, filename: str, file_size: int):
                    super().__init__(filename, file_size)
                    with timing_lock:
                        transfer_times.setdefault(filename, [time.monotonic(), time.monotonic()])

                def end(self) -> None:
                    with timing_lock:
                        transfer_times[self.filename][1] = time.monotonic()

            try:
                report = api.download_repo_report(
                    "wangxingjun778/test_upload_large_2g",
                    "dataset",
                    local_dir=output,
                    tracker_path=tracker,
                    file_list=iter(metadata),
                    max_workers=4,
                    retry_policy=policy,
                    retry_failed=stage == "explicit_repair",
                    progress_callbacks=[CaseProgress],
                )
                rows = {row.path: row for row in report.iter_files()}
                requests = router.for_stage(stage)
                evidence["stages"][stage] = {
                    "summary": asdict(report),
                    "seconds": round(time.monotonic() - started, 4),
                    "logical_file_requests": len(requests),
                    "remote_file_requests": sum(request["transport"] == "modelscope.cn" for request in requests),
                    "redirect_hops": sum(request.get("redirect_hops", 0) for request in requests),
                }
                announce(
                    f"{stage}: {report.status}，成功 {report.succeeded}，失败 {report.failed}，请求 {len(requests)}"
                )
                assert report.total == 200 and len(rows) == 200
                assert report.global_error is None and report.listing_complete and not report.state_warnings
                if stage in ("first", "restart", "source_repaired_without_reset"):
                    assert (report.status, report.succeeded, report.failed, report.deferred, report.pending) == (
                        "PARTIAL",
                        140,
                        60,
                        0,
                        0,
                    ), evidence["stages"][stage]
                    assert {row.path for row in rows.values() if row.status == "FAILED"} == persistent
                    assert all(not (output / path).exists() for path in persistent)
                else:
                    assert report.complete and report.succeeded == 200
                    hints = json.loads(tracker.read_text(encoding="utf-8"))
                    assert hints["files"] == {} and hints["pages"] == {}
                if stage == "first":
                    assert len(requests) == 300
                    for case in cases:
                        selected = [request for request in requests if request["path"] == case.path]
                        row = rows[case.path]
                        assert len(selected) == (2 if case.group else 1), (case.path, selected)
                        if not case.group:
                            continue
                        assert row.retries == 1, (case.path, row)
                        if case.group == 9:
                            assert [request["range"] for request in selected] == [None, f"bytes={case.cut}-"]
                            assert selected[1]["status"] == 206
                        if case.group == 10:
                            assert [request["range"] for request in selected] == [f"bytes={case.cut}-", None]
                            assert [request["status"] for request in selected] == [206, 200]
                        if case.group <= 6:
                            expected_error = "FileIntegrityError" if case.group in (4, 5) else "FileSizeMismatchError"
                            assert row.error_type == expected_error, row
                        group_index = sum(item["group"] == case.group for item in evidence["cases"]) + 1
                        evidence["cases"].append(
                            {
                                "id": f"G{case.group}-{group_index:02}",
                                "group": case.group,
                                "fault": GROUPS[case.group],
                                "path": case.path,
                                "size": len(case.content),
                                "sha256": case.sha256,
                                "lfs": case.lfs,
                                "result": asdict(row),
                                "requests": selected,
                                "passed": True,
                                "seconds": round(transfer_times[case.path][1] - transfer_times[case.path][0], 4),
                            }
                        )
                    assert (output / cases[-1].path).is_file(), "The tail control file must complete"
                    evidence["tail_control_completed"] = cases[-1].path
                    for case in cases:
                        if case.path not in persistent:
                            verify_file(output / case.path, case.metadata)
                elif stage in ("restart", "source_repaired_without_reset"):
                    assert report.skipped_failures == 60 and requests == []
                elif stage == "explicit_repair":
                    assert len(requests) == 60 and {request["path"] for request in requests} == persistent
                    if fresh_manifest is not None:
                        current = {entry["Path"]: entry for entry in fresh_manifest() if entry.get("Type") != "tree"}
                        for entry in metadata:
                            remote = current[entry["Path"]]
                            assert (int(remote["Size"]), remote["Sha256"].lower()) == (entry["Size"], entry["Sha256"])
                    for entry in metadata:
                        verify_file(output / entry["Path"], entry)
                    evidence["independent_size_sha256_verified"] = 200
                    evidence["verified_bytes"] = sum(entry["Size"] for entry in metadata)
                elif stage == "reuse":
                    assert requests == []
                assert_single_json(root, cases)
                assert router.server.credential_requests == 0
            finally:
                if report is not None:
                    report.close()
                    report = None
                api.legacy._read_session.close()
                api.legacy._session.close()
        evidence["passed"] = True
        evidence["passed_fault_cases"] = len(evidence["cases"])
        evidence["group_results"] = {
            f"G{group}": {
                "fault": name,
                "cases": 10,
                "passed": 10,
                "first_status": "FAILED" if group <= 6 else "VERIFIED",
            }
            for group, name in GROUPS.items()
        }
        return evidence
    except BaseException as exc:
        evidence["passed"] = False
        evidence["error"] = safe_error(exc)
        evidence["observed_requests"] = router.records
        raise
    finally:
        # Keep both success and failure evidence in the task's only JSON file.
        if tracker.exists():
            from tempfile import NamedTemporaryFile

            data = json.loads(tracker.read_text(encoding="utf-8"))
            data["corruption_regression"] = evidence
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, suffix=".tmp", delete=False) as stream:
                json.dump(data, stream, ensure_ascii=False, separators=(",", ":"))
                temporary = Path(stream.name)
            temporary.replace(tracker)


@contextlib.contextmanager
def local_experiment(cases: list[FaultCase]) -> Iterator[tuple[FaultHTTPServer, FaultRouter]]:
    with FaultHTTPServer(cases) as server:
        router = FaultRouter(server)
        try:
            yield server, router
        finally:
            router.close()
