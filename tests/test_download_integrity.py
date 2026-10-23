"""Download integrity hardening: size validation, resume, cleanup and failure reporting.

A real HTTP server on 127.0.0.1 stands in for the hub and its LFS CDN, so the
tests exercise requests/urllib3 streaming, redirects, ``Range`` handling and
chunked bodies end to end. It mirrors what ModelScope does in production:
LFS files redirect to a CDN that honours ``Range`` (206 + ``Content-Range``,
416 past the end), while files served directly by the API ignore ``Range`` and
answer 200 with a chunked body.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import pytest

from modelscope_hub import _download
from modelscope_hub.api import HubApi
from modelscope_hub.cli.main import run_cmd
from modelscope_hub.errors import (
    FileDownloadError,
    FileIntegrityError,
    FileSizeMismatchError,
    NetworkError,
)

REPO = "owner/data"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class _RemoteFile:
    content: bytes
    # "lfs": 302 to a Range-capable CDN; "direct": 200 chunked, Range ignored.
    mode: str = "lfs"
    # Bytes actually served when they differ from what the listing describes.
    served: bytes | None = None
    # Served instead while the intra-cloud region header is present.
    served_with_region: bytes | None = None
    # The first body response stops after this many bytes and drops the socket.
    truncate_first: int | None = None
    header_delay: float = 0
    trickle_interval: float = 0
    header_trickle: bool = False
    status: int = 200
    retry_after: str | None = None

    def body(self, region: str | None) -> bytes:
        if region and self.served_with_region is not None:
            return self.served_with_region
        return self.content if self.served is None else self.served


@dataclass
class _Request:
    kind: str
    path: str
    range: str | None
    region: str | None


@dataclass
class _FakeHub:
    url: str = ""
    files: dict[str, _RemoteFile] = field(default_factory=dict)
    requests: list[_Request] = field(default_factory=list)
    truncated: set[str] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, path: str, content: bytes, **kwargs) -> bytes:
        self.files[path] = _RemoteFile(content, **kwargs)
        return content

    def seen(self, path: str, kind: str) -> list[_Request]:
        return [r for r in self.requests if r.path == path and r.kind == kind]


def _handler(hub: _FakeHub) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _record(self, kind: str, path: str) -> None:
            with hub.lock:
                hub.requests.append(
                    _Request(kind, path, self.headers.get("Range"), self.headers.get("x-aliyun-region-id"))
                )

        def _truncate_now(self, path: str) -> bool:
            with hub.lock:
                if hub.files[path].truncate_first is None or path in hub.truncated:
                    return False
                hub.truncated.add(path)
                return True

        def _send(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
            self.send_response(status)
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            url = urlparse(self.path)
            if url.path.endswith("/repo/tree"):
                entries = [
                    {"Path": p, "Type": "blob", "Size": len(f.content), "Sha256": _sha256(f.content)}
                    for p, f in sorted(hub.files.items())
                    if not parse_qs(url.query).get("Root")
                    or p.startswith(parse_qs(url.query)["Root"][0].rstrip("/") + "/")
                ]
                body = json.dumps({"Code": 200, "Data": {"Files": entries}}).encode()
                return self._send(200, body, {"Content-Type": "application/json"})
            if url.path.endswith("/repo"):
                path = parse_qs(url.query)["FilePath"][0]
                self._record("api", path)
                if hub.files[path].mode == "direct":
                    return self._serve_whole(path)
                self.send_response(302)
                self.send_header("Location", f"/cdn/{quote(path)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return None
            if url.path.startswith("/cdn/"):
                path = unquote(url.path[len("/cdn/") :])
                self._record("cdn", path)
                return self._serve_ranged(path)
            return self._send(404, b'{"Code": 404}', {"Content-Type": "application/json"})

        def _serve_whole(self, path: str) -> None:
            body = hub.files[path].body(self.headers.get("x-aliyun-region-id"))
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for start in range(0, len(body), 8192):
                piece = body[start : start + 8192]
                self.wfile.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")

        def _serve_ranged(self, path: str) -> None:
            remote = hub.files[path]
            if remote.header_delay:
                time.sleep(remote.header_delay)
            if remote.header_trickle:
                for byte in b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n":
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    time.sleep(0.05)
                return None
            if remote.status >= 400:
                return self._send(remote.status, b"{}", {"Retry-After": remote.retry_after or "0"})
            body = remote.body(self.headers.get("x-aliyun-region-id"))
            headers = {
                "Accept-Ranges": "bytes",
                "X-Cache": "HIT TCP_MEM_HIT",
                "EagleId": "test-eagle",
                "X-Linked-ETag": _sha256(remote.content),
            }
            status, payload = 200, body
            requested = self.headers.get("Range")
            if requested:
                first, _, last = requested.removeprefix("bytes=").partition("-")
                start = int(first)
                if start >= len(body):
                    return self._send(416, b"", {"Content-Range": f"bytes */{len(body)}"})
                end = min(int(last), len(body) - 1) if last else len(body) - 1
                status, payload = 206, body[start : end + 1]
                headers["Content-Range"] = f"bytes {start}-{end}/{len(body)}"
            if self._truncate_now(path):
                cut = hub.files[path].truncate_first or 0
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload[:cut])
                self.close_connection = True
                return None
            if remote.trickle_interval:
                self.send_response(status)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                for byte in payload:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    time.sleep(remote.trickle_interval)
                return None
            return self._send(status, payload, headers)

    return Handler


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        # Clients abandoning a body they rejected are expected here.
        pass


@pytest.fixture()
def hub(monkeypatch) -> Iterator[_FakeHub]:
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("MODELSCOPE_DOWNLOAD_INTRA_CLOUD", "false")
    monkeypatch.delenv("MODELSCOPE_DOWNLOAD_INTRA_CLOUD_REGION", raising=False)
    monkeypatch.delenv("MODELSCOPE_DOWNLOAD_INTER_CLOUD_REGIONS", raising=False)
    monkeypatch.setattr(_download, "DOWNLOAD_INTEGRITY_RETRY_BACKOFF", 0)
    fake = _FakeHub()
    server = _QuietServer(("127.0.0.1", 0), _handler(fake))
    fake.url = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield fake
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture()
def api(hub) -> HubApi:
    return HubApi(endpoint=hub.url, token="")


@pytest.fixture()
def dirs(tmp_path) -> tuple[Path, Path]:
    return tmp_path / "out", tmp_path / "cache"


class TestSnapshotFailureReporting:
    def test_empty_body_for_listed_file_is_reported_and_leaves_nothing(self, hub, api, dirs):
        out, cache = dirs
        good = hub.add("shards/a.tar", os.urandom(200_000))
        hub.add("shards/b.tar", os.urandom(200_000), served=b"")
        hub.add("shards/c.tar", os.urandom(150_000))

        with pytest.raises(FileDownloadError) as excinfo:
            api.download_repo(REPO, "dataset", local_dir=out, cache_dir=cache)

        error = excinfo.value
        assert list(error.failed_files) == ["shards/b.tar"]
        assert error.total == 3
        assert Path(error.output_dir) == out
        assert error.retryable is False
        assert "1 of 3 file(s) failed" in str(error)
        assert "  - shards/b.tar: [E2020] Server announced 0 bytes" in str(error)

        cause = error.failed_files["shards/b.tar"]
        assert isinstance(cause, FileSizeMismatchError)
        assert "repository lists 200000 bytes" in str(cause)
        assert "Content-Length=0" in str(cause)
        assert "host=127.0.0.1" in str(cause)
        assert "eagleid=test-eagle" in str(cause)

        assert (out / "shards/a.tar").read_bytes() == good
        assert (out / "shards/c.tar").stat().st_size == 150_000
        assert not (out / "shards/b.tar").exists()
        assert not (out / "shards/b.tar.incomplete").exists()
        # One retry over the default route instead of three full re-downloads.
        assert len(hub.seen("shards/b.tar", "cdn")) == 2

    def test_complete_snapshot_returns_output_dir(self, hub, api, dirs):
        out, cache = dirs
        files = {f"part-{i}.bin": hub.add(f"part-{i}.bin", os.urandom(30_000 + i)) for i in range(4)}
        files["README.md"] = hub.add("README.md", b"# data\n", mode="direct")

        assert api.download_repo(REPO, "dataset", local_dir=out, cache_dir=cache) == out
        for path, content in files.items():
            assert (out / path).read_bytes() == content

    def test_cli_exits_non_zero_when_a_file_fails(self, hub, dirs, monkeypatch):
        out, cache = dirs
        monkeypatch.setenv("MODELSCOPE_API_TOKEN", "")
        hub.add("a.bin", os.urandom(10_000))
        hub.add("b.bin", os.urandom(10_000), served=b"")

        argv = ["download", REPO, "--repo-type", "dataset", "--local-dir", str(out), "--cache-dir", str(cache)]
        assert run_cmd(["--endpoint", hub.url, *argv]) == 1
        assert (out / "a.bin").exists()
        assert not (out / "b.bin").exists()


class TestSizeValidation:
    def test_short_chunked_body_is_rejected(self, hub, api, dirs):
        out, cache = dirs
        hub.add("notes.txt", b"x" * 5000, mode="direct", served=b"x" * 100)

        with pytest.raises(FileSizeMismatchError, match="returned 100 of 5000 bytes"):
            api.download_file(REPO, "dataset", "notes.txt", local_dir=out, cache_dir=cache, file_size=5000)

        assert not (out / "notes.txt").exists()
        assert not (out / "notes.txt.incomplete").exists()

    def test_zero_byte_leftover_is_not_a_cache_hit(self, hub, api, dirs):
        out, cache = dirs
        content = hub.add("z.bin", os.urandom(4096))
        (out).mkdir(parents=True)
        (out / "z.bin").write_bytes(b"")

        path = api.download_file(REPO, "dataset", "z.bin", local_dir=out, cache_dir=cache, file_size=len(content))

        assert path.read_bytes() == content

    def test_size_mismatch_retries_once_without_region_routing(self, hub, api, dirs, monkeypatch):
        out, cache = dirs
        monkeypatch.setenv("MODELSCOPE_DOWNLOAD_INTRA_CLOUD", "true")
        monkeypatch.setenv("MODELSCOPE_DOWNLOAD_INTRA_CLOUD_REGION", "cn-test")
        content = hub.add("r.bin", os.urandom(8192), served_with_region=b"")

        path = api.download_file(
            REPO,
            "dataset",
            "r.bin",
            local_dir=out,
            cache_dir=cache,
            file_size=len(content),
            expected_sha256=_sha256(content),
        )

        assert path.read_bytes() == content
        assert [r.region for r in hub.seen("r.bin", "api")] == ["cn-test", None]

    def test_forced_refresh_keeps_previous_copy_on_wrong_size(self, hub, api, dirs):
        out, cache = dirs
        content = hub.add("keep.bin", os.urandom(2048), served=b"")
        out.mkdir(parents=True)
        (out / "keep.bin").write_bytes(content)

        with pytest.raises(FileSizeMismatchError):
            api.download_file(
                REPO, "dataset", "keep.bin", local_dir=out, cache_dir=cache, force=True, file_size=len(content)
            )

        assert (out / "keep.bin").read_bytes() == content


class TestHashValidation:
    def test_cdn_hash_detects_empty_body_without_caller_metadata(self, hub, api, dirs):
        hub.add("empty.bin", b"original", served=b"")
        with pytest.raises(FileIntegrityError):
            api.download_file(REPO, "dataset", "empty.bin", local_dir=dirs[0], cache_dir=dirs[1])
        assert not (dirs[0] / "empty.bin").exists()
        assert len(hub.seen("empty.bin", "cdn")) == 3

    def test_hash_mismatch_is_retried_then_leaves_nothing(self, hub, api, dirs):
        out, cache = dirs
        content = hub.add("w.bin", os.urandom(10_000), served=os.urandom(10_000))

        with pytest.raises(FileIntegrityError) as excinfo:
            api.download_file(
                REPO,
                "dataset",
                "w.bin",
                local_dir=out,
                cache_dir=cache,
                file_size=len(content),
                expected_sha256=_sha256(content),
            )

        assert not isinstance(excinfo.value, FileSizeMismatchError)
        assert len(hub.seen("w.bin", "cdn")) == _download.DOWNLOAD_HASH_RETRY_TIMES
        assert not (out / "w.bin").exists()
        assert not (out / "w.bin.incomplete").exists()


class TestResume:
    def test_truncated_transfer_resumes_from_received_bytes(self, hub, api, dirs, monkeypatch):
        out, cache = dirs
        # read1 preserves partial chunks so interrupted transfers resume from the exact byte offset.
        monkeypatch.setattr(_download, "DOWNLOAD_CHUNK_SIZE", 8192)
        content = hub.add("big.bin", os.urandom(200_000), truncate_first=70_000)

        path = api.download_file(
            REPO,
            "dataset",
            "big.bin",
            local_dir=out,
            cache_dir=cache,
            file_size=len(content),
            expected_sha256=_sha256(content),
        )

        assert path.read_bytes() == content
        assert [r.range for r in hub.seen("big.bin", "cdn")] == [None, "bytes=70000-"]

    def test_server_ignoring_range_rewrites_instead_of_splicing(self, hub, api, dirs):
        out, cache = dirs
        content = hub.add("plain.txt", os.urandom(50_000), mode="direct")
        out.mkdir(parents=True)
        (out / "plain.txt.incomplete").write_bytes(content[:20_000])

        path = api.download_file(
            REPO,
            "dataset",
            "plain.txt",
            local_dir=out,
            cache_dir=cache,
            file_size=len(content),
            expected_sha256=_sha256(content),
        )

        assert path.read_bytes() == content
        assert [r.range for r in hub.seen("plain.txt", "api")] == ["bytes=20000-"]

    def test_complete_partial_is_finalised_without_a_request(self, hub, api, dirs):
        out, cache = dirs
        content = hub.add("done.bin", os.urandom(30_000))
        out.mkdir(parents=True)
        (out / "done.bin.incomplete").write_bytes(content)

        path = api.download_file(
            REPO,
            "dataset",
            "done.bin",
            local_dir=out,
            cache_dir=cache,
            file_size=len(content),
            expected_sha256=_sha256(content),
        )

        assert path.read_bytes() == content
        assert hub.requests == []

    def test_unresumable_partial_restarts_after_416(self, hub, api, dirs):
        out, cache = dirs
        content = hub.add("obj.bin", os.urandom(30_000))
        out.mkdir(parents=True)
        (out / "obj.bin.incomplete").write_bytes(b"\0" * len(content))

        # Without a listed size the stale partial file can only be detected by the 416.
        path = api.download_file(REPO, "dataset", "obj.bin", local_dir=out, cache_dir=cache)

        assert path.read_bytes() == content
        assert [r.range for r in hub.seen("obj.bin", "cdn")] == ["bytes=30000-", None]


class TestParallelDownload:
    @pytest.fixture(autouse=True)
    def _parallel(self, monkeypatch):
        monkeypatch.setattr(_download, "DOWNLOAD_PARALLELS", 4)
        monkeypatch.setattr(_download, "DOWNLOAD_PARALLEL_THRESHOLD", 1024)
        monkeypatch.setattr(_download, "DOWNLOAD_PART_SIZE", 64 * 1024)

    def _fetch(self, api, dirs, path: str, content: bytes) -> Path:
        out, cache = dirs
        return api.download_file(
            REPO,
            "dataset",
            path,
            local_dir=out,
            cache_dir=cache,
            file_size=len(content),
            expected_sha256=_sha256(content),
        )

    def test_parts_are_fetched_by_range_and_merged(self, hub, api, dirs):
        content = hub.add("p.bin", os.urandom(300_000))

        assert self._fetch(api, dirs, "p.bin", content).read_bytes() == content
        assert len(hub.seen("p.bin", "cdn")) == 5
        assert all(r.range for r in hub.seen("p.bin", "cdn"))
        assert list(dirs[0].glob("p.bin_*")) == []

    def test_truncated_part_resumes_within_its_range(self, hub, api, dirs, monkeypatch):
        monkeypatch.setattr(_download, "DOWNLOAD_CHUNK_SIZE", 8192)
        content = hub.add("p.bin", os.urandom(300_000), truncate_first=10_000)

        assert self._fetch(api, dirs, "p.bin", content).read_bytes() == content
        starts = [int(r.range.removeprefix("bytes=").split("-")[0]) for r in hub.seen("p.bin", "cdn")]
        assert any(start % (64 * 1024) == 10_000 for start in starts)

    def test_server_ignoring_range_falls_back_to_single_stream(self, hub, api, dirs):
        content = hub.add("d.bin", os.urandom(300_000), mode="direct")

        assert self._fetch(api, dirs, "d.bin", content).read_bytes() == content
        assert hub.seen("d.bin", "api")[-1].range is None
        assert list(dirs[0].glob("d.bin_*")) == []

    def test_part_from_object_of_another_size_is_rejected(self, hub, api, dirs):
        content = hub.add("s.bin", os.urandom(300_000), served=os.urandom(100_000))

        with pytest.raises(FileSizeMismatchError, match="reports 100000 bytes"):
            self._fetch(api, dirs, "s.bin", content)
        assert list(dirs[0].glob("s.bin*")) == []


class TestFileDownloadError:
    def test_lists_failures_and_derives_retryability(self):
        transient = FileDownloadError("2 of 5 file(s) failed", failed_files={"a": NetworkError("reset")}, total=5)
        mixed = FileDownloadError(
            "2 of 5 file(s) failed",
            failed_files={"a": NetworkError("reset"), "b": FileSizeMismatchError("short")},
            total=5,
        )

        assert str(mixed).splitlines() == [
            "[E1003] 2 of 5 file(s) failed",
            "  - a: [E1020] reset",
            "  - b: [E2020] short",
        ]
        assert transient.retryable is True
        assert mixed.retryable is False
        assert FileDownloadError("none").retryable is False

    def test_long_failure_lists_are_truncated(self):
        failures = {f"f{i}": NetworkError("x") for i in range(12)}

        assert str(FileDownloadError("12 failed", failed_files=failures)).endswith("  ... and 2 more")


class TestBoundedDownloads:
    def test_header_stall_has_one_retry_layer(self, hub, api, dirs, monkeypatch):
        import modelscope_hub._legacy_api as legacy

        monkeypatch.setattr(legacy, "DOWNLOAD_TIMEOUT", 0.15)
        monkeypatch.setattr(_download, "DOWNLOAD_RETRY_TIMES", 1)
        hub.add("stall.bin", b"x" * 2048, header_delay=2)
        started = time.monotonic()
        with pytest.raises(NetworkError):
            api.download_file(REPO, "dataset", "stall.bin", local_dir=dirs[0], cache_dir=dirs[1], file_size=2048)
        assert len(hub.seen("stall.bin", "cdn")) == 2
        assert time.monotonic() - started < 1
        assert api.legacy._read_session.get_adapter(hub.url).max_retries.total == 0
        assert api.legacy._session.get_adapter(hub.url).max_retries.total == 5

    @pytest.mark.parametrize("status", [429, 503])
    def test_retry_after_is_capped(self, hub, api, dirs, monkeypatch, status):
        monkeypatch.setattr(_download, "DOWNLOAD_RETRY_TIMES", 1)
        monkeypatch.setattr(_download, "DOWNLOAD_RETRY_MAX_DELAY_SECONDS", 0.15)
        hub.add("busy.bin", b"x", status=status, retry_after="999999")
        started = time.monotonic()
        with pytest.raises(Exception):
            api.download_file(REPO, "dataset", "busy.bin", local_dir=dirs[0], cache_dir=dirs[1])
        assert len(hub.seen("busy.bin", "cdn")) == 2
        assert time.monotonic() - started < 1

    @pytest.mark.parametrize("headers", [False, True])
    def test_deadline_stops_trickling_headers_and_body(self, hub, api, dirs, monkeypatch, headers):
        from modelscope_hub._download_control import _DownloadDeadlineError

        monkeypatch.setattr(_download, "DOWNLOAD_TOTAL_TIMEOUT", 0.35)
        hub.add("slow.bin", b"x" * 100, header_trickle=headers, trickle_interval=0 if headers else 0.05)
        started = time.monotonic()
        with pytest.raises(_DownloadDeadlineError):
            api.download_file(REPO, "dataset", "slow.bin", local_dir=dirs[0], cache_dir=dirs[1], file_size=100)
        assert time.monotonic() - started < 1.2
        assert len(hub.seen("slow.bin", "cdn")) == 1
        assert not (dirs[0] / "slow.bin").exists()

    def test_minimum_rate_interrupts_a_connection_that_never_goes_idle(self, hub, api, dirs, monkeypatch):
        from modelscope_hub._download_control import _DownloadTooSlowError

        monkeypatch.setattr(_download, "DOWNLOAD_LOW_SPEED_WINDOW_SECONDS", 0.25)
        monkeypatch.setattr(_download, "DOWNLOAD_MIN_BYTES_PER_SECOND", 1000)
        monkeypatch.setattr(_download, "DOWNLOAD_RETRY_TIMES", 0)
        hub.add("slow.bin", b"x" * 100, trickle_interval=0.05)
        with pytest.raises(_DownloadTooSlowError):
            api.download_file(REPO, "dataset", "slow.bin", local_dir=dirs[0], cache_dir=dirs[1], file_size=100)
        assert len(hub.seen("slow.bin", "cdn")) == 1

    def test_ctrl_c_cancels_queued_and_active_downloads(self, hub, api, dirs, monkeypatch):
        for i in range(10):
            hub.add(f"f{i}.bin", b"x", header_delay=2)

        def interrupt(futures, **kwargs):
            deadline = time.monotonic() + 1
            while len([r for r in hub.requests if r.kind == "cdn"]) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            raise KeyboardInterrupt

        monkeypatch.setattr(_download, "wait", interrupt)
        started = time.monotonic()
        with pytest.raises(KeyboardInterrupt):
            api.download_repo(REPO, "dataset", local_dir=dirs[0], cache_dir=dirs[1], max_workers=2)
        assert time.monotonic() - started < 1.5
        assert len({r.path for r in hub.requests if r.kind == "api"}) == 2

    def test_read_pool_does_not_change_global_or_proxy_connection_types(self):
        from urllib3.poolmanager import pool_classes_by_scheme

        from modelscope_hub._download_control import _ReadHTTPAdapter

        original = dict(pool_classes_by_scheme)
        adapter = _ReadHTTPAdapter(max_retries=0)
        proxy = adapter.proxy_manager_for("http://127.0.0.1:9999")
        assert pool_classes_by_scheme == original
        for scheme, parent in original.items():
            pool = proxy.pool_classes_by_scheme[scheme]
            assert issubclass(pool, parent)
            assert issubclass(pool.ConnectionCls, parent.ConnectionCls)
        adapter.close()

    def test_lock_wait_is_cancellable(self, hub, api, dirs):
        from filelock import FileLock

        from modelscope_hub._download_control import _DownloadCancelledError

        stop = threading.Event()
        lock_path = api.downloader._lock_path(REPO, "dataset", cache_dir=dirs[1], file_path="f.bin")
        lock_path.parent.mkdir(parents=True)
        with FileLock(str(lock_path)):
            timer = threading.Timer(0.1, stop.set)
            timer.start()
            with pytest.raises(_DownloadCancelledError):
                api.downloader.download_file(
                    REPO,
                    "dataset",
                    "f.bin",
                    local_dir=dirs[0],
                    cache_dir=dirs[1],
                    _cancel_event=stop,
                )
            timer.join()
        assert hub.requests == []

    def test_corrupt_forced_refresh_does_not_replace_previous_valid_file(self, hub, api, dirs):
        content = hub.add("f.bin", b"good", served=b"evil")
        dirs[0].mkdir()
        target = dirs[0] / "f.bin"
        target.write_bytes(content)
        with pytest.raises(FileIntegrityError):
            api.download_file(
                REPO,
                "dataset",
                "f.bin",
                local_dir=dirs[0],
                cache_dir=dirs[1],
                force=True,
                file_size=4,
                expected_sha256=_sha256(content),
            )
        assert target.read_bytes() == content
        assert not (dirs[0] / "f.bin.incomplete").exists()


class TestListingOptimisation:
    @pytest.mark.parametrize(
        "patterns, expected",
        [
            (["a/*.bin", "a/b/*.txt"], ["a"]),
            (["a/*.bin", "b/f.json"], ["a", "b"]),
            (["*.bin", "a/f.bin"], [None]),
            ([r"a/.*\\.bin"], [None]),
            (["../a/*.bin"], [None]),
        ],
    )
    def test_safe_roots(self, patterns, expected):
        assert _download._listing_roots(patterns) == expected

    def test_scoped_listing_preserves_glob_selection(self, hub, api, dirs):
        hub.add("a/good.bin", b"yes")
        hub.add("a/no.txt", b"no")
        hub.add("b/other.bin", b"no")
        assert (
            api.download_repo(
                REPO,
                "dataset",
                local_dir=dirs[0],
                cache_dir=dirs[1],
                allow_patterns=["a/*.bin"],
            )
            == dirs[0]
        )
        assert (dirs[0] / "a/good.bin").read_bytes() == b"yes"
        assert not (dirs[0] / "b/other.bin").exists()
        assert not (dirs[0] / "a/no.txt").exists()

    def test_file_list_reuse_skips_the_listing_endpoint(self, hub, api, dirs, monkeypatch):
        content = hub.add("a/good.bin", b"yes")
        manifest = [{"Path": "a/good.bin", "Size": 3, "Sha256": _sha256(content)}]
        monkeypatch.setattr(
            api.legacy, "list_dataset_files_paginated", lambda *args, **kwargs: pytest.fail("列举了清单")
        )
        api.download_repo(REPO, "dataset", local_dir=dirs[0], cache_dir=dirs[1], file_list=manifest)
        assert (dirs[0] / "a/good.bin").read_bytes() == content

    def test_endpoint_probe_obeys_its_total_budget(self, hub, monkeypatch):
        import modelscope_hub.constants as constants
        from modelscope_hub._download_control import _DownloadDeadlineError

        monkeypatch.setattr(constants, "REPO_TREE_PAGE_TIMEOUT_SECONDS", 0.2)
        hub.add("metadata.bin", b"x", header_delay=2)

        def exists(probe, repo_id, repo_type):
            probe.openapi.request(
                "GET",
                url=f"{hub.url}/cdn/metadata.bin",
                require_token=False,
                unwrap=False,
            )
            return True

        monkeypatch.setattr(HubApi, "repo_exists", exists)
        caller = HubApi(token="")
        caller._config._endpoint_overridden = False
        started = time.monotonic()
        with pytest.raises(_DownloadDeadlineError):
            caller.resolve_endpoint_for_read(REPO, repo_type="dataset")
        assert time.monotonic() - started < 1

    def test_region_detection_is_single_flight(self, api, monkeypatch):
        calls = []

        def probe():
            calls.append(1)
            time.sleep(0.03)
            return "cn-test"

        monkeypatch.setattr(api.downloader, "_detect_region_uncached", probe)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=8) as pool:
            assert list(pool.map(lambda _: api.downloader._detect_region(), range(16))) == ["cn-test"] * 16
        assert calls == [1]


class TestDownloadConfiguration:
    def test_defaults_exports_registry_and_readme_are_aligned(self):
        import modelscope_hub.constants as constants

        expected = {
            "DOWNLOAD_TOTAL_TIMEOUT": 3600,
            "DOWNLOAD_MIN_BYTES_PER_SECOND": 1024,
            "DOWNLOAD_LOW_SPEED_WINDOW_SECONDS": 60,
            "DOWNLOAD_RETRY_MAX_DELAY_SECONDS": 60,
            "DOWNLOAD_FILE_LOCK_TIMEOUT_SECONDS": 600,
            "REPO_TREE_PAGE_TIMEOUT_SECONDS": 15,
            "REPO_TREE_LIST_TIMEOUT_SECONDS": 3600,
        }
        registry = {entry.name: entry.default for entry in constants.ENV_REGISTRY}
        readme = (Path(__file__).parents[1] / "README.md").read_text()
        for name, value in expected.items():
            assert getattr(constants, name) == value
            assert name in constants.__all__
            env_name = f"MODELSCOPE_{name}"
            assert registry[env_name] == str(value)
            assert f"| `{env_name}` | `{value}` |" in readme

    def test_zero_can_disable_transport_retries(self, monkeypatch):
        import subprocess
        import sys

        monkeypatch.setenv("MODELSCOPE_DOWNLOAD_MAX_RETRIES", "0")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from modelscope_hub.constants import DOWNLOAD_RETRY_TIMES; print(DOWNLOAD_RETRY_TIMES)",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == "0"


class TestDownloadCallbacks:
    def test_resume_reports_each_byte_once_and_closes_callback(self, hub, api, dirs):
        reports = []

        class Callback(_download.ProgressCallback):
            def __init__(self, filename, file_size):
                super().__init__(filename, file_size)
                self.size = 0
                self.closed = False
                reports.append(self)

            def update(self, size):
                self.size += size

            def end(self):
                self.closed = True

        content = hub.add("resume.bin", b"x" * 5000, truncate_first=2000)
        api.downloader.download_file(
            REPO,
            "dataset",
            "resume.bin",
            local_dir=dirs[0],
            cache_dir=dirs[1],
            file_size=5000,
            expected_sha256=_sha256(content),
            progress_callbacks=[Callback],
        )
        assert len(reports) == 1
        assert reports[0].size == 5000
        assert reports[0].closed
