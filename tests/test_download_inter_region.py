"""Regression tests for inter-region download route isolation."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from modelscope_hub._download import DownloadManager
from modelscope_hub.config import HubConfig

_REPO_ID = "wangxingjun778/xxx"


class _Response:
    status_code = 200
    headers = {"Content-Length": "2"}

    @staticmethod
    def iter_content(chunk_size: int):
        del chunk_size
        yield b"ok"


class _Client:
    endpoint = "https://modelscope.cn"
    token = None

    def __init__(self) -> None:
        self.download_headers: dict[str, dict[str, str]] = {}

    @staticmethod
    def get_download_url(repo_id: str, repo_type: str, file_path: str, revision: str) -> str:
        return f"https://modelscope.cn/api/v1/{repo_type}s/{repo_id}/repo?revision={revision}&FilePath={file_path}"

    def download_stream(
        self,
        *,
        repo_id: str,
        repo_type: str,
        file_path: str,
        revision: str,
        headers: dict[str, str],
    ) -> _Response:
        del repo_id, repo_type, revision
        self.download_headers[file_path] = headers
        return _Response()


def _manager(tmp_path: Path) -> tuple[DownloadManager, _Client]:
    client = _Client()
    return DownloadManager(client, HubConfig(cache_dir=tmp_path / "cache", token="")), client


def _file_path(url: str) -> str:
    return parse_qs(urlparse(url).query)["FilePath"][0]


def _download(manager: DownloadManager, tmp_path: Path, file_path: str, revision: str = "master") -> None:
    target = tmp_path / "downloads" / revision / file_path
    target.parent.mkdir(parents=True, exist_ok=True)
    manager._download_with_resume(_REPO_ID, "dataset", file_path, revision, target)


def _route_probe(peer_by_path: dict[str, str], calls: list[tuple[str, str]]):
    def probe(url: str, headers: dict[str, str], cookies=None, timeout: float = 5.0) -> str:
        del cookies, timeout
        path = _file_path(url)
        region = headers.get("x-aliyun-region-id", "")
        calls.append((path, region))
        if path == ".gitattributes":
            return ""
        bucket, peer = peer_by_path[path].split("@")
        if region == peer:
            return f"https://{bucket}.oss-{peer}-internal.aliyuncs.com/object?signature=peer"
        return f"https://{bucket}.oss-cn-hangzhou.aliyuncs.com/object?signature=public"

    return probe


def _configure_regions(monkeypatch) -> None:
    monkeypatch.setenv("MODELSCOPE_DOWNLOAD_INTRA_CLOUD", "false")
    monkeypatch.setenv("MODELSCOPE_DOWNLOAD_INTER_CLOUD_REGIONS", "peer-a,peer-b")


def test_metadata_default_route_does_not_block_lfs_peer_probe(tmp_path, monkeypatch):
    """REG-01: serial metadata 200 must not populate an LFS default cache."""
    _configure_regions(monkeypatch)
    manager, client = _manager(tmp_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(manager, "_probe_redirect_url", _route_probe({"large.tar": "bucket-a@peer-a"}, calls))

    _download(manager, tmp_path, ".gitattributes")
    _download(manager, tmp_path, "large.tar")

    assert client.download_headers["large.tar"]["x-aliyun-region-id"] == "peer-a"
    assert ("large.tar", "peer-a") in calls
    assert len(manager._inter_region_cache) == 1


def test_reversed_file_order_keeps_lfs_internal_route(tmp_path, monkeypatch):
    """REG-02: an earlier LFS resolution must be independent of later metadata."""
    _configure_regions(monkeypatch)
    manager, client = _manager(tmp_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(manager, "_probe_redirect_url", _route_probe({"large.tar": "bucket-a@peer-a"}, calls))

    _download(manager, tmp_path, "large.tar")
    _download(manager, tmp_path, ".gitattributes")

    assert client.download_headers["large.tar"]["x-aliyun-region-id"] == "peer-a"
    assert ".gitattributes" in client.download_headers
    assert len(manager._inter_region_cache) == 1


def test_same_storage_route_reuses_confirmed_internal_result(tmp_path, monkeypatch):
    """REG-04: signed URLs differ per object but the same endpoint shares one probe."""
    _configure_regions(monkeypatch)
    manager, client = _manager(tmp_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        manager,
        "_probe_redirect_url",
        _route_probe({"one.tar": "bucket-a@peer-a", "two.tar": "bucket-a@peer-a"}, calls),
    )

    _download(manager, tmp_path, "one.tar")
    _download(manager, tmp_path, "two.tar")

    assert client.download_headers["one.tar"]["x-aliyun-region-id"] == "peer-a"
    assert client.download_headers["two.tar"]["x-aliyun-region-id"] == "peer-a"
    assert calls.count(("one.tar", "peer-a")) == 1
    assert ("two.tar", "peer-a") not in calls


def test_different_storage_routes_do_not_share_peer_region(tmp_path, monkeypatch):
    """REG-05: route/bucket isolation prevents a peer result crossing objects."""
    _configure_regions(monkeypatch)
    manager, client = _manager(tmp_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        manager,
        "_probe_redirect_url",
        _route_probe({"one.tar": "bucket-a@peer-a", "two.tar": "bucket-b@peer-b"}, calls),
    )

    _download(manager, tmp_path, "one.tar")
    _download(manager, tmp_path, "two.tar")

    assert client.download_headers["one.tar"]["x-aliyun-region-id"] == "peer-a"
    assert client.download_headers["two.tar"]["x-aliyun-region-id"] == "peer-b"
    assert ("two.tar", "peer-b") in calls
    assert len(manager._inter_region_cache) == 2


def test_revision_is_part_of_storage_route_cache_scope(tmp_path, monkeypatch):
    """REG-06: a new revision must resolve its own route before reuse."""
    _configure_regions(monkeypatch)
    manager, client = _manager(tmp_path)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(manager, "_probe_redirect_url", _route_probe({"large.tar": "bucket-a@peer-a"}, calls))

    _download(manager, tmp_path, "large.tar", revision="v1")
    _download(manager, tmp_path, "large.tar", revision="v2")

    assert client.download_headers["large.tar"]["x-aliyun-region-id"] == "peer-a"
    assert calls.count(("large.tar", "peer-a")) == 2
    assert len(manager._inter_region_cache) == 2


def test_probe_failure_does_not_cache_default_result(tmp_path, monkeypatch):
    """REG-07: a transient initial failure must not poison a later LFS request."""
    _configure_regions(monkeypatch)
    manager, client = _manager(tmp_path)
    attempts = 0

    def probe(url: str, headers: dict[str, str], cookies=None, timeout: float = 5.0) -> str:
        nonlocal attempts
        del cookies, timeout
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient probe failure")
        if headers.get("x-aliyun-region-id") == "peer-a":
            return "https://bucket-a.oss-peer-a-internal.aliyuncs.com/object"
        return "https://bucket-a.oss-cn-hangzhou.aliyuncs.com/object"

    monkeypatch.setattr(manager, "_probe_redirect_url", probe)

    _download(manager, tmp_path, "large.tar")
    _download(manager, tmp_path, "large.tar", revision="retry")

    assert client.download_headers["large.tar"]["x-aliyun-region-id"] == "peer-a"
    assert len(manager._inter_region_cache) == 1


def test_peer_miss_is_not_negative_cached(tmp_path, monkeypatch):
    """REG-08: after an all-peer miss, a later request must probe again."""
    _configure_regions(monkeypatch)
    manager, client = _manager(tmp_path)
    resolve_internal = False

    def probe(url: str, headers: dict[str, str], cookies=None, timeout: float = 5.0) -> str:
        del url, cookies, timeout
        if resolve_internal and headers.get("x-aliyun-region-id") == "peer-a":
            return "https://bucket-a.oss-peer-a-internal.aliyuncs.com/object"
        return "https://bucket-a.oss-cn-hangzhou.aliyuncs.com/object"

    monkeypatch.setattr(manager, "_probe_redirect_url", probe)

    _download(manager, tmp_path, "large.tar")
    assert "x-aliyun-region-id" not in client.download_headers["large.tar"]
    assert not manager._inter_region_cache

    resolve_internal = True
    _download(manager, tmp_path, "large.tar", revision="retry")

    assert client.download_headers["large.tar"]["x-aliyun-region-id"] == "peer-a"
    assert len(manager._inter_region_cache) == 1


def test_parallel_same_route_uses_one_peer_probe(tmp_path, monkeypatch):
    """REG-03/04: concurrent files share a route-specific in-flight probe."""
    _configure_regions(monkeypatch)
    manager, client = _manager(tmp_path)
    initial_barrier = threading.Barrier(2)
    calls: list[tuple[str, str]] = []
    calls_lock = threading.Lock()

    def probe(url: str, headers: dict[str, str], cookies=None, timeout: float = 5.0) -> str:
        del cookies, timeout
        path = _file_path(url)
        region = headers.get("x-aliyun-region-id", "")
        if not region:
            initial_barrier.wait(timeout=3)
        with calls_lock:
            calls.append((path, region))
        if region == "peer-a":
            return "https://bucket-a.oss-peer-a-internal.aliyuncs.com/object"
        return "https://bucket-a.oss-cn-hangzhou.aliyuncs.com/object"

    monkeypatch.setattr(manager, "_probe_redirect_url", probe)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_download, manager, tmp_path, name) for name in ("one.tar", "two.tar")]
        for future in futures:
            future.result()

    assert client.download_headers["one.tar"]["x-aliyun-region-id"] == "peer-a"
    assert client.download_headers["two.tar"]["x-aliyun-region-id"] == "peer-a"
    assert sum(region == "peer-a" for _, region in calls) == 1
