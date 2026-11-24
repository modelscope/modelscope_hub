"""Download counting contract: one ``snapshot-identifier`` per logical snapshot download.

The Hub counts the first request carrying a new identifier as one whole-repository
download and a request without one as a share of it. These tests pin which requests
carry the header: every file, retry, resumed transfer and range part of one snapshot task
share a single identifier; standalone file downloads send none; report-mode tasks persist
their identifier so a resumed task keeps it. The fake hub is a real local HTTP server, so
the assertions inspect the headers actually received on the wire.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import threading
import time
import zipfile
from pathlib import Path

import pytest

from modelscope_hub import _download
from modelscope_hub.cli.main import run_cmd
from modelscope_hub.errors import FileDownloadError, FileIntegrityError
from tests.test_download_integrity import REPO, _sha256
from tests.test_download_integrity import api as api
from tests.test_download_integrity import dirs as dirs
from tests.test_download_integrity import hub as hub


def _file_requests(hub, kinds=("api",)):
    return [r for r in hub.requests if r.kind in kinds]


def _ids(requests) -> set:
    return {r.snapshot for r in requests}


def _only_id(requests) -> str:
    ids = _ids(requests)
    assert len(ids) == 1, ids
    (sid,) = ids
    assert sid and sid.strip()
    return sid


def _snapshot(api, out: Path, cache: Path, **kwargs) -> Path:
    return api.download_repo(REPO, "dataset", local_dir=out, cache_dir=cache, **kwargs)


def _tracker(cache: Path) -> dict:
    (path,) = (cache / ".download_state").glob(".ms_download_*.json")
    return json.loads(path.read_text())


class TestSnapshotTask:
    def test_every_request_of_one_snapshot_shares_one_identifier(self, hub, api, dirs):
        for i in range(6):
            hub.add(f"data/{i}.bin", os.urandom(1000 + i))
        hub.add("README.md", b"# card", mode="direct")
        hub.add("notes.txt", b"plain text", mode="direct")

        _snapshot(api, *dirs)

        assert len(_file_requests(hub)) == 8
        sid = _only_id(_file_requests(hub, kinds=("api", "cdn")))
        assert len(sid) == 32

    def test_concurrent_workers_keep_one_identifier(self, hub, api, dirs):
        for i in range(40):
            hub.add(f"shards/{i:02}.bin", os.urandom(512))

        _snapshot(api, *dirs, max_workers=8)

        assert len(_file_requests(hub)) == 40
        _only_id(_file_requests(hub))

    def test_retries_and_resumed_transfers_reuse_the_identifier(self, hub, api, dirs, monkeypatch):
        out, cache = dirs
        monkeypatch.setattr(_download, "DOWNLOAD_CHUNK_SIZE", 8192)
        hub.add("big.bin", os.urandom(200_000), truncate_first=70_000)
        seeded = hub.add("seeded.bin", os.urandom(50_000))
        out.mkdir(parents=True)
        (out / "seeded.bin.incomplete").write_bytes(seeded[:20_000])

        _snapshot(api, out, cache)

        assert [r.range for r in hub.seen("big.bin", "cdn")] == [None, "bytes=70000-"]
        assert [r.range for r in hub.seen("seeded.bin", "api")] == ["bytes=20000-"]
        requests = _file_requests(hub)
        _only_id(requests)
        request_ids = [r.request_id for r in requests]
        assert len(set(request_ids)) == len(request_ids)
        assert not {r.snapshot for r in requests} & set(request_ids)

    def test_integrity_retries_reuse_the_identifier(self, hub, api, dirs):
        hub.add("bad.bin", b"good data", served=b"evil data")
        hub.add("good.bin", b"fine")

        with pytest.raises(FileDownloadError):
            _snapshot(api, *dirs)

        assert len(hub.seen("bad.bin", "api")) == 3
        _only_id(_file_requests(hub))

    def test_parallel_range_parts_carry_the_identifier(self, hub, api, dirs, monkeypatch):
        monkeypatch.setattr(_download, "DOWNLOAD_PARALLELS", 4)
        monkeypatch.setattr(_download, "DOWNLOAD_PARALLEL_THRESHOLD", 1024)
        monkeypatch.setattr(_download, "DOWNLOAD_PART_SIZE", 64 * 1024)
        content = hub.add("p.bin", os.urandom(300_000))

        assert (_snapshot(api, *dirs) / "p.bin").read_bytes() == content

        parts = hub.seen("p.bin", "api")
        assert len(parts) == 5 and all(r.range for r in parts)
        _only_id(parts)

    def test_route_probes_carry_the_task_identifier(self, hub, api, dirs, monkeypatch):
        monkeypatch.setenv("MODELSCOPE_DOWNLOAD_INTER_CLOUD_REGIONS", "cn-hangzhou")
        hub.add("a.bin", os.urandom(800))
        hub.add("b.bin", os.urandom(900))

        _snapshot(api, *dirs)

        heads = _file_requests(hub, kinds=("head",))
        assert heads
        assert _only_id(heads + _file_requests(hub)) == _only_id(_file_requests(hub))

    def test_filtered_snapshot_shares_one_identifier(self, hub, api, dirs):
        for name in ("data/a.bin", "data/b.bin", "data/skip.bin", "other/c.bin"):
            hub.add(name, os.urandom(300))

        _snapshot(api, *dirs, allow_patterns=["data/*"], ignore_patterns=["data/skip.bin"])

        assert sorted(r.path for r in _file_requests(hub)) == ["data/a.bin", "data/b.bin"]
        _only_id(_file_requests(hub))

    def test_sequential_snapshots_get_distinct_identifiers(self, hub, api, tmp_path):
        for i in range(3):
            hub.add(f"f{i}.bin", os.urandom(200))

        _snapshot(api, tmp_path / "one", tmp_path / "cache1")
        first = _only_id(_file_requests(hub))
        hub.requests.clear()
        _snapshot(api, tmp_path / "two", tmp_path / "cache2")
        second = _only_id(_file_requests(hub))

        assert first != second

    def test_concurrent_snapshots_on_one_api_are_isolated(self, hub, api, tmp_path):
        for group in ("a", "b"):
            for i in range(5):
                hub.add(f"{group}/{i}.bin", os.urandom(400))
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []

        def task(group: str) -> None:
            try:
                barrier.wait()
                _snapshot(api, tmp_path / group, tmp_path / f"cache-{group}", allow_patterns=[f"{group}/*"])
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=task, args=(g,)) for g in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors
        a_id = _only_id([r for r in _file_requests(hub) if r.path.startswith("a/")])
        b_id = _only_id([r for r in _file_requests(hub) if r.path.startswith("b/")])
        assert a_id != b_id
        assert not any("snapshot" in name for name in vars(api.downloader))

    def test_a_caller_supplied_identifier_is_used_unchanged(self, hub, api, dirs):
        hub.add("x.bin", os.urandom(100))

        api.downloader.download_repo(REPO, "dataset", local_dir=dirs[0], cache_dir=dirs[1], _snapshot_id="task-42")

        assert _ids(_file_requests(hub)) == {"task-42"}

    def test_a_blank_caller_identifier_is_replaced_by_a_fresh_one(self, hub, api, dirs):
        hub.add("x.bin", os.urandom(100))

        api.downloader.download_repo(REPO, "dataset", local_dir=dirs[0], cache_dir=dirs[1], _snapshot_id="  ")

        assert len(_only_id(_file_requests(hub))) == 32

    def test_full_cache_hit_sends_no_download_request(self, hub, api, dirs):
        for i in range(4):
            hub.add(f"c{i}.bin", os.urandom(256))
        _snapshot(api, *dirs)
        hub.requests.clear()

        _snapshot(api, *dirs)

        assert _file_requests(hub, kinds=("api", "cdn", "head")) == []


class TestStandaloneFiles:
    def test_api_download_file_sends_no_identifier_even_when_retrying(self, hub, api, dirs, monkeypatch):
        monkeypatch.setattr(_download, "DOWNLOAD_CHUNK_SIZE", 8192)
        content = hub.add("big.bin", os.urandom(200_000), truncate_first=70_000)
        hub.add("bad.bin", b"good data", served=b"evil data")

        path = api.download_file(REPO, "dataset", "big.bin", local_dir=dirs[0], file_size=len(content))
        with pytest.raises(FileIntegrityError):
            api.download_file(REPO, "dataset", "bad.bin", local_dir=dirs[0], expected_sha256=_sha256(b"good data"))

        assert path.read_bytes() == content
        requests = _file_requests(hub, kinds=("api", "cdn"))
        assert len(hub.seen("big.bin", "api")) == 2 and len(hub.seen("bad.bin", "api")) == 3
        assert _ids(requests) == {None}

    @pytest.mark.parametrize("name", ["model_file_download", "dataset_file_download"])
    def test_compat_file_downloads_send_no_identifier(self, hub, dirs, name):
        hub.add("weights.bin", os.urandom(300))
        function = getattr(importlib.import_module("modelscope_hub.compat"), name)

        function(REPO, "weights.bin", local_dir=str(dirs[0]), token="", endpoint=hub.url)

        assert len(_file_requests(hub)) == 1
        assert _ids(_file_requests(hub, kinds=("api", "cdn"))) == {None}
        assert "invoked_by/compat" in _file_requests(hub)[0].user_agent

    def test_cli_listed_files_are_standalone_downloads(self, hub, dirs, monkeypatch):
        monkeypatch.setenv("MODELSCOPE_API_TOKEN", "")
        hub.add("a.bin", os.urandom(300))
        hub.add("b.bin", os.urandom(300))

        args = ["--endpoint", hub.url, "download", REPO, "a.bin", "b.bin", "--repo-type", "dataset"]
        assert run_cmd([*args, "--local-dir", str(dirs[0])]) == 0

        assert sorted(r.path for r in _file_requests(hub)) == ["a.bin", "b.bin"]
        assert _ids(_file_requests(hub)) == {None}
        assert all("invoked_by/cli" in r.user_agent for r in _file_requests(hub))

    def test_kernels_single_file_download_sends_no_identifier(self, hub, api, tmp_path):
        pytest.importorskip("huggingface_hub")
        from modelscope_hub.kernels._hub import MsKernelApi

        hub.add("build/torch-cpu/kernel.py", b"print('kernel')", mode="direct")

        MsKernelApi(hub_api=api).hf_hub_download(
            repo_id=REPO, filename="build/torch-cpu/kernel.py", cache_dir=tmp_path / "cache"
        )

        assert len(_file_requests(hub)) == 1
        assert _ids(_file_requests(hub)) == {None}


class TestEntryPoints:
    @pytest.mark.parametrize("name", ["snapshot_download", "dataset_snapshot_download"])
    def test_compat_snapshots_send_one_identifier(self, hub, dirs, name):
        for i in range(3):
            hub.add(f"d{i}.bin", os.urandom(300))
        function = getattr(importlib.import_module("modelscope_hub.compat"), name)
        kwargs = dict(endpoint=hub.url, token="", local_dir=str(dirs[0]), cache_dir=str(dirs[1]))
        if name == "snapshot_download":
            kwargs["repo_type"] = "dataset"

        function(REPO, **kwargs)

        _only_id(_file_requests(hub))
        assert all("invoked_by/compat" in r.user_agent for r in _file_requests(hub))

    @pytest.mark.parametrize("extra", [[], ["--continue-on-error"]])
    def test_cli_snapshots_send_one_identifier(self, hub, dirs, monkeypatch, extra):
        monkeypatch.setenv("MODELSCOPE_API_TOKEN", "")
        for i in range(3):
            hub.add(f"d{i}.bin", os.urandom(300))
        args = ["--endpoint", hub.url, "download", REPO, "--repo-type", "dataset"]

        assert run_cmd([*args, "--local-dir", str(dirs[0]), "--cache-dir", str(dirs[1]), *extra]) == 0

        _only_id(_file_requests(hub))
        assert all("invoked_by/cli" in r.user_agent for r in _file_requests(hub))

    def test_skill_archive_download_sends_no_identifier(self, hub, api, tmp_path):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("skill/SKILL.md", "# skill")
        hub.archive = buffer.getvalue()

        out = api.download_repo("owner/skill", "skill", local_dir=tmp_path / "skill")

        assert (out / "SKILL.md").read_text() == "# skill"
        archives = _file_requests(hub, kinds=("archive",))
        assert len(archives) == 1 and archives[0].snapshot is None


class TestReportTask:
    @staticmethod
    def _run(api, dirs, **kwargs):
        return api.download_repo_report(REPO, "dataset", local_dir=dirs[0], cache_dir=dirs[1], **kwargs)

    def test_identifier_is_persisted_and_reused_after_cancellation(self, hub, api, dirs, monkeypatch):
        import modelscope_hub._resilient_download as module

        hub.add("a.bin", os.urandom(300))
        hub.add("slow.bin", os.urandom(300), header_delay=1)
        original_wait = module.wait
        entered = threading.Event()

        def interrupt(*args, **kwargs):
            if not entered.is_set():
                entered.set()
                time.sleep(0.2)
                raise KeyboardInterrupt
            return original_wait(*args, **kwargs)

        monkeypatch.setattr(module, "wait", interrupt)
        with pytest.raises(KeyboardInterrupt):
            self._run(api, dirs)
        cancelled = _tracker(dirs[1])
        assert cancelled["report"]["status"] == "CANCELLED"
        assert cancelled["snapshot"]["open"] is True
        first_run = list(_file_requests(hub))

        monkeypatch.setattr(module, "wait", original_wait)
        assert self._run(api, dirs).complete

        resumed = _file_requests(hub)[len(first_run) :]
        assert resumed
        assert _only_id(_file_requests(hub)) == cancelled["snapshot"]["id"]
        assert _tracker(dirs[1])["snapshot"] == {"id": cancelled["snapshot"]["id"], "open": False}

    def test_partial_task_keeps_its_identifier_across_reruns(self, hub, api, dirs):
        hub.add("bad.bin", b"data", served=b"")
        hub.add("good.bin", b"good")

        assert self._run(api, dirs).status == "PARTIAL"
        assert _tracker(dirs[1])["snapshot"]["open"] is True
        assert self._run(api, dirs, retry_failed=True).status == "PARTIAL"

        assert _only_id(_file_requests(hub)) == _tracker(dirs[1])["snapshot"]["id"]

    def test_run_after_a_complete_task_starts_a_new_identifier(self, hub, api, dirs):
        hub.add("a.bin", os.urandom(300))
        assert self._run(api, dirs).complete
        first = _only_id(_file_requests(hub))
        assert _tracker(dirs[1])["snapshot"] == {"id": first, "open": False}
        (dirs[0] / "a.bin").unlink()
        hub.requests.clear()

        assert self._run(api, dirs).complete

        second = _only_id(_file_requests(hub))
        assert second != first
        assert _tracker(dirs[1])["snapshot"] == {"id": second, "open": False}

    @pytest.mark.parametrize(
        "damage",
        [
            lambda data: data.update(snapshot={"id": "not-hex", "open": True}),
            lambda data: data.update(snapshot="garbage"),
            lambda data: data.pop("snapshot"),
        ],
    )
    def test_invalid_or_missing_identifier_starts_a_new_one(self, hub, api, dirs, damage):
        hub.add("bad.bin", b"data", served=b"")
        assert self._run(api, dirs).status == "PARTIAL"
        first = _only_id(_file_requests(hub))
        (path,) = (dirs[1] / ".download_state").glob(".ms_download_*.json")
        data = json.loads(path.read_text())
        damage(data)
        path.write_text(json.dumps(data))
        hub.requests.clear()

        self._run(api, dirs, retry_failed=True)

        assert _only_id(_file_requests(hub)) not in (first, "not-hex")

    def test_corrupt_tracker_falls_back_to_a_new_identifier(self, hub, api, dirs):
        hub.add("bad.bin", b"data", served=b"")
        hub.add("good.bin", b"good")
        self._run(api, dirs)
        first = _only_id(_file_requests(hub))
        (path,) = (dirs[1] / ".download_state").glob(".ms_download_*.json")
        path.write_text('{"version":')
        hub.requests.clear()

        again = self._run(api, dirs)

        assert again.state_warnings
        assert _only_id(_file_requests(hub)) != first

    def test_unwritable_tracker_still_sends_one_identifier(self, hub, api, dirs, monkeypatch):
        hub.add("a.bin", os.urandom(300))
        hub.add("b.bin", os.urandom(300))
        original = Path.replace

        def fail_state_replace(path, target):
            if str(target).endswith(".json"):
                raise PermissionError("hint file cannot be replaced")
            return original(path, target)

        monkeypatch.setattr(Path, "replace", fail_state_replace)

        report = self._run(api, dirs)

        assert report.complete and report.state_warnings
        _only_id(_file_requests(hub))

    def test_new_tracker_writes_identifier_with_its_scope_header(self, hub, api, dirs, monkeypatch):
        from modelscope_hub._download_state import _DownloadState

        snapshots = []
        original_save = _DownloadState._save

        def saved(state):
            snapshots.append(dict(state._snapshot))
            original_save(state)

        monkeypatch.setattr(_DownloadState, "_save", saved)
        hub.add("a.bin", os.urandom(300))

        self._run(api, dirs)

        sid = _only_id(_file_requests(hub))
        assert snapshots == [{"id": sid, "open": True}, {"id": sid, "open": False}]


def test_header_builder_forwards_but_never_generates_an_identifier(api, monkeypatch):
    manager = api.downloader

    def forbidden():
        raise AssertionError("the header builder must not generate identifiers")

    monkeypatch.setattr(_download.uuid, "uuid4", forbidden)

    assert "snapshot-identifier" not in manager._build_download_headers(accelerate=False)
    assert "snapshot-identifier" not in manager._build_download_headers(accelerate=False, snapshot_id="")
    assert "snapshot-identifier" not in manager._build_download_headers(accelerate=False, snapshot_id="   ")
    assert manager._build_download_headers(accelerate=False, snapshot_id="abc")["snapshot-identifier"] == "abc"


def test_cached_dataset_meta_files_issue_no_request(hub, tmp_path):
    pytest.importorskip("modelscope.utils.constant")
    from modelscope_hub.compat import LegacyHubApi

    hub.add("meta/train.json", b'{"split": "train"}', mode="direct")
    api = LegacyHubApi(endpoint=hub.url, token="")
    entries = [{"Path": "meta/train.json"}, {"Path": "data/ignored.bin"}]
    kwargs = dict(dataset_name="data", namespace="owner", revision="master", dataset_type=2, file_list=entries)

    first, _ = api.get_dataset_meta_files_local_paths(meta_cache_dir=str(tmp_path), **kwargs)
    second, _ = api.get_dataset_meta_files_local_paths(meta_cache_dir=str(tmp_path), **kwargs)

    assert first == second == {".json": [str(tmp_path / "meta" / "train.json")]}
    assert len(_file_requests(hub)) == 1
