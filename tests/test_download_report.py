"""Persistent isolation and bounded recovery across large download batches."""

from __future__ import annotations

import importlib
import json
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from modelscope_hub import DownloadReport, DownloadRetryPolicy, HubApi
from modelscope_hub.cli.main import run_cmd
from modelscope_hub.errors import InvalidParameter, NetworkError
from tests.test_download_integrity import REPO, _sha256
from tests.test_download_integrity import api as api
from tests.test_download_integrity import dirs as dirs
from tests.test_download_integrity import hub as hub


def manifest(hub):
    return [
        {"Path": path, "Size": len(file.content), "Sha256": _sha256(file.content)} for path, file in hub.files.items()
    ]


def run(api, dirs, **kwargs):
    return api.download_repo_report(REPO, "dataset", local_dir=dirs[0], cache_dir=dirs[1], **kwargs)


def test_failure_does_not_stop_files_after_the_first_batch(hub, api, dirs):
    for i in range(25):
        hub.add(f"shards/{i:02}.tar", b"data" * 1024, served=b"" if i == 3 else None)
    report = run(api, dirs, max_workers=2)
    assert report.status == "PARTIAL"
    assert (report.total, report.succeeded, report.failed) == (25, 24, 1)
    assert report.listing_complete
    assert len(hub.seen("shards/03.tar", "cdn")) == 2
    assert (dirs[0] / "shards/24.tar").exists()
    assert not (dirs[0] / "shards/03.tar").exists()
    assert len(list(report.iter_files("VERIFIED"))) == 24
    assert json.loads(Path(report.report_path).read_text())["report"]["status"] == "PARTIAL"


def test_new_api_instances_and_batches_share_failure_isolation(hub, api, dirs):
    hub.add("bad.bin", b"data", served=b"")
    hub.add("next.bin", b"good")
    first = run(api, dirs, allow_patterns=["bad.bin"])
    again = run(HubApi(endpoint=hub.url, token=""), dirs)
    assert first.tracker_path == again.tracker_path
    assert again.skipped_failures == 1
    assert len(hub.seen("bad.bin", "cdn")) == 2
    assert (dirs[0] / "next.bin").read_bytes() == b"good"
    assert first.total == 1  # The first report remains immutable after later batches.
    assert [file.path for file in first.iter_files()] == ["bad.bin"]


def test_failed_file_is_only_reopened_explicitly_or_on_metadata_change(hub, api, dirs):
    hub.add("bad.bin", b"data", served=b"")
    first = run(api, dirs)
    hub.files["bad.bin"].served = None
    second = run(api, dirs)
    assert not second.complete
    assert len(hub.seen("bad.bin", "cdn")) == 2
    recovered = run(api, dirs, retry_failed=True)
    assert recovered.complete
    assert len(hub.seen("bad.bin", "cdn")) == 3
    hub.files["bad.bin"].content = b"new version"
    changed = run(api, dirs)
    assert changed.complete
    assert (dirs[0] / "bad.bin").read_bytes() == b"new version"
    assert first.failed == 1


def test_tracker_scope_rejects_another_output_or_revision(hub, api, dirs, tmp_path):
    hub.add("x.bin", b"data")
    ledger = tmp_path / ".ms_download_state.json"
    run(api, dirs, tracker_path=ledger)
    with pytest.raises(InvalidParameter):
        run(api, (tmp_path / "other", dirs[1]), tracker_path=ledger)
    with pytest.raises(InvalidParameter):
        run(api, dirs, tracker_path=ledger, revision="v2")
    with pytest.raises(InvalidParameter):
        run(api, dirs, tracker_path=dirs[0] / ".ms_download_state.json")


def test_long_retry_after_defers_without_sleeping_or_occupying_a_worker(hub, api, dirs):
    hub.add("busy.bin", b"data", status=429, retry_after="9999")
    hub.add("next.bin", b"good")
    started = time.monotonic()
    report = run(api, dirs, max_workers=1)
    assert time.monotonic() - started < 1.5
    assert report.deferred == 1
    assert report.succeeded == 1
    assert len(hub.seen("busy.bin", "cdn")) == 1
    run(api, dirs)
    assert len(hub.seen("busy.bin", "cdn")) == 1


def test_transport_and_hash_retries_share_one_file_budget(hub, api, dirs):
    hub.add("bad.bin", b"good" * 2048, served=b"evil" * 2048, truncate_first=3000)
    report = run(api, dirs, file_list=manifest(hub))
    result = next(report.iter_files())
    assert result.status == "FAILED"
    assert result.retries == 2
    assert len(hub.seen("bad.bin", "cdn")) == 3
    assert not (dirs[0] / "bad.bin").exists()


def test_complete_but_corrupt_partial_is_rejected_before_network_and_repaired(hub, api, dirs):
    content = hub.add("partial.bin", b"correct data" * 4096)
    dirs[0].mkdir()
    partial = dirs[0] / "partial.bin.incomplete"
    partial.write_bytes(b"wrong bytes!" * 4096)
    assert partial.stat().st_size == len(content)
    report = run(api, dirs, file_list=manifest(hub))
    assert report.complete
    assert next(report.iter_files()).retries == 1
    assert (dirs[0] / "partial.bin").read_bytes() == content
    assert not partial.exists()
    assert len(hub.seen("partial.bin", "cdn")) == 1
    report.close()


def test_same_size_corrupt_parallel_parts_are_rejected_and_do_not_stop_next_file(hub, api, dirs, monkeypatch):
    from modelscope_hub import _download

    monkeypatch.setattr(_download, "DOWNLOAD_PARALLEL_THRESHOLD", 64 * 1024)
    monkeypatch.setattr(_download, "DOWNLOAD_PART_SIZE", 64 * 1024)
    monkeypatch.setattr(_download, "DOWNLOAD_PARALLELS", 4)
    hub.add("bad.bin", b"good" * 75000, served=b"evil" * 75000)
    good = hub.add("next.bin", b"correct tail")
    report = run(api, dirs, file_list=manifest(hub), max_workers=1)
    assert (report.failed, report.succeeded) == (1, 1)
    failed = next(report.iter_files("FAILED"))
    assert failed.error_type == "FileIntegrityError" and failed.retries == 1
    assert len(hub.seen("bad.bin", "cdn")) == 10  # Five ranges per attempt, two complete attempts.
    assert all(request.range for request in hub.seen("bad.bin", "cdn"))
    assert not list(dirs[0].glob("bad.bin*"))
    assert (dirs[0] / "next.bin").read_bytes() == good
    report.close()


def test_stalled_headers_use_short_policy_and_no_nested_retries(hub, api, dirs):
    hub.add("stall.bin", b"x", header_delay=2)
    hub.add("good.bin", b"good")
    policy = DownloadRetryPolicy(max_retries=1, idle_timeout=0.1, recovery_timeout=0.4)
    report = run(api, dirs, retry_policy=policy, max_workers=1)
    assert report.deferred == 1
    assert report.succeeded == 1
    assert len(hub.seen("stall.bin", "cdn")) == 2
    assert next(report.iter_files("DEFERRED")).retries == 1


def test_invalid_metadata_and_paths_are_isolated_without_network(hub, api, dirs, tmp_path):
    hub.add("good.bin", b"data")
    entries = [
        {"Path": "missing.bin"},
        {"Path": "negative.bin", "Size": -1},
        {"Path": "../escape.bin", "Size": 1},
        {"Path": "bad-hash.bin", "Sha256": "invalid"},
        *manifest(hub),
    ]
    report = run(api, dirs, file_list=iter(entries))
    assert (report.failed, report.succeeded) == (4, 1)
    assert not (tmp_path / "escape.bin").exists()
    assert {request.path for request in hub.requests} == {"good.bin"}


def test_pagination_downloads_page_one_before_page_two_finishes(hub, api, dirs, monkeypatch):
    hub.add("a.bin", b"a")
    hub.add("b.bin", b"b")
    entries = manifest(hub)

    def pages(*args, **kwargs):
        yield 1, entries[:1], None
        limit = time.monotonic() + 2
        while not (dirs[0] / "a.bin").exists() and time.monotonic() < limit:
            time.sleep(0.01)
        assert (dirs[0] / "a.bin").exists()
        yield 2, [], NetworkError("page missing")
        yield 3, entries[1:], None

    monkeypatch.setattr(api.legacy, "iter_dataset_file_pages", pages)
    report = run(api, dirs)
    assert report.succeeded == 2
    assert not report.listing_complete
    assert "page=2" in report.listing_errors[0]


def test_manifest_iterable_is_bounded_and_not_materialized(hub, api, dirs):
    hub.add("x.bin", b"x")

    def entries():
        yield manifest(hub)[0]
        limit = time.monotonic() + 2
        while not (dirs[0] / "x.bin").exists() and time.monotonic() < limit:
            time.sleep(0.01)
        assert (dirs[0] / "x.bin").exists()

    assert run(api, dirs, file_list=entries()).complete


def test_cancelled_run_has_durable_pending_state_and_is_resumable(hub, api, dirs, monkeypatch):
    import modelscope_hub._resilient_download as module

    hub.add("x.bin", b"x" * 100, header_delay=2)
    entered = threading.Event()
    original_wait = module.wait

    def interrupt(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            time.sleep(0.05)
            raise KeyboardInterrupt
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(module, "wait", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run(api, dirs)
    reports = list((dirs[1] / ".download_state").glob("*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["report"]["status"] == "CANCELLED"
    hub.files["x.bin"].header_delay = 0
    assert run(api, dirs).complete


def test_crashed_recovery_does_not_receive_a_fresh_automatic_budget(hub, api, dirs):
    hub.add("x.bin", b"x", served=b"")
    first = run(api, dirs)
    ledger = Path(first.tracker_path)
    data = json.loads(ledger.read_text())
    data["files"]["x.bin"]["status"] = "DOWNLOADING"
    ledger.write_text(json.dumps(data))
    again = run(api, dirs)
    assert again.skipped_failures == 1
    assert again.deferred == 1
    assert len(hub.seen("x.bin", "cdn")) == 2


def test_cli_partial_exit_three_and_strict_mode_unchanged(hub, dirs, monkeypatch):
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "")
    hub.add("bad.bin", b"data", served=b"")
    hub.add("good.bin", b"good")
    args = [
        "--endpoint",
        hub.url,
        "download",
        REPO,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(dirs[0]),
        "--cache-dir",
        str(dirs[1]),
    ]
    assert run_cmd([*args, "--continue-on-error"]) == 3
    assert (dirs[0] / "good.bin").exists()
    assert run_cmd([*args, "--continue-on-error"]) == 3
    assert len(hub.seen("bad.bin", "cdn")) == 2
    assert run_cmd(args) == 1


@pytest.mark.parametrize("name", ["snapshot_download", "dataset_snapshot_download"])
def test_compat_continue_mode_is_explicit_and_survives_new_instances(hub, dirs, name):
    module = importlib.import_module("modelscope_hub.compat.snapshot_download")
    function = getattr(module, name)
    hub.add("bad.bin", b"data", served=b"")
    hub.add("good.bin", b"good")
    kwargs = dict(endpoint=hub.url, token="", local_dir=str(dirs[0]), cache_dir=str(dirs[1]), continue_on_error=True)
    if name == "snapshot_download":
        kwargs["repo_type"] = "dataset"
    assert function(REPO, **kwargs) == str(dirs[0])
    assert function(REPO, **kwargs) == str(dirs[0])
    assert len(hub.seen("bad.bin", "cdn")) == 2


def test_bad_listing_page_is_durably_skipped_and_does_not_hide_later_pages(hub, api, dirs, monkeypatch):
    hub.add("a.bin", b"a")
    hub.add("b.bin", b"b")
    entries = manifest(hub)
    calls = []

    def request(path, params, **kwargs):
        page = params["PageNumber"]
        calls.append(page)
        if page == 2:
            raise NetworkError("missing page")
        response = type(
            "Response",
            (),
            {"json": lambda self: {"Data": {"Files": [entries[0]] * 200 if page == 1 else [entries[1]]}}},
        )()
        return response

    monkeypatch.setattr(api.legacy, "_request_repo_tree", request)
    first = run(api, dirs)
    assert first.succeeded == 2
    assert not first.listing_complete
    assert calls == [1, 2, 3]
    calls.clear()
    again = run(api, dirs)
    assert not again.listing_complete
    assert calls == [1, 3]


def test_report_header_trickle_is_stopped_without_waiting_for_whole_file_deadline(hub, api, dirs):
    hub.add("slow.bin", b"data", header_trickle=True)
    hub.add("next.bin", b"next")
    started = time.monotonic()
    report = run(api, dirs, retry_policy=DownloadRetryPolicy(max_retries=0, idle_timeout=0.15))
    assert time.monotonic() - started < 1
    assert report.deferred == 1
    assert report.succeeded == 1


def test_duplicates_cannot_reset_an_active_files_metadata(hub, api, dirs):
    hub.add("x.bin", b"data")
    entry = manifest(hub)[0]
    report = run(api, dirs, file_list=[entry, dict(entry, Size=999), entry])
    assert report.total == 1
    assert report.succeeded == 1
    assert len(hub.seen("x.bin", "cdn")) == 1


def test_business_listing_failure_is_not_an_empty_complete_dataset(hub, api, dirs, monkeypatch):
    from unittest.mock import MagicMock

    response = MagicMock()
    response.json.return_value = {"Success": False, "Code": 500}
    monkeypatch.setattr(api.legacy, "_request_repo_tree", lambda *args, **kwargs: response)
    report = run(api, dirs)
    assert not report.complete
    assert not report.listing_complete
    assert len(report.listing_errors) == 2
    assert report.total == 0


def test_healthy_files_do_not_write_persistent_records_per_file(hub, api, dirs, monkeypatch):
    from modelscope_hub._download_state import _DownloadState

    writes = []
    original_save = _DownloadState._save

    def saved(state):
        writes.append(1)
        original_save(state)

    monkeypatch.setattr(_DownloadState, "_save", saved)
    for i in range(30):
        hub.add(f"file-{i}.bin", b"good")
    report = run(api, dirs)
    assert report.complete
    assert writes == [1, 1]  # One scope header and one final summary, independent of file count.
    data = json.loads(Path(report.tracker_path).read_text())
    assert data["files"] == {}
    assert Path(report.tracker_path).name.startswith(".")
    assert len(list(report.iter_files("VERIFIED"))) == 30
    run(api, dirs)
    assert writes == [1, 1, 1]  # A rerun only replaces the latest summary.


def test_repaired_file_overrides_stale_failure_hint_without_network(hub, api, dirs):
    content = hub.add("bad.bin", b"valid data", served=b"")
    first = run(api, dirs)
    assert first.failed == 1
    (dirs[0] / "bad.bin").write_bytes(content)
    repaired = run(api, dirs)
    assert repaired.complete
    assert repaired.skipped_failures == 0
    assert len(hub.seen("bad.bin", "cdn")) == 2
    assert json.loads(Path(repaired.tracker_path).read_text())["files"] == {}


def test_deleted_state_does_not_redownload_valid_files(hub, api, dirs):
    hub.add("good.bin", b"valid data")
    first = run(api, dirs)
    Path(first.tracker_path).unlink()
    again = run(api, dirs)
    assert again.complete
    assert len(hub.seen("good.bin", "cdn")) == 1


@pytest.mark.parametrize("damaged", ['{"version":', '{"version": 1, "identity": null}', "\ufffd"])
def test_corrupt_state_warns_and_revalidates_actual_files(hub, api, dirs, damaged):
    hub.add("good.bin", b"valid data")
    first = run(api, dirs)
    Path(first.tracker_path).write_text(damaged)
    again = run(api, dirs)
    assert again.complete
    assert again.state_warnings
    assert len(hub.seen("good.bin", "cdn")) == 1


def test_success_hint_cannot_hide_same_size_corruption(hub, api, dirs):
    hub.add("good.bin", b"good")
    first = run(api, dirs)
    (dirs[0] / "good.bin").write_bytes(b"evil")
    data = json.loads(Path(first.tracker_path).read_text())
    data["files"]["good.bin"] = asdict(next(first.iter_files()))
    Path(first.tracker_path).write_text(json.dumps(data))
    again = run(api, dirs)
    assert again.complete
    assert again.state_warnings  # Ignore VERIFIED claims in auxiliary state.
    assert (dirs[0] / "good.bin").read_bytes() == b"good"
    assert len(hub.seen("good.bin", "cdn")) == 2


def test_cannot_save_hints_does_not_block_downloads(hub, api, dirs, monkeypatch):
    hub.add("bad.bin", b"data", served=b"")
    hub.add("good.bin", b"good")
    original = Path.replace

    def fail_state_replace(path, target):
        if str(target).endswith(".json") and ".report." not in str(target):
            raise PermissionError("hint file cannot be replaced")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", fail_state_replace)
    report = run(api, dirs)
    assert report.failed == 1
    assert report.succeeded == 1
    assert report.state_warnings
    assert (dirs[0] / "good.bin").read_bytes() == b"good"


def test_missing_state_does_not_promote_a_partial_file_to_verified(hub, api, dirs):
    hub.add("good.bin", b"good")
    dirs[0].mkdir()
    (dirs[0] / "good.bin").write_bytes(b"go")
    report = run(api, dirs)
    assert report.complete
    assert (dirs[0] / "good.bin").read_bytes() == b"good"


def test_import_works_when_embedded_database_modules_are_unavailable():
    import subprocess
    import sys

    code = """
import importlib.abc
import sys
class BlockDatabase(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"sqlite3", "duckdb", "_sqlite3"}:
            raise ImportError("Embedded databases are unavailable")
sys.meta_path.insert(0, BlockDatabase())
import modelscope_hub._download_state
import modelscope_hub._resilient_download
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_single_json_is_reused_across_failure_restart_and_repair(hub, api, dirs, tmp_path):
    hub.add("bad.bin", b"data", served=b"")
    hub.add("good.bin", b"good")
    task = tmp_path / "task"
    tracker = task / ".ms_download_state.json"
    first = run(api, dirs, tracker_path=tracker)
    second = run(api, dirs, tracker_path=tracker)
    repaired = run(api, dirs, tracker_path=tracker, retry_failed=True)
    assert first.failed == second.failed == repaired.failed == 1
    hub.files["bad.bin"].served = None
    final = run(api, dirs, tracker_path=tracker, retry_failed=True)
    assert final.complete
    assert all(report.report_path == report.tracker_path == str(tracker) for report in (first, second, final))
    assert final.results_path == ""
    assert list(task.glob("*.json")) == [tracker]
    assert not list(task.glob("*.jsonl"))
    assert not list(task.glob("*.tmp"))
    data = json.loads(tracker.read_text())
    assert data["files"] == {} and data["pages"] == {}
    assert data["report"]["succeeded"] == 2
    assert data["report"]["status"] == "COMPLETE"
    assert [result.status for result in first.iter_files("FAILED")] == ["FAILED"]
    assert len(list(final.iter_files("VERIFIED"))) == 2
    assert "_file_results" not in asdict(final)


@pytest.mark.parametrize("reset", [False, True])
def test_explicit_manifest_recovery_resets_selected_page_hints(hub, api, dirs, reset):
    hub.add("good.bin", b"good")
    first = run(api, dirs)
    tracker = Path(first.tracker_path)
    data = json.loads(tracker.read_text())
    data["pages"] = {"/": {"13": "Previous page timeout"}}
    tracker.write_text(json.dumps(data))
    recovered = run(api, dirs, file_list=manifest(hub), retry_failed=reset)
    assert recovered.complete
    assert json.loads(tracker.read_text())["pages"] == ({} if reset else data["pages"])
    assert len(hub.seen("good.bin", "cdn")) == 1


def test_result_iterators_are_independent_and_close_releases_spool(hub, api, dirs):
    for i in range(3):
        hub.add(f"{i}.bin", b"good")
    report = run(api, dirs)
    left, right = report.iter_files(), report.iter_files()
    assert next(left).path == next(right).path
    assert [entry.path for entry in left] == [entry.path for entry in right]
    assert len(list(report.iter_files())) == 3
    spool = report._file_results
    report.close()
    report.close()
    assert spool.closed
    with pytest.raises(ValueError, match="open live DownloadReport"):
        list(report.iter_files())


def test_persisted_summary_cannot_be_used_as_verified_file_evidence(hub, api, dirs):
    hub.add("good.bin", b"good")
    report = run(api, dirs)
    summary = json.loads(Path(report.report_path).read_text())["report"]
    restored = DownloadReport.from_dict(summary)
    assert restored.complete
    with pytest.raises(ValueError, match="not a verified-file manifest"):
        list(restored.iter_files("VERIFIED"))


def test_report_types_and_policy_are_public():
    assert not DownloadReport().complete
    assert asdict(DownloadRetryPolicy())["max_retries"] == 2
    with pytest.raises(ValueError):
        DownloadRetryPolicy(idle_timeout=0)
