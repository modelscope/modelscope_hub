"""Commit-batch planning, LFS routing and commit throttling behaviour."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import modelscope_hub._upload as upload_module
from modelscope_hub._upload import (
    BatchTracker,
    UploadManager,
    _calculate_adaptive_batch_size,
    _is_inline_metadata,
    _normalize_path_in_repo,
    _plan_commit_batches,
    _upload_mode,
)
from modelscope_hub.errors import InvalidParameter, NetworkError, RateLimitError


def _make_manager() -> tuple[UploadManager, MagicMock]:
    client = MagicMock()
    client.create_commit.return_value = {"ok": True}

    def validate_blobs(*, repo_id: str, repo_type: str, objects: list[dict]) -> dict[str, str]:
        return {obj["oid"]: f"https://upload/{obj['oid']}" for obj in objects}

    def upload_blob(*, upload_url: str, data, size: int) -> None:
        while data.read(1024 * 1024):
            pass

    client.validate_blobs.side_effect = validate_blobs
    client.upload_blob.side_effect = upload_blob
    return UploadManager(client, MagicMock()), client


# ---------------------------------------------------------------- batch sizing


@pytest.mark.parametrize(
    ("total_files", "cap", "expected"),
    [
        (0, 256, 1),
        (1, 256, 1),
        (100, 256, 100),
        (2_000, 256, 256),
        (60_000, 256, 256),
        (60_000, 512, 512),
        # "No cap" still respects the server's action ceiling, not the file count.
        (5_000, 0, 2_000),
        (1_500, 0, 1_500),
        # An explicit cap above the ceiling cannot be honored either.
        (60_000, 4_096, 2_000),
    ],
)
def test_adaptive_batch_size_fills_up_to_the_cap(total_files: int, cap: int, expected: int) -> None:
    assert _calculate_adaptive_batch_size(total_files, cap) == expected


def test_adaptive_batch_size_is_monotonic_and_never_exceeds_cap() -> None:
    sizes = [_calculate_adaptive_batch_size(n, 256) for n in range(1, 3_000)]
    assert sizes == sorted(sizes)
    assert max(sizes) == 256


def test_batch_plan_closes_on_inline_bytes_before_operation_cap(tmp_path: Path) -> None:
    # 8 inline files of 100 KiB: the operation cap would take all 8 in one
    # commit, but their base64 form exceeds a 512 KiB inline budget.
    files = []
    sizes = {}
    for index in range(8):
        path = tmp_path / f"part-{index}.txt"
        path.write_bytes(b"x" * (100 * 1024))
        files.append((f"part-{index}.txt", str(path)))
        sizes[str(path)] = 100 * 1024

    plan = _plan_commit_batches(
        files,
        "dataset",
        max_operations=256,
        max_inline_bytes=512 * 1024,
        sizes=sizes,
    )

    assert sum(plan) == 8
    assert max(plan) < 8
    encoded_per_file = (100 * 1024 + 2) // 3 * 4
    assert all(count * encoded_per_file <= 512 * 1024 for count in plan)


def test_batch_plan_ignores_lfs_bytes_in_the_inline_budget(tmp_path: Path) -> None:
    # .parquet is an LFS suffix for datasets, so these contribute a pointer
    # rather than inline content and must not split the batch.
    files = []
    sizes = {}
    for index in range(8):
        path = tmp_path / f"shard-{index}.parquet"
        files.append((f"shard-{index}.parquet", str(path)))
        sizes[str(path)] = 100 * 1024

    plan = _plan_commit_batches(
        files,
        "dataset",
        max_operations=256,
        max_inline_bytes=512 * 1024,
        sizes=sizes,
    )

    assert plan == [8]


def test_batch_plan_keeps_one_oversized_inline_file_per_batch(tmp_path: Path) -> None:
    # Below the 1 MiB LFS threshold, so these really do travel inline; each one
    # alone blows the inline budget, which must not stall the plan.
    files = []
    sizes = {}
    for index in range(3):
        path = tmp_path / f"big-{index}.txt"
        files.append((f"big-{index}.txt", str(path)))
        sizes[str(path)] = 512 * 1024

    plan = _plan_commit_batches(
        files,
        "dataset",
        max_operations=256,
        max_inline_bytes=1024,
        sizes=sizes,
    )

    assert plan == [1, 1, 1]


def test_batch_tracker_handles_uneven_batches() -> None:
    tracker = BatchTracker(7, [3, 1, 3])

    assert tracker.num_batches == 3
    assert tracker.batch_range(0) == (0, 3)
    assert tracker.batch_range(1) == (3, 4)
    assert tracker.batch_range(2) == (4, 7)
    assert [tracker.batch_index(i) for i in range(7)] == [0, 0, 0, 1, 2, 2, 2]


def test_batch_tracker_absorbs_a_short_plan() -> None:
    tracker = BatchTracker(5, [2])

    assert tracker.num_batches == 2
    assert tracker.batch_range(1) == (2, 5)


def test_batch_tracker_accepts_a_uniform_size() -> None:
    tracker = BatchTracker(5, 2)

    assert tracker.num_batches == 3
    assert [tracker.batch_index(i) for i in range(5)] == [0, 0, 1, 1, 2]


# ------------------------------------------------------- inline metadata gate


@pytest.mark.parametrize(
    "path",
    ["README.md", "nested/README.md", ".gitattributes", "configuration.json", "sub/dir/config.json"],
)
def test_inline_metadata_is_recognized_at_any_depth(path: str) -> None:
    assert _is_inline_metadata(path)


def test_metadata_stays_inline_even_below_a_zero_threshold(monkeypatch) -> None:
    monkeypatch.setattr(upload_module, "UPLOAD_LFS_FORCE_THRESHOLD_BYTES", 0)

    assert _upload_mode("README.md", 500 * 1024, "dataset") == "normal"
    assert _upload_mode("configuration.json", 10, "model") == "normal"
    # Everything that is not repository metadata does move to LFS.
    assert _upload_mode("data/sample.txt", 1, "dataset") == "lfs"


def test_small_files_move_to_lfs_when_the_threshold_is_lowered(monkeypatch) -> None:
    monkeypatch.setattr(upload_module, "UPLOAD_LFS_FORCE_THRESHOLD_BYTES", 32 * 1024)

    assert _upload_mode("data/sample.txt", 115 * 1024, "dataset") == "lfs"
    # A small card or config still rides inline at this threshold.
    assert _upload_mode("data/notes.txt", 2 * 1024, "dataset") == "normal"


def test_upload_folder_routes_small_files_to_blob_storage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(upload_module, "UPLOAD_LFS_FORCE_THRESHOLD_BYTES", 32 * 1024)
    manager, client = _make_manager()
    (tmp_path / "README.md").write_bytes(b"# card")
    for index in range(4):
        (tmp_path / f"sample-{index}.txt").write_bytes(bytes([index]) * (64 * 1024))

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="dataset",
        folder_path=tmp_path,
        max_workers=2,
        use_cache=False,
        disable_tqdm=True,
    )

    operations = client.create_commit.call_args.kwargs["operations"]
    modes = {op["path"]: op["type"] for op in operations}
    assert modes == {
        "README.md": "normal",
        "sample-0.txt": "lfs",
        "sample-1.txt": "lfs",
        "sample-2.txt": "lfs",
        "sample-3.txt": "lfs",
    }
    # The four data files were pre-signed together, not one request each.
    assert client.validate_blobs.call_count == 1
    assert len(client.validate_blobs.call_args.kwargs["objects"]) == 4
    assert client.upload_blob.call_count == 4


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        (".", ""),
        ("./", ""),
        ("/", ""),
        ("  .  ", ""),
        ("sub", "sub"),
        ("./sub", "sub"),
        ("sub/", "sub"),
        ("/sub/", "sub"),
        ("a/./b", "a/b"),
        ("a/../b", "b"),
        ("data\\shard", "data/shard"),
    ],
)
def test_normalize_path_in_repo_collapses_root_aliases(raw: str, expected: str) -> None:
    assert _normalize_path_in_repo(raw) == expected


@pytest.mark.parametrize("escaping", ["..", "../x", "a/../../b"])
def test_normalize_path_in_repo_refuses_escaping_root(escaping: str) -> None:
    with pytest.raises(InvalidParameter):
        _normalize_path_in_repo(escaping)


def test_upload_folder_dot_path_in_repo_commits_root_relative_paths(tmp_path: Path, monkeypatch) -> None:
    # `ms upload REPO LOCAL .` passes path_in_repo=".". Left literal it became a
    # "./" prefix on every commit action path, which the Hub rejects wholesale
    # as an invalid commit action (E3021). "." must map to the repo root.
    manager, client = _make_manager()
    (tmp_path / "README.md").write_bytes(b"# card")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train.txt").write_bytes(b"rows")

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="dataset",
        folder_path=tmp_path,
        path_in_repo=".",
        max_workers=2,
        use_cache=False,
        disable_tqdm=True,
    )

    operations = client.create_commit.call_args.kwargs["operations"]
    assert sorted(op["path"] for op in operations) == ["README.md", "data/train.txt"]
    assert not any(op["path"].startswith("./") for op in operations)


def test_upload_folder_batch_presigns_in_configured_group_size(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(upload_module, "UPLOAD_LFS_FORCE_THRESHOLD_BYTES", 0)
    monkeypatch.setattr(upload_module, "UPLOAD_BLOB_VALIDATION_BATCH_MAX_OBJECTS", 3)
    manager, client = _make_manager()
    for index in range(7):
        (tmp_path / f"sample-{index}.txt").write_bytes(bytes([index]) * 128)

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="dataset",
        folder_path=tmp_path,
        max_workers=2,
        use_cache=False,
        disable_tqdm=True,
    )

    assert client.validate_blobs.call_count == 3
    assert client.upload_blob.call_count == 7


def test_presign_groups_run_concurrently(tmp_path: Path, monkeypatch) -> None:
    # The groups are independent requests; serialising them made the pre-sign
    # phase scale with the file count in pure round-trip latency.
    monkeypatch.setattr(upload_module, "UPLOAD_BLOB_VALIDATION_BATCH_MAX_OBJECTS", 1)
    manager, client = _make_manager()
    peak = 0
    inflight = 0
    lock = threading.Lock()

    def validate_blobs(*, repo_id: str, repo_type: str, objects: list[dict]) -> dict[str, str]:
        nonlocal peak, inflight
        with lock:
            inflight += 1
            peak = max(peak, inflight)
        time.sleep(0.05)
        with lock:
            inflight -= 1
        return {obj["oid"]: f"https://upload/{obj['oid']}" for obj in objects}

    client.validate_blobs.side_effect = validate_blobs
    objects = [{"oid": f"{index:064x}", "size": 1} for index in range(8)]

    result = manager._validate_blobs_batch(repo_id="owner/repo", repo_type="dataset", objects=objects, max_workers=8)

    assert len(result) == 8
    assert peak > 1, "pre-sign groups must overlap"


def test_presign_concurrency_stays_within_the_connection_pool(tmp_path: Path, monkeypatch) -> None:
    # More in-flight requests than pooled connections just trades latency for
    # discarded connections and fresh TLS handshakes.
    monkeypatch.setattr(upload_module, "UPLOAD_BLOB_VALIDATION_BATCH_MAX_OBJECTS", 1)
    monkeypatch.setattr(upload_module, "API_CONNECTION_POOL_MAXSIZE", 2)
    manager, client = _make_manager()
    peak = 0
    inflight = 0
    lock = threading.Lock()

    def validate_blobs(*, repo_id: str, repo_type: str, objects: list[dict]) -> dict[str, str]:
        nonlocal peak, inflight
        with lock:
            inflight += 1
            peak = max(peak, inflight)
        time.sleep(0.02)
        with lock:
            inflight -= 1
        return {obj["oid"]: "https://upload/x" for obj in objects}

    client.validate_blobs.side_effect = validate_blobs
    objects = [{"oid": f"{index:064x}", "size": 1} for index in range(12)]

    manager._validate_blobs_batch(repo_id="owner/repo", repo_type="dataset", objects=objects, max_workers=32)

    assert peak <= 2


def test_presign_normalizes_unmentioned_objects_as_existing(monkeypatch) -> None:
    # The batch endpoint answers only about objects that need uploading and
    # returns empty arrays for ones that already exist. Leaving those out of the
    # map would send every file of a re-run back to per-file negotiation.
    monkeypatch.setattr(upload_module, "UPLOAD_BLOB_VALIDATION_BATCH_MAX_OBJECTS", 2)
    manager, client = _make_manager()
    objects = [{"oid": f"{index:064x}", "size": 1} for index in range(4)]
    # Server mentions only the last object; the rest already exist.
    wanted = objects[-1]["oid"]
    client.validate_blobs.side_effect = lambda **kw: (
        {wanted: "https://upload/x"} if any(o["oid"] == wanted for o in kw["objects"]) else {}
    )

    result = manager._validate_blobs_batch(repo_id="owner/repo", repo_type="dataset", objects=objects, max_workers=4)

    assert set(result) == {o["oid"] for o in objects}
    assert result[wanted] == "https://upload/x"
    assert all(result[o["oid"]] is None for o in objects[:-1])


def test_a_failed_presign_group_falls_back_to_per_file_negotiation(tmp_path: Path, monkeypatch) -> None:
    # Losing the optimisation must never escalate into losing the upload -- and
    # critically, an oid missing from the map must not read as "already exists",
    # which would skip the transfer and commit a pointer to a blob that was
    # never stored.
    monkeypatch.setattr(upload_module, "UPLOAD_LFS_FORCE_THRESHOLD_BYTES", 0)
    monkeypatch.setattr(upload_module, "UPLOAD_BLOB_VALIDATION_BATCH_MAX_OBJECTS", 2)
    manager, client = _make_manager()
    for index in range(6):
        (tmp_path / f"blob-{index}.dat").write_bytes(bytes([index]) * 512)

    seen: list[int] = []

    def validate_blobs(*, repo_id: str, repo_type: str, objects: list[dict]) -> dict[str, str]:
        seen.append(len(objects))
        if len(objects) > 1 and len(seen) == 1:
            raise NetworkError("pre-sign group failed")
        return {obj["oid"]: f"https://upload/{obj['oid']}" for obj in objects}

    client.validate_blobs.side_effect = validate_blobs

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="dataset",
        folder_path=tmp_path,
        max_workers=1,
        use_cache=False,
        disable_tqdm=True,
    )

    # Every file still reached object storage, none was silently treated as reused.
    assert client.upload_blob.call_count == 6
    operations = client.create_commit.call_args.kwargs["operations"]
    assert len(operations) == 6
    assert all(op["type"] == "lfs" and op["sha256"] for op in operations)


@pytest.mark.parametrize("use_cache", [True, False])
def test_pre_hashed_files_are_not_hashed_twice(tmp_path: Path, monkeypatch, use_cache: bool) -> None:
    # Pre-hashing feeds group pre-signing; without threading the digest through
    # to the upload worker it would read and hash every file a second time.
    monkeypatch.setattr(upload_module, "UPLOAD_LFS_FORCE_THRESHOLD_BYTES", 0)
    manager, _ = _make_manager()
    for index in range(5):
        (tmp_path / f"sample-{index}.txt").write_bytes(bytes([index]) * 256)

    calls: list[object] = []
    original = upload_module._compute_file_hash

    def counting_hash(*args, **kwargs):
        calls.append(kwargs.get("file_path_or_obj", args[0] if args else None))
        return original(*args, **kwargs)

    monkeypatch.setattr(upload_module, "_compute_file_hash", counting_hash)

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="dataset",
        folder_path=tmp_path,
        max_workers=2,
        use_cache=use_cache,
        disable_tqdm=True,
    )

    assert len(calls) == 5
    assert len(set(calls)) == 5


# ------------------------------------------------------------- tracker path


def test_tracker_path_keeps_resume_state_outside_the_uploaded_tree(tmp_path: Path) -> None:
    manager, _ = _make_manager()
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "README.md").write_bytes(b"hello")
    cache_path = tmp_path / "state" / "upload.json"
    cache_path.parent.mkdir()

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="model",
        folder_path=staging,
        max_workers=1,
        use_cache=True,
        disable_tqdm=True,
        tracker_path=cache_path,
    )

    assert not (staging / ".ms_upload_cache").exists()
    assert cache_path.exists()
    assert json.loads(cache_path.read_text(encoding="utf-8"))["repo_id"] == "owner/repo"


def test_external_tracker_lets_a_second_run_skip_committed_files(tmp_path: Path) -> None:
    manager, client = _make_manager()
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "README.md").write_bytes(b"hello")
    cache_path = tmp_path / "upload.json"

    for _ in range(2):
        manager.upload_folder(
            repo_id="owner/repo",
            repo_type="model",
            folder_path=staging,
            max_workers=1,
            use_cache=True,
            disable_tqdm=True,
            tracker_path=cache_path,
        )

    # The second run recognises the committed file and issues no new commit.
    assert client.create_commit.call_count == 1


# ------------------------------------------------------------ progress events


def test_progress_callback_reports_every_batch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(upload_module, "UPLOAD_COMMIT_BATCH_MAX_OPERATIONS", 2)
    monkeypatch.setattr(upload_module, "UPLOAD_ADAPTIVE_BATCHING_ENABLED", False)
    manager, _ = _make_manager()
    for index in range(4):
        (tmp_path / f"file-{index}.txt").write_bytes(bytes([index]) * (index + 1))
    events: list[dict] = []

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="model",
        folder_path=tmp_path,
        max_workers=1,
        use_cache=False,
        disable_tqdm=True,
        progress_callback=events.append,
    )

    commits = [e for e in events if e["event"] == "batch_committed"]
    assert [e["batch_index"] for e in commits] == [0, 1]
    assert commits[-1]["committed_files"] == 4
    assert commits[-1]["num_batches"] == 2
    assert all(e["total_files"] == 4 for e in commits)


def test_progress_events_carry_byte_counts(tmp_path: Path, monkeypatch) -> None:
    # A consumer that only learns file counts cannot derive a throughput rate or
    # an ETA, which is most of what a progress feed is for.
    monkeypatch.setattr(upload_module, "UPLOAD_COMMIT_BATCH_MAX_OPERATIONS", 2)
    monkeypatch.setattr(upload_module, "UPLOAD_ADAPTIVE_BATCHING_ENABLED", False)
    manager, _ = _make_manager()
    expected_total = 0
    for index in range(4):
        payload = bytes([index]) * (1024 * (index + 1))
        (tmp_path / f"file-{index}.txt").write_bytes(payload)
        expected_total += len(payload)
    events: list[dict] = []

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="model",
        folder_path=tmp_path,
        max_workers=1,
        use_cache=False,
        disable_tqdm=True,
        progress_callback=events.append,
    )

    commits = [e for e in events if e["event"] == "batch_committed"]
    assert all(e["total_bytes"] == expected_total for e in commits)
    assert sum(e["batch_bytes"] for e in commits) == expected_total
    assert commits[-1]["committed_bytes"] == expected_total
    # Cumulative counters advance monotonically alongside the per-batch ones.
    assert commits[0]["committed_bytes"] == commits[0]["batch_bytes"]
    # These files ride inline, so all of their volume is attributed to the commit
    # rather than to an earlier object-storage transfer.
    assert sum(e["batch_inline_bytes"] for e in commits) == expected_total


def test_wire_progress_tracks_object_storage_transfers(tmp_path: Path, monkeypatch) -> None:
    # Commits land in lumps, so a rate built from them alone reads as a spike
    # followed by zero. Blob uploads finish continuously and are the honest
    # source; only bytes that really moved are counted.
    monkeypatch.setattr(upload_module, "UPLOAD_LFS_FORCE_THRESHOLD_BYTES", 0)
    monkeypatch.setattr(upload_module, "UPLOAD_PROGRESS_MIN_INTERVAL_SECONDS", 0)
    manager, _ = _make_manager()
    expected_total = 0
    for index in range(6):
        payload = bytes([index]) * (2048 * (index + 1))
        (tmp_path / f"blob-{index}.dat").write_bytes(payload)
        expected_total += len(payload)
    events: list[dict] = []

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="dataset",
        folder_path=tmp_path,
        max_workers=2,
        use_cache=False,
        disable_tqdm=True,
        progress_callback=events.append,
    )

    wire = [e for e in events if e["event"] == "upload_progress"]
    assert wire, "per-file transfers must report progress before the commit lands"
    # Cumulative and never decreasing.
    assert [e["uploaded_bytes"] for e in wire] == sorted(e["uploaded_bytes"] for e in wire)
    assert wire[-1]["uploaded_bytes"] == expected_total
    assert wire[-1]["uploaded_files"] == 6
    # Deltas partition the total exactly, so a consumer can sum them into a rate.
    assert sum(e["uploaded_bytes_delta"] for e in wire) == expected_total
    # Nothing rode inline, so the commit attributes no wire traffic to itself.
    commits = [e for e in events if e["event"] == "batch_committed"]
    assert sum(e["batch_inline_bytes"] for e in commits) == 0


def test_reused_blobs_report_no_wire_bytes(tmp_path: Path, monkeypatch) -> None:
    # A deduplicated blob transfers nothing; counting its size would inflate the
    # reported throughput above what the link actually carried.
    monkeypatch.setattr(upload_module, "UPLOAD_LFS_FORCE_THRESHOLD_BYTES", 0)
    monkeypatch.setattr(upload_module, "UPLOAD_PROGRESS_MIN_INTERVAL_SECONDS", 0)
    manager, client = _make_manager()
    client.validate_blobs.side_effect = lambda **kwargs: {obj["oid"]: None for obj in kwargs["objects"]}
    for index in range(4):
        (tmp_path / f"blob-{index}.dat").write_bytes(bytes([index]) * 4096)
    events: list[dict] = []

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="dataset",
        folder_path=tmp_path,
        max_workers=2,
        use_cache=False,
        disable_tqdm=True,
        progress_callback=events.append,
    )

    client.upload_blob.assert_not_called()
    wire = [e for e in events if e["event"] == "upload_progress"]
    assert all(e["uploaded_bytes"] == 0 for e in wire)
    assert wire[-1]["uploaded_files"] == 4
    # The files are still committed, and the commit still reports their volume.
    commits = [e for e in events if e["event"] == "batch_committed"]
    assert commits[-1]["committed_bytes"] == 4 * 4096


def test_recovered_files_are_reported_so_totals_stay_complete(tmp_path: Path, monkeypatch) -> None:
    # Recovery is exactly when an operator is watching. An 8 GiB run that lost one
    # 512-file batch to a rejected commit put all 40000 files on the Hub but left
    # 91 MB missing from done_bytes, because the recovery path emitted nothing.
    monkeypatch.setattr(upload_module, "UPLOAD_COMMIT_BATCH_MAX_OPERATIONS", 2)
    monkeypatch.setattr(upload_module, "UPLOAD_ADAPTIVE_BATCHING_ENABLED", False)
    monkeypatch.setattr(upload_module.time, "sleep", lambda _s: None)
    manager, client = _make_manager()
    expected_total = 0
    for index in range(4):
        payload = bytes([index]) * (1024 * (index + 1))
        (tmp_path / f"file-{index}.txt").write_bytes(payload)
        expected_total += len(payload)
    events: list[dict] = []

    # Exhaust every in-commit attempt for the first batch so it really falls
    # through to the recovery path instead of being absorbed earlier. The attempt
    # budget is a default argument, so it is spelled out rather than patched.
    attempts = upload_module.UPLOAD_COMMIT_MAX_ATTEMPTS
    calls = {"n": 0}

    def create_commit(**kwargs):
        calls["n"] += 1
        if calls["n"] <= attempts:
            raise NetworkError("commit rejected")
        return {"ok": True}

    client.create_commit.side_effect = create_commit

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="model",
        folder_path=tmp_path,
        max_workers=1,
        use_cache=False,
        disable_tqdm=True,
        progress_callback=events.append,
    )

    recovery = [e for e in events if e["event"] == "recovery_committed"]
    assert recovery, "a recovered commit must report its progress"
    accounted = [e for e in events if e["event"] in ("batch_committed", "recovery_committed")]
    # Every byte is accounted for exactly once across the two commit paths.
    assert sum(e["batch_bytes"] for e in accounted) == expected_total
    assert accounted[-1]["committed_bytes"] == expected_total
    assert accounted[-1]["committed_files"] == 4


def test_progress_callback_failure_does_not_abort_the_upload(tmp_path: Path) -> None:
    manager, client = _make_manager()
    (tmp_path / "README.md").write_bytes(b"hello")

    def explode(_event: dict) -> None:
        raise RuntimeError("reporter is broken")

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="model",
        folder_path=tmp_path,
        max_workers=1,
        use_cache=False,
        disable_tqdm=True,
        progress_callback=explode,
    )

    client.create_commit.assert_called_once()


# ------------------------------------------------------------ commit throttle


def test_commit_honors_server_retry_after(monkeypatch) -> None:
    manager, client = _make_manager()
    slept: list[float] = []
    monkeypatch.setattr(upload_module.time, "sleep", slept.append)
    client.create_commit.side_effect = [
        RateLimitError("commit budget exhausted", retry_after=42),
        {"ok": True},
    ]

    result = manager._commit_with_retry(
        repo_id="owner/repo",
        repo_type="model",
        operations=[{"action": "create", "path": "a.txt"}],
        commit_message="retry",
    )

    assert result == {"ok": True}
    assert slept == [42.0]


def test_commit_retry_after_above_the_ceiling_aborts(monkeypatch) -> None:
    manager, client = _make_manager()
    slept: list[float] = []
    monkeypatch.setattr(upload_module.time, "sleep", slept.append)
    monkeypatch.setattr(upload_module, "UPLOAD_COMMIT_MAX_RETRY_AFTER_SECONDS", 60)
    client.create_commit.side_effect = RateLimitError("commit budget exhausted", retry_after=3600)

    with pytest.raises(RateLimitError):
        manager._commit_with_retry(
            repo_id="owner/repo",
            repo_type="model",
            operations=[{"action": "create", "path": "a.txt"}],
            commit_message="retry",
        )

    assert slept == []


def test_throttled_wait_does_not_consume_the_transient_error_budget(monkeypatch) -> None:
    # A long, server-declared wait must not exhaust the allowance reserved for
    # failures of unknown duration, or a throttle would look like a hard error.
    manager, client = _make_manager()
    monkeypatch.setattr(upload_module, "UPLOAD_COMMIT_RETRY_TOTAL_WAIT_SECONDS", 10)
    monkeypatch.setattr(upload_module.time, "sleep", lambda _s: None)
    client.create_commit.side_effect = [
        RateLimitError("throttled", retry_after=600),
        RateLimitError("throttled", retry_after=600),
        {"ok": True},
    ]

    assert manager._commit_with_retry(
        repo_id="owner/repo",
        repo_type="model",
        operations=[{"action": "create", "path": "a.txt"}],
        commit_message="retry",
        max_attempts=4,
    ) == {"ok": True}


def test_commit_rate_governor_paces_once_the_budget_is_spent(monkeypatch) -> None:
    governor = upload_module._CommitRateGovernor(2)
    clock = {"now": 0.0}
    slept: list[float] = []
    monkeypatch.setattr(upload_module.time, "monotonic", lambda: clock["now"])

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(upload_module.time, "sleep", fake_sleep)

    assert governor.acquire() == 0.0
    assert governor.acquire() == 0.0
    waited = governor.acquire()

    assert slept, "the third commit within the window must wait"
    assert waited == pytest.approx(3600.0, abs=1.0)


def test_commit_rate_governor_is_inert_when_disabled() -> None:
    governor = upload_module._CommitRateGovernor(0)

    assert not governor.enabled
    assert [governor.acquire() for _ in range(50)] == [0.0] * 50


def test_every_commit_path_goes_through_the_shared_governor(tmp_path: Path, monkeypatch) -> None:
    # The budget belongs to the repository, not to one call. Pacing only the
    # happy path would leave recovery rounds free to hammer a throttled server.
    monkeypatch.setattr(upload_module, "UPLOAD_COMMIT_MAX_PER_HOUR", 500)
    monkeypatch.setattr(upload_module, "UPLOAD_COMMIT_BATCH_MAX_OPERATIONS", 2)
    monkeypatch.setattr(upload_module, "UPLOAD_ADAPTIVE_BATCHING_ENABLED", False)
    manager, client = _make_manager()
    assert manager._commit_governor.enabled

    acquired: list[int] = []
    original_acquire = manager._commit_governor.acquire
    monkeypatch.setattr(
        manager._commit_governor,
        "acquire",
        lambda: (acquired.append(1), original_acquire())[1],
    )

    for index in range(4):
        (tmp_path / f"file-{index}.txt").write_bytes(bytes([index]))

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="model",
        folder_path=tmp_path,
        max_workers=1,
        use_cache=False,
        disable_tqdm=True,
    )
    manager.upload_file(
        repo_id="owner/repo",
        repo_type="model",
        path_or_fileobj=b"single",
        path_in_repo="extra.txt",
        disable_tqdm=True,
    )

    # Two batch commits plus the single-file commit.
    assert client.create_commit.call_count == 3
    assert len(acquired) == 3


def test_upload_file_retries_a_transient_commit_failure(monkeypatch) -> None:
    manager, client = _make_manager()
    monkeypatch.setattr(upload_module.time, "sleep", lambda _s: None)
    client.create_commit.side_effect = [
        NetworkError("commit temporarily unavailable"),
        {"ok": True},
    ]

    result = manager.upload_file(
        repo_id="owner/repo",
        repo_type="model",
        path_or_fileobj=b"hello",
        path_in_repo="README.md",
        disable_tqdm=True,
    )

    assert result == {"ok": True}
    assert client.create_commit.call_count == 2


def test_upload_file_honors_retry_after(monkeypatch) -> None:
    manager, client = _make_manager()
    slept: list[float] = []
    monkeypatch.setattr(upload_module.time, "sleep", slept.append)
    client.create_commit.side_effect = [
        RateLimitError("commit budget exhausted", retry_after=7),
        {"ok": True},
    ]

    manager.upload_file(
        repo_id="owner/repo",
        repo_type="model",
        path_or_fileobj=b"hello",
        path_in_repo="README.md",
        disable_tqdm=True,
    )

    assert slept == [7.0]


def test_upload_file_still_fails_fast_on_a_permanent_error() -> None:
    manager, client = _make_manager()
    client.create_commit.side_effect = InvalidParameter("path is not allowed")

    with pytest.raises(InvalidParameter):
        manager.upload_file(
            repo_id="owner/repo",
            repo_type="model",
            path_or_fileobj=b"hello",
            path_in_repo="README.md",
            disable_tqdm=True,
        )

    assert client.create_commit.call_count == 1


def test_prepare_upload_folder_reports_sizes_for_reuse(tmp_path: Path) -> None:
    # Planning needs these sizes immediately afterwards; re-stating the tree
    # costs one syscall per file and yields nothing new.
    manager, _ = _make_manager()
    (tmp_path / "a.txt").write_bytes(b"x" * 10)
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "b.txt").write_bytes(b"y" * 20)

    sizes: dict[str, int] = {}
    prepared = manager._prepare_upload_folder(
        folder_path=tmp_path,
        path_in_repo="",
        repo_type="model",
        sizes_out=sizes,
    )

    assert sizes == {str(tmp_path / "a.txt"): 10, str(nested / "b.txt"): 20}
    assert {path for _, path in prepared} == set(sizes)
