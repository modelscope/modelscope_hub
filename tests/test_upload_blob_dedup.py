"""Identical content inside one upload must be transferred once.

The batch pre-sign step asks the server about every distinct oid *before* any
upload starts, so the server cannot answer "already stored" for a duplicate --
it is not stored yet. Verified against the Hub: pre-signing the same fresh oid
twice yields two upload URLs, and only after the blob lands does the endpoint
report it as existing. Without an owner election, every occurrence therefore
PUTs the same bytes.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import modelscope_hub._upload as upload_module
from modelscope_hub._upload import DUPLICATE_BLOB, UploadManager
from modelscope_hub.errors import NetworkError, StorageError


def _make_manager() -> tuple[UploadManager, MagicMock, list[str]]:
    """Manager whose blob PUTs are recorded by oid."""
    client = MagicMock()
    client.create_commit.return_value = {"ok": True}
    put_oids: list[str] = []
    lock = threading.Lock()

    def validate_blobs(*, repo_id: str, repo_type: str, objects: list[dict]) -> dict[str, str]:
        return {obj["oid"]: f"https://upload/{obj['oid']}" for obj in objects}

    def upload_blob(*, upload_url: str, data, size: int) -> None:
        with lock:
            put_oids.append(upload_url.rsplit("/", 1)[-1])
        while data.read(1024 * 1024):
            pass

    client.validate_blobs.side_effect = validate_blobs
    client.upload_blob.side_effect = upload_blob
    return UploadManager(client, MagicMock()), client, put_oids


def _write_duplicates(root: Path, groups: dict[str, int], size: int = 4096) -> int:
    """Write ``{content_tag: copies}``; returns the number of distinct contents."""
    index = 0
    for tag, copies in groups.items():
        payload = tag.encode() * (size // len(tag.encode()))
        for _ in range(copies):
            (root / f"file-{index:04d}.dat").write_bytes(payload)
            index += 1
    return len(groups)


def test_duplicate_content_is_uploaded_once(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(upload_module, "UPLOAD_LFS_FORCE_THRESHOLD_BYTES", 0)
    manager, client, put_oids = _make_manager()
    distinct = _write_duplicates(tmp_path, {"aa": 5, "bb": 3, "cc": 1})
    total_files = 9

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="dataset",
        folder_path=tmp_path,
        max_workers=4,
        use_cache=False,
        disable_tqdm=True,
    )

    assert len(put_oids) == distinct, f"expected one PUT per distinct content, got {len(put_oids)}"
    assert len(set(put_oids)) == distinct
    # All nine files are still committed, each as an LFS pointer.
    operations = client.create_commit.call_args.kwargs["operations"]
    assert len(operations) == total_files
    assert all(op["type"] == "lfs" and op["sha256"] for op in operations)
    # The pointers cover exactly the distinct contents that were uploaded.
    assert {op["sha256"] for op in operations} == set(put_oids)


def test_duplicates_report_no_wire_bytes(tmp_path: Path, monkeypatch) -> None:
    # Only the owner moves bytes, so throughput must not count the copies.
    monkeypatch.setattr(upload_module, "UPLOAD_LFS_FORCE_THRESHOLD_BYTES", 0)
    monkeypatch.setattr(upload_module, "UPLOAD_PROGRESS_MIN_INTERVAL_SECONDS", 0)
    manager, _, put_oids = _make_manager()
    _write_duplicates(tmp_path, {"aa": 4, "bb": 4}, size=8192)
    events: list[dict] = []

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="dataset",
        folder_path=tmp_path,
        max_workers=4,
        use_cache=False,
        disable_tqdm=True,
        progress_callback=events.append,
    )

    wire = [e for e in events if e["event"] == "upload_progress"]
    assert len(put_oids) == 2
    # Two owners at 8192 bytes each; the six copies contribute nothing.
    assert wire[-1]["uploaded_bytes"] == 2 * 8192
    assert wire[-1]["uploaded_files"] == 8


def test_owner_is_the_earliest_file_so_no_commit_precedes_its_blob(tmp_path: Path, monkeypatch) -> None:
    # Correctness rests on the owner never landing in a later batch than a copy:
    # batches commit in order, so a copy in batch j needs its owner in batch <= j.
    monkeypatch.setattr(upload_module, "UPLOAD_LFS_FORCE_THRESHOLD_BYTES", 0)
    monkeypatch.setattr(upload_module, "UPLOAD_COMMIT_BATCH_MAX_OPERATIONS", 2)
    monkeypatch.setattr(upload_module, "UPLOAD_ADAPTIVE_BATCHING_ENABLED", False)
    manager, client, put_oids = _make_manager()
    # Same content in the first and last file: owner in batch 0, copy in batch 2.
    payload = b"x" * 4096
    for name in ("a.dat", "b.dat", "c.dat", "d.dat", "e.dat"):
        (tmp_path / name).write_bytes(payload if name in ("a.dat", "e.dat") else name.encode() * 512)

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="dataset",
        folder_path=tmp_path,
        max_workers=1,
        use_cache=False,
        disable_tqdm=True,
    )

    # Four distinct contents for five files.
    assert len(put_oids) == 4
    committed = [op["path"] for call in client.create_commit.call_args_list for op in call.kwargs["operations"]]
    assert committed == ["a.dat", "b.dat", "c.dat", "d.dat", "e.dat"]


def test_a_copy_is_not_committed_when_its_owner_fails(tmp_path: Path, monkeypatch) -> None:
    # The whole point of the owner election is that copies skip the transfer.
    # If the owner fails, committing a copy would publish an LFS pointer to a
    # blob that was never stored.
    monkeypatch.setattr(upload_module, "UPLOAD_LFS_FORCE_THRESHOLD_BYTES", 0)
    monkeypatch.setattr(upload_module, "UPLOAD_RECOVERY_ENABLED", False)
    monkeypatch.setattr(upload_module, "UPLOAD_FAILED_FILE_MAX_RETRY_ROUNDS", 0)
    manager, client, _ = _make_manager()
    payload = b"shared" * 700
    for name in ("dup-0.dat", "dup-1.dat", "dup-2.dat"):
        (tmp_path / name).write_bytes(payload)
    (tmp_path / "solo.dat").write_bytes(b"solo" * 900)

    def failing_upload(*, upload_url: str, data, size: int) -> None:
        if size == len(payload):
            raise NetworkError("owner transfer failed")
        while data.read(1024 * 1024):
            pass

    client.upload_blob.side_effect = failing_upload
    monkeypatch.setattr(upload_module.time, "sleep", lambda _s: None)

    # The owner and both copies count as failures, so the upload reports itself
    # as failed rather than quietly publishing broken pointers.
    with pytest.raises(StorageError, match="3 file"):
        manager.upload_folder(
            repo_id="owner/repo",
            repo_type="dataset",
            folder_path=tmp_path,
            max_workers=1,
            use_cache=False,
            disable_tqdm=True,
        )

    committed = {op["path"] for call in client.create_commit.call_args_list for op in call.kwargs["operations"]}
    # Only the independent file is committed; no copy of the failed content is.
    assert committed == {"solo.dat"}
    assert not any(path.startswith("dup-") for path in committed)


def test_globally_existing_content_needs_no_owner(tmp_path: Path, monkeypatch) -> None:
    # When the server already holds the blob, nobody transfers it and the copies
    # are not deferred either.
    monkeypatch.setattr(upload_module, "UPLOAD_LFS_FORCE_THRESHOLD_BYTES", 0)
    manager, client, put_oids = _make_manager()
    client.validate_blobs.side_effect = lambda **kw: {obj["oid"]: None for obj in kw["objects"]}
    _write_duplicates(tmp_path, {"aa": 3, "bb": 2})

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="dataset",
        folder_path=tmp_path,
        max_workers=2,
        use_cache=False,
        disable_tqdm=True,
    )

    assert put_oids == []
    operations = client.create_commit.call_args.kwargs["operations"]
    assert len(operations) == 5


def test_inline_files_are_not_deduplicated(tmp_path: Path, monkeypatch) -> None:
    # Inline content travels in the commit body, not as a blob, so there is no
    # transfer to skip and every occurrence must carry its own bytes.
    monkeypatch.setattr(upload_module, "UPLOAD_LFS_FORCE_THRESHOLD_BYTES", 1024 * 1024)
    manager, client, put_oids = _make_manager()
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_bytes(b"identical")

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="dataset",
        folder_path=tmp_path,
        max_workers=2,
        use_cache=False,
        disable_tqdm=True,
    )

    assert put_oids == []
    operations = client.create_commit.call_args.kwargs["operations"]
    assert len(operations) == 3
    assert all(op["type"] == "normal" and op["content"] for op in operations)


def test_duplicate_marker_skips_the_transfer_without_claiming_global_reuse() -> None:
    manager, client, put_oids = _make_manager()

    result = manager._upload_blob(
        repo_id="owner/repo",
        repo_type="dataset",
        sha256="a" * 64,
        size=123,
        data=b"x" * 123,
        disable_tqdm=True,
        pre_validated=DUPLICATE_BLOB,
    )

    assert put_oids == []
    client.validate_blobs.assert_not_called()
    assert result["is_uploaded"] is True
    assert result["is_reused"] is True
    # Nothing went over the wire, so throughput accounting must see zero.
    assert result["is_blob_uploaded"] is False


@pytest.mark.parametrize("workers", [1, 4, 16])
def test_dedup_holds_at_any_concurrency(tmp_path: Path, monkeypatch, workers: int) -> None:
    # Election happens before any worker starts, so the outcome must not depend
    # on scheduling -- and no worker may ever block on another.
    monkeypatch.setattr(upload_module, "UPLOAD_LFS_FORCE_THRESHOLD_BYTES", 0)
    manager, _, put_oids = _make_manager()
    distinct = _write_duplicates(tmp_path, {"aa": 8, "bb": 8, "cc": 8, "dd": 1})

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="dataset",
        folder_path=tmp_path,
        max_workers=workers,
        use_cache=False,
        disable_tqdm=True,
    )

    assert len(put_oids) == distinct
