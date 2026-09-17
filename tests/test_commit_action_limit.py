"""Commit requests must respect the server's hard action ceiling.

The server rejects an oversized commit outright rather than truncating it::

    HTTP 422  commit request exceeds actions limit: 3300 > 2000;
              split the commit into smaller batches

Observed while removing 3300 verification files from a test dataset. Uploads were
accidentally safe because the default operation cap is well below the ceiling,
but an explicitly configured larger cap, or any delete of more than 2000 paths,
would hit it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import modelscope_hub._upload as upload_module
from modelscope_hub._upload import (
    UploadManager,
    _calculate_adaptive_batch_size,
    _plan_commit_batches,
)
from modelscope_hub.errors import APIError


def _make_manager() -> tuple[UploadManager, MagicMock]:
    client = MagicMock()
    client.create_commit.return_value = {"ok": True}
    client.validate_blobs.side_effect = lambda **kw: {o["oid"]: f"https://upload/{o['oid']}" for o in kw["objects"]}
    client.upload_blob.side_effect = lambda **kw: None
    return UploadManager(client, MagicMock()), client


def test_batch_size_is_clamped_to_the_server_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(upload_module, "COMMIT_MAX_ACTIONS_PER_REQUEST", 2000)

    # An explicit cap above the ceiling must not be honored: the server would
    # reject the request instead of trimming it.
    assert _calculate_adaptive_batch_size(50_000, 4096) == 2000
    assert _calculate_adaptive_batch_size(50_000, 512) == 512
    # "No cap" must not mean "one commit for everything" either.
    assert _calculate_adaptive_batch_size(50_000, 0) == 2000


def test_batch_plan_never_exceeds_the_server_ceiling(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(upload_module, "COMMIT_MAX_ACTIONS_PER_REQUEST", 10)
    files = [(f"f{i}.bin", str(tmp_path / f"f{i}.bin")) for i in range(35)]
    sizes = dict.fromkeys((path for _, path in files), 1)

    plan = _plan_commit_batches(
        files,
        "dataset",
        max_operations=1000,
        max_inline_bytes=0,
        sizes=sizes,
    )

    assert sum(plan) == 35
    assert max(plan) <= 10


def test_delete_files_splits_at_the_server_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(upload_module, "COMMIT_MAX_ACTIONS_PER_REQUEST", 100)
    manager, client = _make_manager()
    paths = [f"junk/file-{i}.bin" for i in range(250)]

    result = manager.delete_files(repo_id="owner/repo", repo_type="dataset", file_paths=paths)

    assert client.create_commit.call_count == 3
    sent = [len(call.kwargs["operations"]) for call in client.create_commit.call_args_list]
    assert sent == [100, 100, 50]
    assert result["total_files"] == 250
    assert result["deleted_files"] == paths


def test_delete_files_keeps_one_commit_when_it_fits(monkeypatch) -> None:
    monkeypatch.setattr(upload_module, "COMMIT_MAX_ACTIONS_PER_REQUEST", 2000)
    manager, client = _make_manager()

    manager.delete_files(repo_id="owner/repo", repo_type="dataset", file_paths=["a.bin", "b.bin"])

    client.create_commit.assert_called_once()
    assert len(client.create_commit.call_args.kwargs["operations"]) == 2


def test_delete_files_reports_what_a_partial_split_removed(monkeypatch) -> None:
    # Across a split the deletion is no longer atomic. Failing loudly beats
    # returning a success that claims paths which are still present.
    monkeypatch.setattr(upload_module, "COMMIT_MAX_ACTIONS_PER_REQUEST", 2)
    manager, client = _make_manager()
    client.create_commit.side_effect = [{"ok": True}, APIError("commit rejected", status_code=422)]

    with pytest.raises(APIError):
        manager.delete_files(
            repo_id="owner/repo",
            repo_type="dataset",
            file_paths=["a", "b", "c", "d"],
        )

    assert client.create_commit.call_count == 2


def test_server_ceiling_is_env_tunable() -> None:
    from modelscope_hub.constants import COMMIT_MAX_ACTIONS_PER_REQUEST

    assert COMMIT_MAX_ACTIONS_PER_REQUEST == 2000
