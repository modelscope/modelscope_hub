from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from modelscope_hub._upload import UploadManager


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


@pytest.mark.mock_only
def test_upload_file_normal_commits_inline_without_blob_api() -> None:
    manager, client = _make_manager()

    manager.upload_file(
        repo_id="owner/repo",
        repo_type="model",
        path_or_fileobj=b"hello",
        path_in_repo="README.md",
        disable_tqdm=True,
    )

    client.validate_blobs.assert_not_called()
    client.upload_blob.assert_not_called()
    operation = client.create_commit.call_args.kwargs["operations"][0]
    assert operation["type"] == "normal"
    assert operation["sha256"] == ""
    assert base64.b64decode(operation["content"]) == b"hello"


@pytest.mark.mock_only
def test_upload_file_lfs_uses_blob_api_then_commits_pointer() -> None:
    manager, client = _make_manager()

    manager.upload_file(
        repo_id="owner/repo",
        repo_type="model",
        path_or_fileobj=b"weights",
        path_in_repo="model.bin",
        disable_tqdm=True,
    )

    client.validate_blobs.assert_called_once()
    client.upload_blob.assert_called_once()
    operation = client.create_commit.call_args.kwargs["operations"][0]
    assert operation["type"] == "lfs"
    assert operation["sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert operation["content"] == ""


@pytest.mark.mock_only
def test_upload_folder_mixed_files_only_uploads_lfs_blob(tmp_path: Path) -> None:
    manager, client = _make_manager()
    (tmp_path / "README.md").write_bytes(b"hello")
    (tmp_path / "model.bin").write_bytes(b"weights")

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="model",
        folder_path=tmp_path,
        max_workers=1,
        use_cache=False,
        disable_tqdm=True,
    )

    client.validate_blobs.assert_called_once()
    client.upload_blob.assert_called_once()
    operations = client.create_commit.call_args.kwargs["operations"]
    assert {op["path"]: op["type"] for op in operations} == {
        "README.md": "normal",
        "model.bin": "lfs",
    }


@pytest.mark.mock_only
def test_upload_folder_cached_normal_hash_skips_batch_blob_validation(tmp_path: Path) -> None:
    manager, client = _make_manager()
    file_path = tmp_path / "README.md"
    content = b"hello"
    file_path.write_bytes(content)
    st = file_path.stat()
    cache_key = f"README.md|{st.st_mtime}|{st.st_size}"
    cache_path = tmp_path / ".ms_upload_cache"
    cache_path.write_text(
        json.dumps({
            "version": 3,
            "repo_id": "owner/repo",
            "files": {
                cache_key: {
                    "hash": hashlib.sha256(content).hexdigest(),
                    "size": st.st_size,
                },
            },
        }),
        encoding="utf-8",
    )

    manager.upload_folder(
        repo_id="owner/repo",
        repo_type="model",
        folder_path=tmp_path,
        max_workers=1,
        use_cache=True,
        disable_tqdm=True,
    )

    client.validate_blobs.assert_not_called()
    client.upload_blob.assert_not_called()
    operation = client.create_commit.call_args.kwargs["operations"][0]
    assert operation["type"] == "normal"
