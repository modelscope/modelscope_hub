from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest

from modelscope_hub._cache_manager import verify_cache
from modelscope_hub.api import HubApi
from modelscope_hub.errors import CacheError
from modelscope_hub.types import CacheVerification, FileInfo


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_verify_local_directory_reports_all_categories(tmp_path):
    (tmp_path / "good.txt").write_bytes(b"good")
    (tmp_path / "bad.txt").write_bytes(b"bad")
    (tmp_path / "extra.txt").write_bytes(b"extra")

    result = verify_cache(
        "owner/repo",
        "model",
        {
            "good.txt": _sha256(b"good"),
            "bad.txt": _sha256(b"expected"),
            "missing.txt": _sha256(b"missing"),
            "without-hash.txt": None,
        },
        local_dir=tmp_path,
    )

    assert result.checked_count == 2
    assert [item.path for item in result.mismatches] == ["bad.txt"]
    assert result.missing_paths == ["missing.txt", "without-hash.txt"]
    assert result.extra_paths == ["extra.txt"]


def test_verify_cached_snapshot(tmp_path):
    snapshot = tmp_path / "models" / "owner--repo" / "snapshots" / "master"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(b"{}")

    result = verify_cache(
        "owner/repo",
        "model",
        {"config.json": _sha256(b"{}")},
        cache_dir=tmp_path,
    )

    assert result.checked_count == 1
    assert result.revision == "master"


def test_verify_accepts_string_paths(tmp_path):
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    (local_dir / "config.json").write_bytes(b"{}")

    result = verify_cache(
        "owner/repo",
        "model",
        {"config.json": _sha256(b"{}")},
        local_dir=str(local_dir),
    )

    assert result.checked_count == 1


def test_verify_rejects_ambiguous_cached_revision(tmp_path):
    snapshots = tmp_path / "models" / "owner--repo" / "snapshots"
    (snapshots / "one").mkdir(parents=True)
    (snapshots / "two").mkdir()

    with pytest.raises(CacheError, match="ambiguous"):
        verify_cache("owner/repo", "model", {}, cache_dir=tmp_path)


def test_verify_reports_empty_snapshot_directory(tmp_path):
    snapshots = tmp_path / "models" / "owner--repo" / "snapshots"
    snapshots.mkdir(parents=True)

    with pytest.raises(CacheError, match="No cached revisions"):
        verify_cache("owner/repo", "model", {}, cache_dir=str(tmp_path))


def test_api_passes_remote_sha256_metadata_to_verifier(tmp_path):
    api = HubApi()
    files = [
        FileInfo(path="config.json", sha256="abc"),
        FileInfo(path="folder", type="tree"),
    ]
    expected_result = CacheVerification(revision="master", verified_path=str(tmp_path))

    with (
        patch.object(api, "list_repo_files", return_value=files),
        patch("modelscope_hub.api._verify_cache", return_value=expected_result) as mocked_verify,
    ):
        result = api.verify_cache("owner/repo", local_dir=tmp_path)

    assert result is expected_result
    assert mocked_verify.call_args.args[:3] == (
        "owner/repo",
        "model",
        {"config.json": "abc"},
    )
