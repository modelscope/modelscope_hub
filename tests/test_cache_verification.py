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


def test_verify_local_directory_ignores_hidden_paths(tmp_path):
    (tmp_path / "visible.txt").write_bytes(b"visible")
    (tmp_path / ".DS_Store").write_bytes(b"metadata")
    hidden_dir = tmp_path / ".git"
    hidden_dir.mkdir()
    (hidden_dir / "config").write_bytes(b"git config")

    result = verify_cache(
        "owner/repo",
        "model",
        {"visible.txt": _sha256(b"visible")},
        local_dir=tmp_path,
    )

    assert result.checked_count == 1
    assert result.extra_paths == []


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


def test_verify_uses_main_as_default_cached_revision(tmp_path):
    snapshots = tmp_path / "models" / "owner--repo" / "snapshots"
    main_snapshot = snapshots / "main"
    main_snapshot.mkdir(parents=True)
    (snapshots / "dev").mkdir()
    (main_snapshot / "config.json").write_bytes(b"{}")

    result = verify_cache(
        "owner/repo",
        "model",
        {"config.json": _sha256(b"{}")},
        cache_dir=tmp_path,
    )

    assert result.checked_count == 1
    assert result.revision == "main"


def test_verify_accepts_plural_repo_type(tmp_path):
    snapshot = tmp_path / "models" / "owner--repo" / "snapshots" / "master"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_bytes(b"{}")

    result = verify_cache(
        "owner/repo",
        "models",
        {"config.json": _sha256(b"{}")},
        cache_dir=tmp_path,
    )

    assert result.checked_count == 1


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
        patch.object(api, "list_repo_files", return_value=files) as mocked_list,
        patch("modelscope_hub.api._verify_cache", return_value=expected_result) as mocked_verify,
    ):
        result = api.verify_cache("owner/repo", local_dir=tmp_path)

    assert result is expected_result
    assert mocked_verify.call_args.args[:3] == (
        "owner/repo",
        "model",
        {"config.json": "abc"},
    )
    mocked_list.assert_called_once_with(
        "owner/repo",
        "model",
        revision="master",
        recursive=True,
    )


def test_api_uses_resolved_cached_revision_for_remote_metadata(tmp_path):
    snapshot = tmp_path / "models" / "owner--repo" / "snapshots" / "dev"
    snapshot.mkdir(parents=True)
    api = HubApi()

    with (
        patch.object(api, "list_repo_files", return_value=[]) as mocked_list,
        patch("modelscope_hub.api._verify_cache") as mocked_verify,
    ):
        api.verify_cache("owner/repo", cache_dir=tmp_path)

    mocked_list.assert_called_once_with(
        "owner/repo",
        "model",
        revision="dev",
        recursive=True,
    )
    assert mocked_verify.call_args.kwargs["revision"] == "dev"


def test_file_info_positional_fields_remain_compatible():
    info = FileInfo("file.txt", 10, "blob", "tree", None, {"sha256": "abc"})

    assert info.type == "tree"
    assert info.lfs == {"sha256": "abc"}
    assert info.sha256 is None
