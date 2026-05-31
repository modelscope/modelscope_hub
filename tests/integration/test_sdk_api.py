"""Integration tests for the HubApi public SDK interface.

These tests exercise the SDK's public methods directly (not via CLI),
making real API calls to ModelScope Hub. They focus on the most important
user-facing interfaces: download, upload, repo management, and OpenAPI queries.

Requires MODELSCOPE_TEST_TOKEN and MODELSCOPE_TEST_OWNER in .env.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from modelscope_hub import HubApi, RepoType
from modelscope_hub.errors import NotFoundError


@pytest.mark.remote
class TestRepoManagement:
    """Test HubApi repo CRUD operations directly."""

    def test_create_get_delete_model(self, api, test_owner, unique_repo_name):
        repo_id = f"{test_owner}/{unique_repo_name}"
        try:
            info = api.create_repo(repo_id, "model", visibility="private")
            assert info is not None
            assert info.repo_id == repo_id or info.name == unique_repo_name

            repo = api.get_repo(repo_id, "model")
            assert repo.owner == test_owner

            assert api.repo_exists(repo_id, "model") is True
        finally:
            api.delete_repo(repo_id, "model")

        assert api.repo_exists(repo_id, "model") is False

    def test_repo_exists_returns_false_for_nonexistent(self, api, test_owner):
        assert api.repo_exists(f"{test_owner}/nonexistent_repo_xyz_999", "model") is False

    def test_list_repos_returns_paged_result(self, api, test_owner):
        result = api.list_repos("model", owner=test_owner, page_size=3)
        assert hasattr(result, "items")
        assert hasattr(result, "total_count")
        assert isinstance(result.items, list)
        assert result.page_size == 3


@pytest.mark.remote
class TestFileOperations:
    """Test upload_file, download_file, download_repo, list_repo_files, delete_files."""

    @pytest.fixture(autouse=True)
    def setup_repo(self, api, test_owner, unique_repo_name):
        self.repo_id = f"{test_owner}/{unique_repo_name}"
        self.api = api
        api.create_repo(self.repo_id, "model", visibility="private")
        yield
        try:
            api.delete_repo(self.repo_id, "model")
        except Exception:
            pass

    def test_upload_and_download_file(self, tmp_path):
        self.api.upload_file(
            self.repo_id, "model",
            b"test content for sdk",
            "sdk_test.txt",
            commit_message="sdk test upload",
        )
        local = self.api.download_file(
            self.repo_id, "model", "sdk_test.txt",
            cache_dir=str(tmp_path), force=True,
        )
        assert local.exists()
        assert local.read_text() == "test content for sdk"

    def test_download_file_to_local_dir(self, tmp_path):
        self.api.upload_file(
            self.repo_id, "model",
            b"local dir content",
            "subdir/data.txt",
            commit_message="upload for local_dir test",
        )
        local = self.api.download_file(
            self.repo_id, "model", "subdir/data.txt",
            local_dir=str(tmp_path), force=True,
        )
        expected = tmp_path / "subdir" / "data.txt"
        assert local == expected
        assert expected.exists()
        assert expected.read_text() == "local dir content"

    def test_download_repo_snapshot(self, tmp_path):
        self.api.upload_file(
            self.repo_id, "model", b"file1", "a.txt", commit_message="a",
        )
        self.api.upload_file(
            self.repo_id, "model", b"file2", "b.txt", commit_message="b",
        )
        output = self.api.download_repo(
            self.repo_id, "model", cache_dir=str(tmp_path),
        )
        assert output.is_dir()
        files = [p.name for p in output.rglob("*") if p.is_file()]
        assert "a.txt" in files
        assert "b.txt" in files

    def test_download_repo_to_local_dir(self, tmp_path):
        self.api.upload_file(
            self.repo_id, "model", b"x", "x.txt", commit_message="x",
        )
        output = self.api.download_repo(
            self.repo_id, "model", local_dir=str(tmp_path / "out"),
        )
        assert output == tmp_path / "out"
        assert (tmp_path / "out" / "x.txt").exists()

    def test_download_repo_with_patterns(self, tmp_path):
        self.api.upload_file(
            self.repo_id, "model", b"bin", "weights.bin", commit_message="bin",
        )
        self.api.upload_file(
            self.repo_id, "model", b"json", "config.json", commit_message="json",
        )
        output = self.api.download_repo(
            self.repo_id, "model",
            cache_dir=str(tmp_path),
            allow_patterns=["*.json"],
        )
        files = [p.name for p in output.rglob("*") if p.is_file()]
        assert "config.json" in files
        assert "weights.bin" not in files

    def test_list_repo_files(self):
        self.api.upload_file(
            self.repo_id, "model", b"data", "list_test.txt", commit_message="list",
        )
        files = self.api.list_repo_files(self.repo_id, "model")
        paths = [f.path for f in files]
        assert "list_test.txt" in paths

    def test_delete_files(self):
        self.api.upload_file(
            self.repo_id, "model", b"del", "to_delete.txt", commit_message="del",
        )
        self.api.delete_files(
            self.repo_id, "model", ["to_delete.txt"], commit_message="cleanup",
        )
        files = self.api.list_repo_files(self.repo_id, "model")
        paths = [f.path for f in files]
        assert "to_delete.txt" not in paths

    def test_upload_folder(self, tmp_path):
        folder = tmp_path / "upload_src"
        folder.mkdir()
        (folder / "f1.txt").write_text("one")
        (folder / "f2.txt").write_text("two")

        self.api.upload_folder(
            self.repo_id, "model",
            str(folder),
            path_in_repo="",
            commit_message="folder upload",
        )
        files = self.api.list_repo_files(self.repo_id, "model")
        paths = [f.path for f in files]
        assert "f1.txt" in paths
        assert "f2.txt" in paths


@pytest.mark.remote
class TestVersioning:
    """Test list_repo_revisions, create_repo_tag."""

    @pytest.fixture(autouse=True)
    def setup_repo(self, api, test_owner, unique_repo_name):
        self.repo_id = f"{test_owner}/{unique_repo_name}"
        self.api = api
        api.create_repo(self.repo_id, "model", visibility="private")
        api.upload_file(
            self.repo_id, "model", b"init", "init.txt", commit_message="initial",
        )
        yield
        try:
            api.delete_repo(self.repo_id, "model")
        except Exception:
            pass

    def test_list_revisions(self):
        revisions = self.api.list_repo_revisions(self.repo_id, "model")
        assert isinstance(revisions, list)
        assert len(revisions) > 0
        names = [r.get("name") or r.get("Name") or "" for r in revisions]
        assert "master" in names

    def test_create_tag(self):
        self.api.create_repo_tag(self.repo_id, "model", "v1.0")
        revisions = self.api.list_repo_revisions(self.repo_id, "model")
        names = [r.get("name") or r.get("Name") or "" for r in revisions]
        assert "v1.0" in names


@pytest.mark.remote
class TestAuth:
    """Test login, whoami, logout."""

    def test_whoami_returns_user_info(self, api, test_owner):
        user = api.whoami()
        assert user.username == test_owner or user.email is not None

    def test_login_with_valid_token(self, test_token, test_endpoint):
        api = HubApi(endpoint=test_endpoint)
        user = api.login(test_token)
        assert user.username is not None or user.email is not None


@pytest.mark.remote
class TestOpenAPIQueries:
    """Test OpenAPI-backed read operations (list models, datasets, MCP)."""

    def test_list_models(self, api):
        result = api.list_repos("model", page_size=5)
        assert hasattr(result, "items")
        assert len(result.items) <= 5

    def test_list_datasets(self, api):
        result = api.list_repos("dataset", page_size=3)
        assert hasattr(result, "items")
        assert isinstance(result.items, list)

    def test_get_public_model_info(self, api):
        info = api.get_repo("Qwen/Qwen2.5-0.5B", "model")
        assert info is not None
        assert info.repo_id is not None or info.name is not None

    def test_list_mcp_servers(self, api):
        result = api.list_mcp_servers(page_size=5)
        assert hasattr(result, "items")
        assert hasattr(result, "total_count")


@pytest.mark.remote
class TestCache:
    """Test scan_cache and clear_cache with real downloaded data."""

    def test_scan_cache_on_tmp(self, tmp_path):
        api = HubApi()
        report = api.scan_cache(cache_dir=str(tmp_path))
        assert report.total_size == 0
        assert report.total_repos == 0

    def test_download_then_scan_cache(self, api, tmp_path):
        api.download_file(
            "Qwen/Qwen2.5-0.5B", "model",
            "config.json",
            cache_dir=str(tmp_path), force=True,
        )
        report = api.scan_cache(cache_dir=str(tmp_path))
        assert report.total_repos >= 1
        assert report.total_size > 0

    def test_clear_cache_by_type(self, api, tmp_path):
        api.download_file(
            "Qwen/Qwen2.5-0.5B", "model",
            "config.json",
            cache_dir=str(tmp_path), force=True,
        )
        freed = api.clear_cache(cache_dir=str(tmp_path), repo_type="model")
        assert freed >= 0


@pytest.mark.remote
class TestCompatSDK:
    """Test the compat SDK wrappers with real API calls."""

    def test_snapshot_download(self, test_token, test_endpoint, tmp_path):
        from modelscope_hub.compat import snapshot_download

        result = snapshot_download(
            "Qwen/Qwen2.5-0.5B",
            cache_dir=str(tmp_path),
            allow_file_pattern="config.json",
            token=test_token,
            endpoint=test_endpoint,
        )
        assert Path(result).is_dir()
        files = [p.name for p in Path(result).rglob("*") if p.is_file()]
        assert "config.json" in files

    def test_model_file_download(self, test_token, test_endpoint, tmp_path):
        from modelscope_hub.compat import model_file_download

        result = model_file_download(
            "Qwen/Qwen2.5-0.5B",
            "config.json",
            cache_dir=str(tmp_path),
            token=test_token,
            endpoint=test_endpoint,
        )
        assert Path(result).exists()

    def test_legacy_hub_api_get_model(self, test_token, test_endpoint):
        from modelscope_hub.compat import LegacyHubApi

        legacy = LegacyHubApi(token=test_token, endpoint=test_endpoint)
        info = legacy.get_model("Qwen/Qwen2.5-0.5B")
        assert isinstance(info, dict)
        assert len(info) > 0

    def test_legacy_hub_api_get_model_files(self, test_token, test_endpoint):
        from modelscope_hub.compat import LegacyHubApi

        legacy = LegacyHubApi(token=test_token, endpoint=test_endpoint)
        files = legacy.get_model_files("Qwen/Qwen2.5-0.5B")
        assert isinstance(files, list)
        paths = [f.get("Path") for f in files]
        assert "config.json" in paths
