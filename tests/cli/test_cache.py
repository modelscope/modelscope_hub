"""Tests for ``ms cache`` group — real API download + local cache operations."""
from __future__ import annotations

import base64
import warnings

import pytest

from .conftest import run_cli


@pytest.mark.remote
class TestCacheLifecycle:
    """Download a file to populate the cache, then scan and clear it.

    Uses a real remote repo + real local cache directory (tmp_path).
    """

    @pytest.fixture(autouse=True, scope="class")
    def setup_repo(self, api, test_owner, test_endpoint, repo_name):
        """Create a model repo, commit a test file, download to cache."""
        cls = type(self)
        cls.repo_id = f"{test_owner}/{repo_name}_cache"
        print(f"\n** Setup: creating repo {cls.repo_id}")
        api.create_repo(cls.repo_id, "model", visibility="private")

        file_bytes = b"cache test content for scan and clear"
        content_b64 = base64.b64encode(file_bytes).decode()
        api.legacy.create_commit(
            repo_id=cls.repo_id,
            repo_type="model",
            operations=[{
                "action": "create",
                "path": "cache_data.txt",
                "type": "normal",
                "size": len(file_bytes),
                "sha256": "",
                "content": content_b64,
                "encoding": "base64",
            }],
            commit_message="Add test file for cache tests",
            revision="master",
        )
        cls.api = api
        print("** Setup: done")
        yield
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                api.delete_repo(cls.repo_id, "model")
            except Exception as e:
                print(f"** Teardown: cleanup via web console: {cls.repo_id} ({e})")

    def test_01_download_populates_cache(self, test_token, test_endpoint, tmp_path_factory):
        """Download snapshot to a shared cache dir for subsequent tests."""
        cls = type(self)
        cls.cache_dir = str(tmp_path_factory.mktemp("cache"))

        exit_code, out, err = run_cli(
            ["download", self.repo_id, "--cache-dir", cls.cache_dir],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [download] exit_code={exit_code}, out={out!r}")
        assert exit_code == 0
        assert "Snapshot ready" in out

    def test_02_cache_scan_shows_repo(self, test_token, test_endpoint):
        """cache scan --cache-dir should list the downloaded repo."""
        exit_code, out, err = run_cli(
            ["cache", "scan", "--cache-dir", self.cache_dir],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [cache scan] exit_code={exit_code}, out={out!r}")
        assert exit_code == 0
        repo_name_part = self.repo_id.split("/")[-1]
        assert repo_name_part in out
        assert "1 repo(s)" in out

    def test_03_cache_clear_by_type(self, test_token, test_endpoint):
        """cache clear --repo-type model frees bytes."""
        exit_code, out, err = run_cli(
            ["cache", "clear", "--repo-type", "model", "--cache-dir", self.cache_dir, "--yes"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [cache clear model] exit_code={exit_code}, out={out!r}")
        assert exit_code == 0
        assert "Freed" in out

    def test_04_cache_scan_empty_after_clear(self, test_token, test_endpoint):
        """cache scan should show 0 repos after clear."""
        exit_code, out, err = run_cli(
            ["cache", "scan", "--cache-dir", self.cache_dir],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [cache scan empty] exit_code={exit_code}, out={out!r}")
        assert exit_code == 0
        assert "0 repo(s)" in out

    def test_05_download_then_clear_specific_repo(self, test_token, test_endpoint, tmp_path_factory):
        """Download again, then clear only the specific repo by id."""
        cache2 = str(tmp_path_factory.mktemp("cache2"))

        exit_code, out, err = run_cli(
            ["download", self.repo_id, "--cache-dir", cache2],
            token=test_token,
            endpoint=test_endpoint,
        )
        assert exit_code == 0

        exit_code, out, err = run_cli(
            ["cache", "clear",
             "--repo-type", "model",
             "--repo-id", self.repo_id,
             "--cache-dir", cache2,
             "--yes"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [cache clear specific] exit_code={exit_code}, out={out!r}")
        assert exit_code == 0
        assert "Freed" in out

        exit_code, out, err = run_cli(
            ["cache", "scan", "--cache-dir", cache2],
            token=test_token,
            endpoint=test_endpoint,
        )
        assert exit_code == 0
        assert "0 repo(s)" in out

    def test_06_cache_clear_repo_id_without_type_error(self, test_token, test_endpoint):
        """cache clear --repo-id without --repo-type exits 2."""
        exit_code, out, err = run_cli(
            ["cache", "clear", "--repo-id", "owner/model", "--yes"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [clear missing type] exit_code={exit_code}, err={err!r}")
        assert exit_code == 2
        assert "repo-type" in err.lower()
