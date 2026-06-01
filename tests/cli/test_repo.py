"""Tests for ``ms repo`` group — real API lifecycle: create → info → list → delete."""
from __future__ import annotations

import pytest

from .conftest import run_cli


@pytest.mark.remote
class TestRepoLifecycle:
    """Test repo CRUD operations with real API."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_repo(self, api, test_owner, repo_name):
        """Store repo metadata for tests; cleanup at end."""
        cls = type(self)
        cls.repo_id = f"{test_owner}/{repo_name}"
        cls.api = api
        yield
        # Cleanup: ensure the repo is deleted regardless of test outcome
        try:
            api.delete_repo(cls.repo_id, "model")
        except Exception:
            pass

    def test_01_create_repo(self, test_token, test_endpoint):
        """Create a private model repo."""
        exit_code, out, err = run_cli(
            ["repo", "create", self.repo_id, "--repo-type", "model", "--visibility", "private"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [repo create] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Created" in out

    def test_02_repo_info(self, test_token, test_endpoint):
        """Get repo info shows metadata."""
        exit_code, out, err = run_cli(
            ["repo", "info", self.repo_id, "--repo-type", "model"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [repo info] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "repo_id" in out or self.repo_id in out

    def test_03_repo_list(self, test_token, test_endpoint, test_owner):
        """List repos for the test owner."""
        exit_code, out, err = run_cli(
            ["repo", "list", "--repo-type", "model", "--owner", test_owner],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [repo list] owner={test_owner}")
        print(f"** exit_code={exit_code}, out={out[:200]!r}, err={err!r}")
        assert exit_code == 0

    def test_04_delete_repo(self, test_token, test_endpoint):
        """Delete the created repo (currently deprecated — expected to fail)."""
        exit_code, out, err = run_cli(
            ["repo", "delete", self.repo_id, "--repo-type", "model", "--yes"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [repo delete] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        # delete_repo is deprecated; hub rejects token-based deletion for now
        if exit_code != 0:
            pytest.skip("delete_repo not yet supported via SDK token — clean up via web console")
