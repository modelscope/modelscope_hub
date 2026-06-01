"""Tests for ``ms secret`` group — real API secret CRUD lifecycle."""
from __future__ import annotations

import warnings

import pytest

from .conftest import run_cli


@pytest.mark.remote
class TestSecretLifecycle:
    """Test secret CRUD operations with real API on a studio repo."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_repo(self, api, test_owner, repo_name):
        """Create a studio repo for secret testing."""
        cls = type(self)
        cls.repo_id = f"{test_owner}/{repo_name}_secrets"
        api.create_repo(cls.repo_id, "studio", visibility="private")
        cls.api = api
        yield
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                api.delete_repo(cls.repo_id, "studio")
            except Exception:
                pass

    def test_01_add_secret(self, test_token, test_endpoint):
        """Add a new secret."""
        exit_code, out, err = run_cli(
            ["secret", "add", self.repo_id, "TEST_KEY", "test_value"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [secret add] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Added" in out

    def test_02_list_secrets(self, test_token, test_endpoint):
        """List secrets shows the added secret."""
        exit_code, out, err = run_cli(
            ["secret", "list", self.repo_id],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [secret list] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "TEST_KEY" in out

    def test_03_update_secret(self, test_token, test_endpoint):
        """Update an existing secret."""
        exit_code, out, err = run_cli(
            ["secret", "update", self.repo_id, "TEST_KEY", "new_value"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [secret update] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Updated" in out

    def test_04_delete_secret(self, test_token, test_endpoint):
        """Delete the secret with --yes to skip confirmation."""
        exit_code, out, err = run_cli(
            ["secret", "delete", self.repo_id, "TEST_KEY", "--yes"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [secret delete] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Deleted" in out
