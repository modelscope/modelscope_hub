"""Tests for ``ms deploy``, ``ms stop``, ``ms logs``, ``ms settings`` — real API."""
from __future__ import annotations

import pytest

from .conftest import run_cli


@pytest.mark.remote
class TestDeployLifecycle:
    """Test deploy/stop/logs/settings with real API on a studio repo."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_studio(self, api, test_owner, repo_name):
        """Create a studio for deploy testing."""
        cls = type(self)
        cls.repo_id = f"{test_owner}/{repo_name}_studio"
        api.create_repo(cls.repo_id, "studio", visibility="private")
        cls.api = api
        yield
        # Ensure stopped then deleted
        try:
            api.stop_repo(cls.repo_id, "studio")
        except Exception:
            pass
        try:
            api.delete_repo(cls.repo_id, "studio")
        except Exception:
            pass

    def test_01_deploy(self, test_token, test_endpoint):
        """Deploy the studio space."""
        exit_code, out, err = run_cli(
            ["deploy", self.repo_id, "--repo-type", "studio"],
            token=test_token,
            endpoint=test_endpoint,
        )
        assert exit_code == 0
        assert "Deploy requested" in out

    def test_02_logs(self, test_token, test_endpoint):
        """Fetch runtime logs (may be empty for a fresh deploy)."""
        exit_code, out, err = run_cli(
            ["logs", self.repo_id],
            token=test_token,
            endpoint=test_endpoint,
        )
        # Logs command should succeed even if empty
        assert exit_code == 0

    def test_03_settings(self, test_token, test_endpoint):
        """Update studio settings."""
        exit_code, out, err = run_cli(
            ["settings", self.repo_id, "cpu=2"],
            token=test_token,
            endpoint=test_endpoint,
        )
        assert exit_code == 0
        assert "Updated" in out

    def test_04_stop(self, test_token, test_endpoint):
        """Stop the studio space."""
        exit_code, out, err = run_cli(
            ["stop", self.repo_id, "--repo-type", "studio"],
            token=test_token,
            endpoint=test_endpoint,
        )
        assert exit_code == 0
        assert "Stop requested" in out
