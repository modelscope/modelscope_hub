"""Tests for ``ms deploy``, ``ms stop``, ``ms logs``, ``ms settings`` — real API."""
from __future__ import annotations

import warnings

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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
        print(f"\n** [deploy] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Deploy requested" in out

    def test_02_logs(self, test_token, test_endpoint):
        """Fetch run logs (may be empty or 404 while studio is starting)."""
        exit_code, out, err = run_cli(
            ["logs", self.repo_id],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [logs] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out[:300]!r}, err={err!r}")
        if exit_code != 0 and "启动中" in err:
            pytest.skip("Studio still starting — logs not yet available")
        assert exit_code == 0

    def test_03_settings(self, test_token, test_endpoint):
        """Update studio settings."""
        exit_code, out, err = run_cli(
            ["settings", self.repo_id, "cpu=2"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [settings] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Updated" in out

    def test_04_stop(self, test_token, test_endpoint):
        """Stop the studio space."""
        exit_code, out, err = run_cli(
            ["stop", self.repo_id, "--repo-type", "studio"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [stop] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Stop requested" in out
