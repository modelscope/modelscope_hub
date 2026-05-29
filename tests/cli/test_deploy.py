"""Tests for ``ms deploy``, ``ms stop``, ``ms logs``, ``ms settings`` commands."""
from __future__ import annotations

import pytest

from modelscope_hub.errors import HubError


class TestDeploy:
    """Verify deploy command."""

    def test_deploy_studio(self, mock_api, run_cli):
        """Deploy a studio space."""
        exit_code, out, err = run_cli(
            ["deploy", "owner/my-studio", "--repo-type", "studio"]
        )
        assert exit_code == 0
        assert "Deploy requested" in out
        mock_api.deploy_repo.assert_called_once_with("owner/my-studio", "studio")

    def test_deploy_mcp(self, mock_api, run_cli):
        """Deploy an MCP server via deploy command."""
        exit_code, out, err = run_cli(
            ["deploy", "owner/my-mcp", "--repo-type", "mcp"]
        )
        assert exit_code == 0
        assert "Deploy requested" in out
        mock_api.deploy_repo.assert_called_once_with("owner/my-mcp", "mcp")

    def test_deploy_default_type_is_studio(self, mock_api, run_cli):
        """Deploy defaults to studio repo type."""
        exit_code, out, err = run_cli(["deploy", "owner/space"])
        assert exit_code == 0
        mock_api.deploy_repo.assert_called_once_with("owner/space", "studio")

    def test_deploy_api_error(self, mock_api, run_cli):
        """Deploy exits 1 on API error."""
        mock_api.deploy_repo.side_effect = HubError("deploy failed")
        exit_code, out, err = run_cli(["deploy", "owner/broken"])
        assert exit_code == 1
        assert "deploy failed" in err


class TestStop:
    """Verify stop command."""

    def test_stop_studio(self, mock_api, run_cli):
        """Stop a studio space."""
        exit_code, out, err = run_cli(
            ["stop", "owner/my-studio", "--repo-type", "studio"]
        )
        assert exit_code == 0
        assert "Stop requested" in out
        mock_api.stop_repo.assert_called_once_with("owner/my-studio", "studio")

    def test_stop_mcp(self, mock_api, run_cli):
        """Stop/undeploy an MCP server."""
        exit_code, out, err = run_cli(
            ["stop", "owner/my-mcp", "--repo-type", "mcp"]
        )
        assert exit_code == 0
        assert "Stop requested" in out
        mock_api.stop_repo.assert_called_once_with("owner/my-mcp", "mcp")


class TestLogs:
    """Verify logs command."""

    def test_logs_runtime(self, mock_api, run_cli):
        """Fetch runtime logs."""
        exit_code, out, err = run_cli(["logs", "owner/my-studio"])
        assert exit_code == 0
        assert "line1" in out
        mock_api.get_repo_logs.assert_called_once_with(
            "owner/my-studio",
            "studio",
            log_type="runtime",
            page_num=1,
            page_size=100,
            keyword=None,
        )

    def test_logs_build(self, mock_api, run_cli):
        """Fetch build logs."""
        exit_code, out, err = run_cli(
            ["logs", "owner/my-studio", "--log-type", "build"]
        )
        assert exit_code == 0
        mock_api.get_repo_logs.assert_called_once()
        call_kwargs = mock_api.get_repo_logs.call_args[1]
        assert call_kwargs["log_type"] == "build"

    def test_logs_with_keyword(self, mock_api, run_cli):
        """Fetch logs with keyword filter."""
        exit_code, out, err = run_cli(
            ["logs", "owner/my-studio", "--keyword", "ERROR"]
        )
        assert exit_code == 0
        call_kwargs = mock_api.get_repo_logs.call_args[1]
        assert call_kwargs["keyword"] == "ERROR"

    def test_logs_with_pagination(self, mock_api, run_cli):
        """Fetch logs with custom page/page-size."""
        exit_code, out, err = run_cli(
            ["logs", "owner/studio", "--page", "3", "--page-size", "100"]
        )
        assert exit_code == 0
        call_kwargs = mock_api.get_repo_logs.call_args[1]
        assert call_kwargs["page_num"] == 3
        assert call_kwargs["page_size"] == 100

    def test_logs_empty_payload(self, mock_api, run_cli):
        """Logs with empty dict payload produces JSON output."""
        mock_api.get_repo_logs.return_value = {}
        exit_code, out, err = run_cli(["logs", "owner/my-studio"])
        assert exit_code == 0
        # Empty dict should produce JSON output
        assert "{}" in out


class TestSettings:
    """Verify settings command."""

    def test_settings_update(self, mock_api, run_cli):
        """Update settings with key=value pairs."""
        exit_code, out, err = run_cli(
            ["settings", "owner/studio", "cpu=4", "memory=8Gi"]
        )
        assert exit_code == 0
        assert "Updated 2 setting(s)" in out
        mock_api.update_repo_settings.assert_called_once_with(
            "owner/studio", "studio", cpu="4", memory="8Gi"
        )

    def test_settings_invalid_kv(self, mock_api, run_cli):
        """Settings with invalid key=value pair exits 2."""
        exit_code, out, err = run_cli(
            ["settings", "owner/studio", "no-equals-sign"]
        )
        assert exit_code == 2
        assert "key=value" in err.lower() or "Invalid" in err

    def test_settings_skill_type(self, mock_api, run_cli):
        """Settings with --repo-type skill."""
        exit_code, out, err = run_cli(
            ["settings", "owner/skill", "--repo-type", "skill", "timeout=30"]
        )
        assert exit_code == 0
        mock_api.update_repo_settings.assert_called_once_with(
            "owner/skill", "skill", timeout="30"
        )
