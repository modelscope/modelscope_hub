"""Tests for ``ms mcp`` group — list / info / deploy / undeploy."""
from __future__ import annotations

import json

import pytest

from modelscope_hub.errors import HubError
from modelscope_hub.types import PagedResult


class TestMcpList:
    """Verify mcp list subcommand."""

    def test_mcp_list(self, mock_api, run_cli):
        """List MCP servers displays table."""
        exit_code, out, err = run_cli(["mcp", "list"])
        assert exit_code == 0
        assert "mcp-server-1" in out
        mock_api.list_mcp_servers.assert_called_once_with(
            search=None,
            page_number=1,
            page_size=20,
        )

    def test_mcp_list_with_search(self, mock_api, run_cli):
        """List MCP servers with --search filter."""
        exit_code, out, err = run_cli(["mcp", "list", "--search", "tools"])
        assert exit_code == 0
        mock_api.list_mcp_servers.assert_called_once_with(
            search="tools",
            page_number=1,
            page_size=20,
        )

    def test_mcp_list_empty(self, mock_api, run_cli):
        """List MCP servers with no results."""
        mock_api.list_mcp_servers.return_value = PagedResult(
            items=[], total_count=0, page_number=1, page_size=20
        )
        exit_code, out, err = run_cli(["mcp", "list"])
        assert exit_code == 0
        assert "no mcp servers found" in out.lower()

    def test_mcp_list_with_pagination(self, mock_api, run_cli):
        """List MCP with page/page-size parameters."""
        exit_code, out, err = run_cli(
            ["mcp", "list", "--page", "2", "--page-size", "5"]
        )
        assert exit_code == 0
        mock_api.list_mcp_servers.assert_called_once_with(
            search=None,
            page_number=2,
            page_size=5,
        )


class TestMcpInfo:
    """Verify mcp info subcommand."""

    def test_mcp_info(self, mock_api, run_cli):
        """Get MCP server info displays JSON."""
        exit_code, out, err = run_cli(["mcp", "info", "owner/my-server"])
        assert exit_code == 0
        # Output should be valid JSON
        data = json.loads(out)
        assert data["id"] == "srv1"
        mock_api.get_mcp_server.assert_called_once_with("owner/my-server")

    def test_mcp_info_api_error(self, mock_api, run_cli):
        """mcp info exits 1 on API error."""
        mock_api.get_mcp_server.side_effect = HubError("not found")
        exit_code, out, err = run_cli(["mcp", "info", "owner/ghost"])
        assert exit_code == 1


class TestMcpDeploy:
    """Verify mcp deploy subcommand."""

    def test_mcp_deploy(self, mock_api, run_cli):
        """Deploy an MCP server."""
        exit_code, out, err = run_cli(["mcp", "deploy", "owner/my-server"])
        assert exit_code == 0
        assert "Deploy requested" in out
        mock_api.deploy_mcp_server.assert_called_once_with("owner/my-server")


class TestMcpUndeploy:
    """Verify mcp undeploy subcommand."""

    def test_mcp_undeploy(self, mock_api, run_cli):
        """Undeploy an MCP server."""
        exit_code, out, err = run_cli(["mcp", "undeploy", "owner/my-server"])
        assert exit_code == 0
        assert "Undeploy requested" in out
        mock_api.undeploy_mcp_server.assert_called_once_with("owner/my-server")

    def test_mcp_undeploy_api_error(self, mock_api, run_cli):
        """Undeploy exits 1 on API error."""
        mock_api.undeploy_mcp_server.side_effect = HubError("server busy")
        exit_code, out, err = run_cli(["mcp", "undeploy", "owner/busy-server"])
        assert exit_code == 1
        assert "server busy" in err
