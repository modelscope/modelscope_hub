"""Tests for ``ms mcp`` group — list / info / deploy / undeploy.

Includes:
- Parser tests: all subcommands and flags
- Execution tests: mock HubApi for MCP server operations
- Remote tests: real API (existing)
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from modelscope_hub.cli.mcp import _McpDeploy, _McpInfo, _McpList, _McpUndeploy
from modelscope_hub.types import PagedResult

from .conftest import run_cli


# ===================================================================
# Parser tests
# ===================================================================
class TestMcpListParser:
    """``ms mcp list`` argument parsing."""

    def test_basic(self, parser):
        args = parser.parse_args(["mcp", "list"])
        assert hasattr(args, "_mcp_leaf")

    def test_subcmd_token_endpoint(self, parser):
        args = parser.parse_args([
            "mcp", "list", "--token", "tk", "--endpoint", "https://x.cn",
        ])
        assert args.subcmd_token == "tk"
        assert args.subcmd_endpoint == "https://x.cn"

    def test_search(self, parser):
        args = parser.parse_args(["mcp", "list", "--search", "weather"])
        assert args.search == "weather"

    def test_search_default_none(self, parser):
        args = parser.parse_args(["mcp", "list"])
        assert args.search is None

    def test_page(self, parser):
        args = parser.parse_args(["mcp", "list", "--page", "3"])
        assert args.page_number == 3

    def test_page_size(self, parser):
        args = parser.parse_args(["mcp", "list", "--page-size", "10"])
        assert args.page_size == 10

    def test_page_defaults(self, parser):
        args = parser.parse_args(["mcp", "list"])
        assert args.page_number == 1
        assert args.page_size == 20

    def test_all_options(self, parser):
        args = parser.parse_args([
            "mcp", "list",
            "--search", "test",
            "--page", "2",
            "--page-size", "50",
        ])
        assert args.search == "test"
        assert args.page_number == 2
        assert args.page_size == 50


class TestMcpInfoParser:
    """``ms mcp info`` argument parsing."""

    def test_basic(self, parser):
        args = parser.parse_args(["mcp", "info", "org/weather-mcp"])
        assert args.server_id == "org/weather-mcp"

    def test_missing_server_id_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["mcp", "info"])

    def test_subcmd_token_endpoint(self, parser):
        args = parser.parse_args([
            "mcp", "info", "org/srv", "--token", "tk", "--endpoint", "https://x.cn",
        ])
        assert args.subcmd_token == "tk"
        assert args.subcmd_endpoint == "https://x.cn"


class TestMcpDeployParser:
    """``ms mcp deploy`` argument parsing."""

    def test_basic(self, parser):
        args = parser.parse_args(["mcp", "deploy", "org/weather-mcp"])
        assert args.server_id == "org/weather-mcp"

    def test_missing_server_id_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["mcp", "deploy"])

    def test_subcmd_token_endpoint(self, parser):
        args = parser.parse_args([
            "mcp", "deploy", "org/srv", "--token", "tk", "--endpoint", "https://x.cn",
        ])
        assert args.subcmd_token == "tk"


class TestMcpUndeployParser:
    """``ms mcp undeploy`` argument parsing."""

    def test_basic(self, parser):
        args = parser.parse_args(["mcp", "undeploy", "org/weather-mcp"])
        assert args.server_id == "org/weather-mcp"

    def test_missing_server_id_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["mcp", "undeploy"])

    def test_subcmd_token_endpoint(self, parser):
        args = parser.parse_args([
            "mcp", "undeploy", "org/srv", "--token", "tk", "--endpoint", "https://x.cn",
        ])
        assert args.subcmd_token == "tk"


# ===================================================================
# Execution tests — mock HubApi
# ===================================================================
@pytest.mark.mock_only
class TestMcpListExecute:
    def test_list_with_results(self, parser, mock_api, capsys):
        args = parser.parse_args(["mcp", "list"])
        with patch("modelscope_hub.cli.mcp.make_api", return_value=mock_api):
            _McpList(args).execute()
        mock_api.list_mcp_servers.assert_called_once_with(
            search=None, page_number=1, page_size=20,
        )
        out = capsys.readouterr().out
        assert "weather" in out

    def test_list_empty(self, parser, mock_api, capsys):
        mock_api.list_mcp_servers.return_value = PagedResult(
            items=[], total_count=0, page_number=1, page_size=20,
        )
        args = parser.parse_args(["mcp", "list"])
        with patch("modelscope_hub.cli.mcp.make_api", return_value=mock_api):
            _McpList(args).execute()
        out = capsys.readouterr().out
        assert "no MCP servers found" in out

    def test_list_with_search(self, parser, mock_api, capsys):
        args = parser.parse_args(["mcp", "list", "--search", "weather"])
        with patch("modelscope_hub.cli.mcp.make_api", return_value=mock_api):
            _McpList(args).execute()
        assert mock_api.list_mcp_servers.call_args.kwargs["search"] == "weather"

    def test_list_with_pagination(self, parser, mock_api, capsys):
        args = parser.parse_args(["mcp", "list", "--page", "2", "--page-size", "5"])
        with patch("modelscope_hub.cli.mcp.make_api", return_value=mock_api):
            _McpList(args).execute()
        kw = mock_api.list_mcp_servers.call_args.kwargs
        assert kw["page_number"] == 2
        assert kw["page_size"] == 5


@pytest.mark.mock_only
class TestMcpInfoExecute:
    def test_info_prints_json(self, parser, mock_api, capsys):
        args = parser.parse_args(["mcp", "info", "org/weather-mcp"])
        with patch("modelscope_hub.cli.mcp.make_api", return_value=mock_api):
            _McpInfo(args).execute()
        mock_api.get_mcp_server.assert_called_once_with("org/weather-mcp")
        out = capsys.readouterr().out
        assert "weather" in out


@pytest.mark.mock_only
class TestMcpDeployExecute:
    def test_deploy(self, parser, mock_api, capsys):
        args = parser.parse_args(["mcp", "deploy", "org/weather-mcp"])
        with patch("modelscope_hub.cli.mcp.make_api", return_value=mock_api):
            _McpDeploy(args).execute()
        mock_api.deploy_mcp_server.assert_called_once_with("org/weather-mcp")
        out = capsys.readouterr().out
        assert "Deploy requested" in out


@pytest.mark.mock_only
class TestMcpUndeployExecute:
    def test_undeploy(self, parser, mock_api, capsys):
        args = parser.parse_args(["mcp", "undeploy", "org/weather-mcp"])
        with patch("modelscope_hub.cli.mcp.make_api", return_value=mock_api):
            _McpUndeploy(args).execute()
        mock_api.undeploy_mcp_server.assert_called_once_with("org/weather-mcp")
        out = capsys.readouterr().out
        assert "Undeploy requested" in out


# ===================================================================
# Remote integration tests (existing)
# ===================================================================
@pytest.mark.remote
class TestMcpOperations:
    """Test MCP server operations with real API."""

    def test_list_mcp_servers(self, test_token, test_endpoint):
        """List MCP servers returns successfully."""
        exit_code, out, err = run_cli(
            ["mcp", "list"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [mcp list]")
        print(f"** exit_code={exit_code}, out={out[:300]!r}, err={err!r}")
        assert exit_code == 0
        assert "mcp" in out.lower() or "no MCP servers found" in out or "id" in out.lower()

    def test_list_mcp_with_search(self, test_token, test_endpoint):
        """List MCP servers with --search filter."""
        exit_code, out, err = run_cli(
            ["mcp", "list", "--search", "test"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [mcp list --search test]")
        print(f"** exit_code={exit_code}, out={out[:300]!r}, err={err!r}")
        assert exit_code == 0
