"""Tests for ``ms mcp`` group — real API MCP server operations."""
from __future__ import annotations

import pytest

from .conftest import run_cli


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
        # May have servers or show "no MCP servers found"
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
