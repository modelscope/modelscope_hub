"""Tests for ``ms login`` and ``ms whoami`` commands — real API."""
from __future__ import annotations

import pytest

from .conftest import run_cli


@pytest.mark.remote
class TestLogin:
    """Test login with real API credentials."""

    def test_login_with_token_flag(self, test_token, test_endpoint, test_owner):
        """Login with --token flag succeeds and returns user identity."""
        exit_code, out, err = run_cli(
            ["login", "--token", test_token],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n**Exit code: {exit_code}\n**Output: {out}\n**Error: {err}")
        assert exit_code == 0
        assert "Logged in as" in out


@pytest.mark.remote
class TestWhoami:
    """Test whoami with real API credentials."""

    def test_whoami_success(self, test_token, test_endpoint, test_owner):
        """whoami displays user information."""
        exit_code, out, err = run_cli(
            ["whoami"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n**Exit code: {exit_code}\n**Output: {out}\n**Error: {err}")
        assert exit_code == 0
        assert test_owner in out
        assert "username" in out
