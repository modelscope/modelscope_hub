"""Tests for ``ms login`` and ``ms whoami`` commands."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from modelscope_hub.errors import AuthenticationError
from modelscope_hub.types import UserInfo


class TestLogin:
    """Verify login command variants."""

    def test_login_with_token_flag(self, mock_api, run_cli):
        """Login with --token flag succeeds."""
        exit_code, out, err = run_cli(["login", "--token", "my_secret_token"])
        assert exit_code == 0
        assert "test_user" in out
        mock_api.login.assert_called_once_with("my_secret_token")

    def test_login_interactive_input(self, mock_api, run_cli):
        """Login with interactive stdin input succeeds."""
        with patch("getpass.getpass", return_value="interactive_token"):
            exit_code, out, err = run_cli(["login"])
        assert exit_code == 0
        mock_api.login.assert_called_once_with("interactive_token")

    def test_login_empty_token_error(self, mock_api, run_cli):
        """Login with empty token raises SystemExit(2)."""
        exit_code, out, err = run_cli(["login", "--token", "   "])
        assert exit_code == 2
        assert "non-empty" in err.lower() or "token" in err.lower()

    def test_login_api_failure(self, mock_api, run_cli):
        """Login fails when API rejects token."""
        mock_api.login.side_effect = AuthenticationError(
            "Invalid token", status_code=401
        )
        exit_code, out, err = run_cli(["login", "--token", "bad_token"])
        assert exit_code == 1
        assert "Invalid token" in err or "401" in err

    def test_login_interactive_eof(self, mock_api, run_cli):
        """Login aborts on EOF during interactive input."""
        with patch("getpass.getpass", side_effect=EOFError):
            exit_code, out, err = run_cli(["login"])
        assert exit_code == 130


class TestWhoami:
    """Verify whoami command."""

    def test_whoami_success(self, mock_api, run_cli):
        """whoami displays username, email, and id."""
        exit_code, out, err = run_cli(["whoami"])
        assert exit_code == 0
        assert "test_user" in out
        assert "test@example.com" in out
        assert "123" in out

    def test_whoami_not_logged_in(self, mock_api, run_cli):
        """whoami fails with AuthenticationError when not logged in."""
        mock_api.whoami.side_effect = AuthenticationError(
            "Not authenticated", status_code=401
        )
        exit_code, out, err = run_cli(["whoami"])
        assert exit_code == 1
        assert "Not authenticated" in err or "401" in err
