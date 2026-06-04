"""Tests for ``ms login`` and ``ms whoami`` commands.

Includes:
- Parser tests: argument parsing
- Execution tests: mock HubApi for login/whoami logic
- Remote tests: real API (existing)
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from modelscope_hub.cli.login import LoginCommand, WhoamiCommand
from modelscope_hub.types import UserInfo

from .conftest import run_cli


# ===================================================================
# Parser tests
# ===================================================================
class TestLoginParser:
    """``ms login`` argument parsing."""

    def test_no_args(self, parser):
        args = parser.parse_args(["login"])
        assert args.login_token is None

    def test_with_token(self, parser):
        args = parser.parse_args(["login", "--token", "my-token-123"])
        assert args.login_token == "my-token-123"

    def test_subcmd_endpoint(self, parser):
        args = parser.parse_args(["login", "--endpoint", "https://custom.cn"])
        assert args.subcmd_endpoint == "https://custom.cn"


class TestWhoamiParser:
    """``ms whoami`` argument parsing."""

    def test_no_args(self, parser):
        args = parser.parse_args(["whoami"])
        assert hasattr(args, "_command")


# ===================================================================
# Execution tests — mock HubApi
# ===================================================================
@pytest.mark.mock_only
class TestLoginExecute:
    """LoginCommand.execute() logic."""

    def test_login_with_token_flag(self, parser, mock_api, capsys):
        args = parser.parse_args(["login", "--token", "my-token"])
        with patch("modelscope_hub.cli.login.make_api", return_value=mock_api):
            LoginCommand(args).execute()
        mock_api.login.assert_called_once_with("my-token")
        out = capsys.readouterr().out
        assert "Logged in as" in out
        assert "testuser" in out

    def test_login_with_global_token(self, parser, mock_api, capsys):
        args = parser.parse_args(["--token", "global-tok", "login"])
        with patch("modelscope_hub.cli.login.make_api", return_value=mock_api):
            LoginCommand(args).execute()
        mock_api.login.assert_called_once_with("global-tok")

    def test_login_interactive(self, parser, mock_api, capsys):
        args = parser.parse_args(["login"])
        with (
            patch("modelscope_hub.cli.login.make_api", return_value=mock_api),
            patch("modelscope_hub.cli.login.getpass.getpass", return_value="interactive-tok"),
        ):
            LoginCommand(args).execute()
        mock_api.login.assert_called_once_with("interactive-tok")

    def test_login_empty_token_exits(self, parser, mock_api):
        args = parser.parse_args(["login"])
        with (
            patch("modelscope_hub.cli.login.make_api", return_value=mock_api),
            patch("modelscope_hub.cli.login.getpass.getpass", return_value=""),
        ):
            with pytest.raises(SystemExit) as exc_info:
                LoginCommand(args).execute()
            assert exc_info.value.code == 2

    def test_login_whitespace_only_exits(self, parser, mock_api):
        args = parser.parse_args(["login"])
        with (
            patch("modelscope_hub.cli.login.make_api", return_value=mock_api),
            patch("modelscope_hub.cli.login.getpass.getpass", return_value="   "),
        ):
            with pytest.raises(SystemExit) as exc_info:
                LoginCommand(args).execute()
            assert exc_info.value.code == 2

    def test_login_ctrl_c_exits_130(self, parser, mock_api):
        args = parser.parse_args(["login"])
        with (
            patch("modelscope_hub.cli.login.make_api", return_value=mock_api),
            patch("modelscope_hub.cli.login.getpass.getpass", side_effect=KeyboardInterrupt),
        ):
            with pytest.raises(SystemExit) as exc_info:
                LoginCommand(args).execute()
            assert exc_info.value.code == 130

    def test_login_eof_exits_130(self, parser, mock_api):
        args = parser.parse_args(["login"])
        with (
            patch("modelscope_hub.cli.login.make_api", return_value=mock_api),
            patch("modelscope_hub.cli.login.getpass.getpass", side_effect=EOFError),
        ):
            with pytest.raises(SystemExit) as exc_info:
                LoginCommand(args).execute()
            assert exc_info.value.code == 130

    def test_login_subcmd_endpoint_merged(self, parser, mock_api, capsys):
        args = parser.parse_args([
            "login", "--token", "tok", "--endpoint", "https://custom.cn",
        ])
        with patch("modelscope_hub.cli.login.make_api", return_value=mock_api):
            LoginCommand(args).execute()
        assert args.endpoint == "https://custom.cn"


@pytest.mark.mock_only
class TestWhoamiExecute:
    """WhoamiCommand.execute() logic."""

    def test_whoami_prints_user_info(self, parser, mock_api, capsys):
        args = parser.parse_args(["whoami"])
        with patch("modelscope_hub.cli.login.make_api", return_value=mock_api):
            WhoamiCommand(args).execute()
        mock_api.whoami.assert_called_once()
        out = capsys.readouterr().out
        assert "testuser" in out
        assert "test@example.com" in out
        assert "42" in out

    def test_whoami_missing_fields(self, parser, mock_api, capsys):
        mock_api.whoami.return_value = UserInfo()
        args = parser.parse_args(["whoami"])
        with patch("modelscope_hub.cli.login.make_api", return_value=mock_api):
            WhoamiCommand(args).execute()
        out = capsys.readouterr().out
        assert "-" in out


# ===================================================================
# Remote integration tests (existing)
# ===================================================================
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
