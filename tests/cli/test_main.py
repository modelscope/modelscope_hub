"""Tests for CLI entry point, global parameters, exception handling, and version."""
from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from modelscope_hub import __version__
from modelscope_hub.cli.main import run_cmd
from modelscope_hub.errors import HubError, InvalidParameter, NetworkError, NotSupportedError

from .conftest import run_cli


# ---------------------------------------------------------------------------
# Version & help (no API needed)
# ---------------------------------------------------------------------------
class TestVersionAndHelp:
    """Verify version flag and help output."""

    def test_version_flag(self):
        """--version prints version string and exits 0."""
        with pytest.raises(SystemExit) as exc_info:
            run_cmd(["--version"])
        print(f"\n** [--version] exit_code={exc_info.value.code}")
        assert exc_info.value.code == 0

    def test_help_flag(self, capsys):
        """--help prints usage and exits 0."""
        with pytest.raises(SystemExit) as exc_info:
            run_cmd(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        print(f"\n** [--help] exit_code={exc_info.value.code}, out={out[:200]!r}")
        assert "ModelScope Hub" in out

    def test_no_subcommand_shows_error(self):
        """Missing subcommand prints error and exits 2."""
        with pytest.raises(SystemExit) as exc_info:
            run_cmd([])
        print(f"\n** [no subcommand] exit_code={exc_info.value.code}")
        assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Global parameter parsing
# ---------------------------------------------------------------------------
class TestGlobalParameterParsing:
    """Verify global --token, --endpoint, --verbose are parsed correctly."""

    def test_global_token(self, parser):
        args = parser.parse_args(["--token", "my-tok", "whoami"])
        assert args.token == "my-tok"

    def test_global_endpoint(self, parser):
        args = parser.parse_args(["--endpoint", "https://custom.cn", "whoami"])
        assert args.endpoint == "https://custom.cn"

    def test_global_verbose(self, parser):
        args = parser.parse_args(["--verbose", "whoami"])
        assert args.verbose is True

    def test_verbose_short(self, parser):
        args = parser.parse_args(["-v", "whoami"])
        assert args.verbose is True

    def test_verbose_default_false(self, parser):
        args = parser.parse_args(["whoami"])
        assert args.verbose is False

    def test_global_flags_before_subcommand(self, parser):
        args = parser.parse_args([
            "--token", "tok", "--endpoint", "https://x.cn", "-v", "whoami",
        ])
        assert args.token == "tok"
        assert args.endpoint == "https://x.cn"
        assert args.verbose is True


# ---------------------------------------------------------------------------
# Exception handling (unit tests with mocks — no API needed)
# ---------------------------------------------------------------------------
@pytest.mark.mock_only
class TestExceptionHandlingUnit:
    """Verify run_cmd translates exceptions into correct exit codes."""

    def test_hub_error_exits_1(self):
        with patch("modelscope_hub.cli.login.make_api") as mock_make:
            mock_make.return_value.whoami.side_effect = HubError("server error")
            code, out, err = run_cli(["whoami"], token="fake")
        assert code == 1
        assert "server error" in err

    def test_invalid_parameter_exits_2(self):
        exc = InvalidParameter("bad param")
        exc.suggestion = "try X"
        with patch("modelscope_hub.cli.login.make_api") as mock_make:
            mock_make.return_value.whoami.side_effect = exc
            code, out, err = run_cli(["whoami"], token="fake")
        assert code == 2
        assert "bad param" in err
        assert "try X" in out

    def test_not_supported_error_exits_2(self):
        with patch("modelscope_hub.cli.login.make_api") as mock_make:
            mock_make.return_value.whoami.side_effect = NotSupportedError(
                "not supported", suggestion="use Y"
            )
            code, out, err = run_cli(["whoami"], token="fake")
        assert code == 2
        assert "not supported" in err

    def test_network_error_exits_1(self):
        with patch("modelscope_hub.cli.login.make_api") as mock_make:
            mock_make.return_value.whoami.side_effect = NetworkError("connection refused")
            code, out, err = run_cli(["whoami"], token="fake")
        assert code == 1
        assert "connection refused" in err

    def test_value_error_exits_2(self):
        with patch("modelscope_hub.cli.login.make_api") as mock_make:
            mock_make.return_value.whoami.side_effect = ValueError("invalid input")
            code, out, err = run_cli(["whoami"], token="fake")
        assert code == 2
        assert "invalid input" in err

    def test_not_implemented_error_exits_2(self):
        with patch("modelscope_hub.cli.login.make_api") as mock_make:
            mock_make.return_value.whoami.side_effect = NotImplementedError("not yet")
            code, out, err = run_cli(["whoami"], token="fake")
        assert code == 2
        assert "not yet" in err

    def test_keyboard_interrupt_exits_130(self):
        with patch("modelscope_hub.cli.login.make_api") as mock_make:
            mock_make.return_value.whoami.side_effect = KeyboardInterrupt
            code, out, err = run_cli(["whoami"], token="fake")
        assert code == 130

    def test_system_exit_forwarded(self):
        with patch("modelscope_hub.cli.login.make_api") as mock_make:
            mock_make.return_value.whoami.side_effect = SystemExit(42)
            code, out, err = run_cli(["whoami"], token="fake")
        assert code == 42


# ---------------------------------------------------------------------------
# Global parameters — real API
# ---------------------------------------------------------------------------
@pytest.mark.remote
class TestGlobalParameters:
    """Verify global --token, --endpoint, --verbose flags with real API."""

    def test_whoami_with_token(self, test_token, test_endpoint, test_owner):
        """--token is passed through to the API and returns user info."""
        exit_code, out, err = run_cli(
            ["whoami"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [whoami] exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert test_owner in out
        assert "username" in out

    def test_whoami_with_endpoint(self, test_token, test_endpoint, test_owner):
        """--endpoint is passed through and used for the API call."""
        exit_code, out, err = run_cli(
            ["whoami"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [whoami+endpoint] exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert test_owner in out

    def test_verbose_flag(self, test_token, test_endpoint, test_owner):
        """--verbose enables debug logging without breaking the command."""
        exit_code, out, err = run_cli(
            ["--verbose", "whoami"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [--verbose whoami] exit_code={exit_code}, out={out!r}")
        assert exit_code == 0
        assert test_owner in out
        logging.basicConfig(level=logging.WARNING, force=True)


# ---------------------------------------------------------------------------
# Exception handling — real API error conditions
# ---------------------------------------------------------------------------
@pytest.mark.remote
class TestExceptionHandling:
    """Verify CLI translates real API errors into correct exit codes."""

    def test_invalid_token_exits_1(self, test_endpoint):
        """Invalid token → HubError → exit 1."""
        exit_code, out, err = run_cli(
            ["whoami"],
            token="invalid_token_that_will_fail_auth",
            endpoint=test_endpoint,
        )
        print(f"\n** [bad token] exit_code={exit_code}, err={err!r}")
        assert exit_code == 1

    def test_unreachable_endpoint_exits_1(self, test_token):
        """Unreachable endpoint → NetworkError → exit 1."""
        exit_code, out, err = run_cli(
            ["whoami"],
            token=test_token,
            endpoint="http://127.0.0.1:1",
        )
        print(f"\n** [bad endpoint] exit_code={exit_code}, err={err!r}")
        assert exit_code == 1
        assert "error" in err.lower()

    def test_invalid_repo_id_exits_2(self, test_token, test_endpoint):
        """Malformed repo_id → ValueError → exit 2."""
        exit_code, out, err = run_cli(
            ["info", "no-slash-invalid", "--repo-type", "model"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [bad repo_id] exit_code={exit_code}, err={err!r}")
        assert exit_code == 2
        assert "owner/name" in err

    def test_nonexistent_repo_exits_1(self, test_token, test_endpoint):
        """Non-existent repo → HubError (404) → exit 1."""
        exit_code, out, err = run_cli(
            ["info", "nonexistent_owner_xyz/nonexistent_repo_xyz", "--repo-type", "model"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [404 repo] exit_code={exit_code}, err={err!r}")
        assert exit_code == 1
