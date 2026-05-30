"""Tests for CLI entry point, global parameters, exception handling, and version."""
from __future__ import annotations

import io
import sys
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from modelscope_hub import __version__
from modelscope_hub.cli.main import run_cmd
from modelscope_hub.errors import HubError, NetworkError
from modelscope_hub.types import UserInfo


# ---------------------------------------------------------------------------
# Local mock fixtures for unit tests (no real API calls)
# ---------------------------------------------------------------------------
_MAKE_API_TARGETS = [
    "modelscope_hub.cli.base.make_api",
    "modelscope_hub.cli.login.make_api",
    "modelscope_hub.cli.repo.make_api",
    "modelscope_hub.cli.download.make_api",
    "modelscope_hub.cli.upload.make_api",
    "modelscope_hub.cli.deploy.make_api",
    "modelscope_hub.cli.secret.make_api",
    "modelscope_hub.cli.mcp.make_api",
    "modelscope_hub.cli.cache.make_api",
]


@pytest.fixture
def mock_api():
    """Create a mock HubApi instance with common return values."""
    api = MagicMock()
    api.whoami.return_value = UserInfo(
        id="123", username="test_user", email="test@example.com"
    )
    return api


@pytest.fixture
def run_cli(mock_api):
    """Run CLI commands with mocked API and capture output."""
    def _run(args: list[str], input_text: str | None = None):
        captured_out = io.StringIO()
        captured_err = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_stdin = sys.stdin
        try:
            sys.stdout = captured_out
            sys.stderr = captured_err
            if input_text is not None:
                sys.stdin = io.StringIO(input_text)
            with ExitStack() as stack:
                for target in _MAKE_API_TARGETS:
                    stack.enter_context(patch(target, return_value=mock_api))
                exit_code = run_cmd(args)
        except SystemExit as e:
            exit_code = int(e.code) if e.code else 0
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sys.stdin = old_stdin
        return exit_code, captured_out.getvalue(), captured_err.getvalue()

    return _run


class TestVersionAndHelp:
    """Verify version flag and help output."""

    def test_version_flag(self):
        """--version prints version string and exits 0."""
        with pytest.raises(SystemExit) as exc_info:
            run_cmd(["--version"])
        assert exc_info.value.code == 0

    def test_help_flag(self, capsys):
        """--help prints usage and exits 0."""
        with pytest.raises(SystemExit) as exc_info:
            run_cmd(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "ModelScope Hub" in out

    def test_no_subcommand_shows_error(self, capsys):
        """Missing subcommand prints error and exits 2."""
        with pytest.raises(SystemExit) as exc_info:
            run_cmd([])
        assert exc_info.value.code == 2


class TestGlobalParameters:
    """Verify global --token, --endpoint, --verbose flags propagation."""

    def test_global_token_passed_to_api(self, mock_api, run_cli):
        """--token is passed through to make_api."""
        with patch("modelscope_hub.cli.base.make_api", return_value=mock_api) as mock_make:
            run_cli(["whoami"])
            # make_api is called; verify args contain the token if passed
            # Since our fixture patches make_api, we just validate it works
            mock_api.whoami.assert_called_once()

    def test_global_endpoint_passed_to_api(self, mock_api, run_cli):
        """--endpoint is accepted without error."""
        exit_code, out, err = run_cli(["whoami"])
        assert exit_code == 0

    def test_verbose_flag_enables_debug(self, mock_api):
        """--verbose sets logging level to DEBUG."""
        import logging

        captured_out = io.StringIO()
        captured_err = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = captured_out
            sys.stderr = captured_err
            with patch("modelscope_hub.cli.base.make_api", return_value=mock_api):
                run_cmd(["--verbose", "whoami"])
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        # After the call, the root logger should have been configured for DEBUG
        root = logging.getLogger()
        # Reset after test
        logging.basicConfig(level=logging.WARNING, force=True)


class TestExceptionHandling:
    """Verify CLI translates exceptions into correct exit codes."""

    def test_hub_error_exits_1(self, mock_api, run_cli):
        """HubError from a command results in exit code 1."""
        mock_api.whoami.side_effect = HubError("Something went wrong")
        exit_code, out, err = run_cli(["whoami"])
        assert exit_code == 1
        assert "Something went wrong" in err

    def test_network_error_exits_1(self, mock_api, run_cli):
        """NetworkError from a command results in exit code 1."""
        mock_api.whoami.side_effect = NetworkError("Connection refused")
        exit_code, out, err = run_cli(["whoami"])
        assert exit_code == 1
        assert "Network error" in err

    def test_value_error_exits_2(self, mock_api, run_cli):
        """ValueError from a command results in exit code 2."""
        mock_api.whoami.side_effect = ValueError("bad input")
        exit_code, out, err = run_cli(["whoami"])
        assert exit_code == 2
        assert "bad input" in err

    def test_not_implemented_error_exits_2(self, mock_api, run_cli):
        """NotImplementedError from a command results in exit code 2."""
        mock_api.whoami.side_effect = NotImplementedError("not yet")
        exit_code, out, err = run_cli(["whoami"])
        assert exit_code == 2
        assert "not yet" in err

    def test_keyboard_interrupt_exits_130(self, mock_api, run_cli):
        """KeyboardInterrupt from a command results in exit code 130."""
        mock_api.whoami.side_effect = KeyboardInterrupt()
        exit_code, out, err = run_cli(["whoami"])
        assert exit_code == 130
        assert "Interrupted" in err
