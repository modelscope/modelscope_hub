"""Tests for CLI entry point, global parameters, exception handling, and version."""
from __future__ import annotations

import logging

import pytest

from modelscope_hub import __version__
from modelscope_hub.cli.main import run_cmd

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
            ["repo", "info", "no-slash-invalid", "--repo-type", "model"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [bad repo_id] exit_code={exit_code}, err={err!r}")
        assert exit_code == 2
        assert "owner/name" in err

    def test_nonexistent_repo_exits_1(self, test_token, test_endpoint):
        """Non-existent repo → HubError (404) → exit 1."""
        exit_code, out, err = run_cli(
            ["repo", "info", "nonexistent_owner_xyz/nonexistent_repo_xyz", "--repo-type", "model"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [404 repo] exit_code={exit_code}, err={err!r}")
        assert exit_code == 1
