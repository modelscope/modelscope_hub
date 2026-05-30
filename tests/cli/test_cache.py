"""Tests for ``ms cache`` group — scan and clear (local mock tests)."""
from __future__ import annotations

import io
import sys
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

from modelscope_hub.cli.main import run_cmd
from modelscope_hub.types import CachedRepoInfo, CacheInfo


# ---------------------------------------------------------------------------
# Local mock fixtures for cache unit tests (no real API calls)
# ---------------------------------------------------------------------------
_MAKE_API_TARGETS = [
    "modelscope_hub.cli.base.make_api",
    "modelscope_hub.cli.cache.make_api",
]


@pytest.fixture
def mock_api():
    """Create a mock HubApi instance with cache return values."""
    api = MagicMock()
    api.scan_cache.return_value = CacheInfo(
        repos=[
            CachedRepoInfo(
                repo_id="owner/model1",
                repo_type="model",
                revision="master",
                size_on_disk=1024 * 1024 * 50,
                nb_files=10,
                local_path="/tmp/cache/models/owner/model1",
            ),
        ],
        total_size=1024 * 1024 * 50,
        cache_dir="/tmp/cache",
    )
    api.clear_cache.return_value = 1024 * 1024 * 50
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


class TestCacheScan:
    """Verify cache scan subcommand."""

    def test_cache_scan(self, mock_api, run_cli):
        """cache scan displays summary and table."""
        exit_code, out, err = run_cli(["cache", "scan"])
        print(f"\n** [cache scan] exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "/tmp/cache" in out
        assert "owner/model1" in out
        assert "1 repo(s)" in out
        mock_api.scan_cache.assert_called_once_with(None)

    def test_cache_scan_empty(self, mock_api, run_cli):
        """cache scan with no repos shows summary only."""
        mock_api.scan_cache.return_value = CacheInfo(
            repos=[], total_size=0, cache_dir="/tmp/cache"
        )
        exit_code, out, err = run_cli(["cache", "scan"])
        print(f"\n** [cache scan empty] exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "0 repo(s)" in out

    def test_cache_scan_with_cache_dir(self, mock_api, run_cli):
        """cache scan with --cache-dir."""
        exit_code, out, err = run_cli(
            ["cache", "scan", "--cache-dir", "/custom/cache"]
        )
        print(f"\n** [cache scan --cache-dir] exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        mock_api.scan_cache.assert_called_once_with("/custom/cache")


class TestCacheClear:
    """Verify cache clear subcommand."""

    def test_cache_clear_all_confirmed(self, mock_api, run_cli):
        """cache clear with confirmation 'y' clears everything."""
        with patch("builtins.input", return_value="y"):
            exit_code, out, err = run_cli(["cache", "clear"])
        print(f"\n** [cache clear confirmed] exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Freed" in out
        mock_api.clear_cache.assert_called_once_with(
            cache_dir=None,
            repo_type=None,
            repo_id=None,
        )

    def test_cache_clear_all_cancelled(self, mock_api, run_cli):
        """cache clear with confirmation 'n' aborts."""
        with patch("builtins.input", return_value="n"):
            exit_code, out, err = run_cli(["cache", "clear"])
        print(f"\n** [cache clear cancelled] exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Aborted" in out
        mock_api.clear_cache.assert_not_called()

    def test_cache_clear_by_type(self, mock_api, run_cli):
        """cache clear --repo-type model."""
        with patch("builtins.input", return_value="y"):
            exit_code, out, err = run_cli(
                ["cache", "clear", "--repo-type", "model"]
            )
        print(f"\n** [cache clear --repo-type] exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        mock_api.clear_cache.assert_called_once_with(
            cache_dir=None,
            repo_type="model",
            repo_id=None,
        )

    def test_cache_clear_repo_id_without_type_error(self, mock_api, run_cli):
        """cache clear --repo-id without --repo-type exits 2."""
        exit_code, out, err = run_cli(
            ["cache", "clear", "--repo-id", "owner/model", "--yes"]
        )
        print(f"\n** [cache clear missing type] exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 2
        assert "repo-type" in err.lower()

    def test_cache_clear_force_yes(self, mock_api, run_cli):
        """cache clear --yes skips confirmation."""
        exit_code, out, err = run_cli(["cache", "clear", "--yes"])
        print(f"\n** [cache clear --yes] exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Freed" in out
        mock_api.clear_cache.assert_called_once()

    def test_cache_clear_specific_repo(self, mock_api, run_cli):
        """cache clear with --repo-type and --repo-id."""
        exit_code, out, err = run_cli([
            "cache", "clear",
            "--repo-type", "model",
            "--repo-id", "owner/model1",
            "--yes",
        ])
        print(f"\n** [cache clear specific] exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        mock_api.clear_cache.assert_called_once_with(
            cache_dir=None,
            repo_type="model",
            repo_id="owner/model1",
        )
