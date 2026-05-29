"""Tests for ``ms cache`` group — scan and clear."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from modelscope_hub.types import CachedRepoInfo, CacheInfo


class TestCacheScan:
    """Verify cache scan subcommand."""

    def test_cache_scan(self, mock_api, run_cli):
        """cache scan displays summary and table."""
        exit_code, out, err = run_cli(["cache", "scan"])
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
        assert exit_code == 0
        assert "0 repo(s)" in out

    def test_cache_scan_with_cache_dir(self, mock_api, run_cli):
        """cache scan with --cache-dir."""
        exit_code, out, err = run_cli(
            ["cache", "scan", "--cache-dir", "/custom/cache"]
        )
        assert exit_code == 0
        mock_api.scan_cache.assert_called_once_with("/custom/cache")


class TestCacheClear:
    """Verify cache clear subcommand."""

    def test_cache_clear_all_confirmed(self, mock_api, run_cli):
        """cache clear with confirmation 'y' clears everything."""
        with patch("builtins.input", return_value="y"):
            exit_code, out, err = run_cli(["cache", "clear"])
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
        assert exit_code == 0
        assert "Aborted" in out
        mock_api.clear_cache.assert_not_called()

    def test_cache_clear_by_type(self, mock_api, run_cli):
        """cache clear --repo-type model."""
        with patch("builtins.input", return_value="y"):
            exit_code, out, err = run_cli(
                ["cache", "clear", "--repo-type", "model"]
            )
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
        assert exit_code == 2
        assert "repo-type" in err.lower()

    def test_cache_clear_force_yes(self, mock_api, run_cli):
        """cache clear --yes skips confirmation."""
        exit_code, out, err = run_cli(["cache", "clear", "--yes"])
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
        assert exit_code == 0
        mock_api.clear_cache.assert_called_once_with(
            cache_dir=None,
            repo_type="model",
            repo_id="owner/model1",
        )
