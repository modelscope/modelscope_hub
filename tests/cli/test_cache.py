"""Tests for ``ms cache`` group — scan and clear.

Includes:
- Parser tests: all flags and subcommands
- Execution tests: mock HubApi for scan/clear logic
- Legacy alias tests: scan-cache, clear-cache
- Remote tests: real API cache lifecycle (existing)
"""
from __future__ import annotations

import base64
import warnings
from unittest.mock import patch

import pytest

from modelscope_hub.cli.cache import CacheCommand, _CacheClear, _CacheScan
from modelscope_hub.types import CacheInfo

from .conftest import run_cli


# ===================================================================
# Parser tests
# ===================================================================
class TestCacheScanParser:
    """``ms cache scan`` argument parsing."""

    def test_basic(self, parser):
        args = parser.parse_args(["cache", "scan"])
        assert hasattr(args, "_cache_leaf")

    def test_cache_dir(self, parser):
        args = parser.parse_args(["cache", "scan", "--cache-dir", "/data/cache"])
        assert args.cache_dir == "/data/cache"

    def test_cache_dir_default_none(self, parser):
        args = parser.parse_args(["cache", "scan"])
        assert args.cache_dir is None


class TestCacheClearParser:
    """``ms cache clear`` argument parsing."""

    def test_basic_with_yes(self, parser):
        args = parser.parse_args(["cache", "clear", "--yes"])
        assert args.yes is True

    def test_repo_type(self, parser):
        args = parser.parse_args([
            "cache", "clear", "--repo-type", "model", "--yes",
        ])
        assert args.repo_type == "model"

    @pytest.mark.parametrize("repo_type", ["model", "dataset", "studio", "skill", "mcp"])
    def test_all_repo_types(self, parser, repo_type):
        args = parser.parse_args([
            "cache", "clear", "--repo-type", repo_type, "--yes",
        ])
        assert args.repo_type == repo_type

    def test_repo_id(self, parser):
        args = parser.parse_args([
            "cache", "clear",
            "--repo-type", "model", "--repo-id", "owner/model1",
            "--yes",
        ])
        assert args.repo_id == "owner/model1"

    def test_cache_dir(self, parser):
        args = parser.parse_args([
            "cache", "clear", "--cache-dir", "/data/cache", "--yes",
        ])
        assert args.cache_dir == "/data/cache"

    def test_yes_short_flag(self, parser):
        args = parser.parse_args(["cache", "clear", "-y"])
        assert args.yes is True

    def test_yes_default_false(self, parser):
        args = parser.parse_args(["cache", "clear"])
        assert args.yes is False


class TestCacheLegacyAliases:
    """Legacy ``ms scan-cache`` / ``ms clear-cache`` aliases."""

    def test_scan_cache_alias(self, parser):
        args = parser.parse_args(["scan-cache"])
        assert hasattr(args, "_command")

    def test_scan_cache_with_cache_dir(self, parser):
        args = parser.parse_args(["scan-cache", "--cache-dir", "/data"])
        assert args.cache_dir == "/data"

    def test_scan_cache_with_dir_alias(self, parser):
        args = parser.parse_args(["scan-cache", "--dir", "/data"])
        assert args.cache_dir == "/data"

    def test_clear_cache_model(self, parser):
        args = parser.parse_args(["clear-cache", "--model", "owner/repo"])
        assert args.model == "owner/repo"

    def test_clear_cache_dataset(self, parser):
        args = parser.parse_args(["clear-cache", "--dataset", "owner/data"])
        assert args.dataset == "owner/data"

    def test_clear_cache_with_cache_dir(self, parser):
        args = parser.parse_args([
            "clear-cache", "--model", "owner/repo", "--cache-dir", "/tmp",
        ])
        assert args.cache_dir == "/tmp"

    def test_clear_cache_yes_flag(self, parser):
        args = parser.parse_args(["clear-cache", "--model", "o/r", "--yes"])
        assert args.yes is True

    def test_clear_cache_model_dataset_mutually_exclusive(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["clear-cache", "--model", "a", "--dataset", "b"])


# ===================================================================
# Execution tests — mock HubApi
# ===================================================================
class TestCacheScanExecute:
    """_CacheScan.execute() logic."""

    def test_scan_with_repos(self, parser, mock_api, capsys):
        args = parser.parse_args(["cache", "scan"])
        with patch("modelscope_hub.cli.cache.make_api", return_value=mock_api):
            _CacheScan(args).execute()
        mock_api.scan_cache.assert_called_once_with(None)
        out = capsys.readouterr().out
        assert "1 repo(s)" in out
        assert "owner/repo" in out

    def test_scan_with_cache_dir(self, parser, mock_api, capsys):
        args = parser.parse_args(["cache", "scan", "--cache-dir", "/data"])
        with patch("modelscope_hub.cli.cache.make_api", return_value=mock_api):
            _CacheScan(args).execute()
        mock_api.scan_cache.assert_called_once_with("/data")

    def test_scan_empty(self, parser, mock_api, capsys):
        mock_api.scan_cache.return_value = CacheInfo(
            cache_dir="/tmp/cache", total_size=0, repos=[],
        )
        args = parser.parse_args(["cache", "scan"])
        with patch("modelscope_hub.cli.cache.make_api", return_value=mock_api):
            _CacheScan(args).execute()
        out = capsys.readouterr().out
        assert "0 repo(s)" in out


class TestCacheClearExecute:
    """_CacheClear.execute() logic."""

    def test_clear_with_yes(self, parser, mock_api, capsys):
        args = parser.parse_args([
            "cache", "clear", "--repo-type", "model", "--yes",
        ])
        with patch("modelscope_hub.cli.cache.make_api", return_value=mock_api):
            _CacheClear(args).execute()
        mock_api.clear_cache.assert_called_once()
        out = capsys.readouterr().out
        assert "Freed" in out

    def test_clear_aborted(self, parser, mock_api, capsys):
        args = parser.parse_args(["cache", "clear"])
        with (
            patch("modelscope_hub.cli.cache.make_api", return_value=mock_api),
            patch("builtins.input", return_value="n"),
        ):
            _CacheClear(args).execute()
        mock_api.clear_cache.assert_not_called()
        out = capsys.readouterr().out
        assert "Aborted" in out

    def test_clear_confirmed_interactively(self, parser, mock_api, capsys):
        args = parser.parse_args(["cache", "clear"])
        with (
            patch("modelscope_hub.cli.cache.make_api", return_value=mock_api),
            patch("builtins.input", return_value="yes"),
        ):
            _CacheClear(args).execute()
        mock_api.clear_cache.assert_called_once()

    def test_clear_repo_id_without_type_exits(self, parser, mock_api, capsys):
        args = parser.parse_args(["cache", "clear", "--repo-id", "o/r", "--yes"])
        with patch("modelscope_hub.cli.cache.make_api", return_value=mock_api):
            with pytest.raises(SystemExit) as exc_info:
                _CacheClear(args).execute()
            assert exc_info.value.code == 2

    def test_clear_specific_repo(self, parser, mock_api, capsys):
        args = parser.parse_args([
            "cache", "clear",
            "--repo-type", "model", "--repo-id", "owner/repo",
            "--yes",
        ])
        with patch("modelscope_hub.cli.cache.make_api", return_value=mock_api):
            _CacheClear(args).execute()
        kw = mock_api.clear_cache.call_args.kwargs
        assert kw["repo_type"] == "model"
        assert kw["repo_id"] == "owner/repo"

    def test_clear_cache_dir_forwarded(self, parser, mock_api, capsys):
        args = parser.parse_args([
            "cache", "clear", "--cache-dir", "/data", "--yes",
        ])
        with patch("modelscope_hub.cli.cache.make_api", return_value=mock_api):
            _CacheClear(args).execute()
        assert mock_api.clear_cache.call_args.kwargs["cache_dir"] == "/data"


# ===================================================================
# Remote integration tests (existing)
# ===================================================================
@pytest.mark.remote
class TestCacheLifecycle:
    """Download a file to populate the cache, then scan and clear it.

    Uses a real remote repo + real local cache directory (tmp_path).
    """

    @pytest.fixture(autouse=True, scope="class")
    def setup_repo(self, api, test_owner, test_endpoint, repo_name):
        """Create a model repo, commit a test file, download to cache."""
        cls = type(self)
        cls.repo_id = f"{test_owner}/{repo_name}_cache"
        print(f"\n** Setup: creating repo {cls.repo_id}")
        api.create_repo(cls.repo_id, "model", visibility="private")

        file_bytes = b"cache test content for scan and clear"
        content_b64 = base64.b64encode(file_bytes).decode()
        api.legacy.create_commit(
            repo_id=cls.repo_id,
            repo_type="model",
            operations=[{
                "action": "create",
                "path": "cache_data.txt",
                "type": "normal",
                "size": len(file_bytes),
                "sha256": "",
                "content": content_b64,
                "encoding": "base64",
            }],
            commit_message="Add test file for cache tests",
            revision="master",
        )
        cls.api = api
        print("** Setup: done")
        yield
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                api.delete_repo(cls.repo_id, "model")
            except Exception as e:
                print(f"** Teardown: cleanup via web console: {cls.repo_id} ({e})")

    def test_01_download_populates_cache(self, test_token, test_endpoint, tmp_path_factory):
        """Download snapshot to a shared cache dir for subsequent tests."""
        cls = type(self)
        cls.cache_dir = str(tmp_path_factory.mktemp("cache"))

        exit_code, out, err = run_cli(
            ["download", self.repo_id, "--cache-dir", cls.cache_dir],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [download] exit_code={exit_code}, out={out!r}")
        assert exit_code == 0
        assert "Snapshot ready" in out

    def test_02_cache_scan_shows_repo(self, test_token, test_endpoint):
        """cache scan --cache-dir should list the downloaded repo."""
        exit_code, out, err = run_cli(
            ["cache", "scan", "--cache-dir", self.cache_dir],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [cache scan] exit_code={exit_code}, out={out!r}")
        assert exit_code == 0
        repo_name_part = self.repo_id.split("/")[-1]
        assert repo_name_part in out
        assert "1 repo(s)" in out

    def test_03_cache_clear_by_type(self, test_token, test_endpoint):
        """cache clear --repo-type model frees bytes."""
        exit_code, out, err = run_cli(
            ["cache", "clear", "--repo-type", "model", "--cache-dir", self.cache_dir, "--yes"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [cache clear model] exit_code={exit_code}, out={out!r}")
        assert exit_code == 0
        assert "Freed" in out

    def test_04_cache_scan_empty_after_clear(self, test_token, test_endpoint):
        """cache scan should show 0 repos after clear."""
        exit_code, out, err = run_cli(
            ["cache", "scan", "--cache-dir", self.cache_dir],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [cache scan empty] exit_code={exit_code}, out={out!r}")
        assert exit_code == 0
        assert "0 repo(s)" in out

    def test_05_download_then_clear_specific_repo(self, test_token, test_endpoint, tmp_path_factory):
        """Download again, then clear only the specific repo by id."""
        cache2 = str(tmp_path_factory.mktemp("cache2"))

        exit_code, out, err = run_cli(
            ["download", self.repo_id, "--cache-dir", cache2],
            token=test_token,
            endpoint=test_endpoint,
        )
        assert exit_code == 0

        exit_code, out, err = run_cli(
            ["cache", "clear",
             "--repo-type", "model",
             "--repo-id", self.repo_id,
             "--cache-dir", cache2,
             "--yes"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [cache clear specific] exit_code={exit_code}, out={out!r}")
        assert exit_code == 0
        assert "Freed" in out

        exit_code, out, err = run_cli(
            ["cache", "scan", "--cache-dir", cache2],
            token=test_token,
            endpoint=test_endpoint,
        )
        assert exit_code == 0
        assert "0 repo(s)" in out

    def test_06_cache_clear_repo_id_without_type_error(self, test_token, test_endpoint):
        """cache clear --repo-id without --repo-type exits 2."""
        exit_code, out, err = run_cli(
            ["cache", "clear", "--repo-id", "owner/model", "--yes"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [clear missing type] exit_code={exit_code}, err={err!r}")
        assert exit_code == 2
        assert "repo-type" in err.lower()
