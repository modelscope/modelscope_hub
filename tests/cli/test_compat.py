"""Unit tests for CLI backward compatibility.

These tests verify that the argparse layer accepts both legacy and new-style
arguments without making any remote API calls.
"""

from __future__ import annotations

import warnings

import pytest

from modelscope_hub.cli.main import _build_parser
from modelscope_hub.cli.compat import normalize_download_args, normalize_patterns


# ---------------------------------------------------------------------------
# Parser fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def parser():
    return _build_parser()


# ---------------------------------------------------------------------------
# Download: legacy argument styles
# ---------------------------------------------------------------------------
class TestDownloadLegacyArgs:
    def test_legacy_model_flag(self, parser):
        """ms download --model owner/repo"""
        args = parser.parse_args(["download", "--model", "owner/repo"])
        assert args.model == "owner/repo"

    def test_legacy_dataset_flag(self, parser):
        """ms download --dataset owner/repo"""
        args = parser.parse_args(["download", "--dataset", "owner/repo"])
        assert args.dataset == "owner/repo"

    def test_legacy_local_dir_underscore(self, parser):
        """ms download --model owner/repo --local_dir ./tmp"""
        args = parser.parse_args([
            "download", "--model", "owner/repo", "--local_dir", "./tmp"
        ])
        assert args.local_dir_legacy == "./tmp"

    def test_legacy_cache_dir_underscore(self, parser):
        """ms download --model owner/repo --cache_dir /cache"""
        args = parser.parse_args([
            "download", "--model", "owner/repo", "--cache_dir", "/cache"
        ])
        assert args.cache_dir_legacy == "/cache"

    def test_new_style_positional(self, parser):
        """ms download owner/repo --repo-type dataset --local-dir ./tmp"""
        args = parser.parse_args([
            "download", "owner/repo", "--repo-type", "dataset",
            "--local-dir", "./tmp",
        ])
        assert args.repo_id == "owner/repo"
        assert args.repo_type == "dataset"
        assert args.local_dir == "./tmp"

    def test_new_style_with_files(self, parser):
        """ms download owner/repo file1.bin file2.bin"""
        args = parser.parse_args(["download", "owner/repo", "file1.bin", "file2.bin"])
        assert args.repo_id == "owner/repo"
        assert args.files == ["file1.bin", "file2.bin"]

    def test_legacy_dataset_with_local_dir(self, parser):
        """ms download --dataset owner/repo --local_dir ./temp (the user's failing command)"""
        args = parser.parse_args([
            "download", "--dataset", "wangxingjun778/self_cog_data",
            "--local_dir", "./temp",
        ])
        assert args.dataset == "wangxingjun778/self_cog_data"
        assert args.local_dir_legacy == "./temp"

    def test_repo_type_underscore(self, parser):
        """ms download owner/repo --repo_type dataset"""
        args = parser.parse_args([
            "download", "owner/repo", "--repo_type", "dataset",
        ])
        assert args.repo_type == "dataset"

    def test_subcmd_token_endpoint(self, parser):
        """Legacy per-subcommand --token/--endpoint are accepted."""
        args = parser.parse_args([
            "download", "--model", "owner/repo",
            "--token", "ms-xxx", "--endpoint", "https://custom.cn",
        ])
        assert args.subcmd_token == "ms-xxx"
        assert args.subcmd_endpoint == "https://custom.cn"


# ---------------------------------------------------------------------------
# Download: normalize_download_args
# ---------------------------------------------------------------------------
class TestNormalizeDownloadArgs:
    def test_model_to_repo_id(self, parser):
        args = parser.parse_args(["download", "--model", "owner/repo"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            normalize_download_args(args)
        assert args.repo_id == "owner/repo"
        assert args.repo_type == "model"

    def test_dataset_to_repo_id(self, parser):
        args = parser.parse_args(["download", "--dataset", "owner/data"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            normalize_download_args(args)
        assert args.repo_id == "owner/data"
        assert args.repo_type == "dataset"

    def test_local_dir_legacy_merged(self, parser):
        args = parser.parse_args([
            "download", "--model", "owner/repo", "--local_dir", "/tmp/out",
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            normalize_download_args(args)
        assert args.local_dir == "/tmp/out"

    def test_missing_repo_id_raises(self, parser):
        args = parser.parse_args(["download"])
        with pytest.raises(ValueError, match="repo_id is required"):
            normalize_download_args(args)

    def test_model_with_positional_becomes_file(self, parser):
        """ms download --model owner/repo file.bin → file.bin is a file to download."""
        args = parser.parse_args(["download", "file.bin", "--model", "owner/repo"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            normalize_download_args(args)
        assert args.repo_id == "owner/repo"
        assert "file.bin" in args.files

    def test_model_overrides_explicit_repo_type(self, parser):
        """--model always implies repo_type='model', even if --repo-type is given."""
        args = parser.parse_args(["download", "--model", "o/r", "--repo-type", "dataset"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            normalize_download_args(args)
        assert args.repo_type == "model"

    def test_collection_to_repo_id(self, parser):
        """--collection maps to repo_id with repo_type='collection'."""
        args = parser.parse_args(["download", "--collection", "my_collection"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            normalize_download_args(args)
        assert args.repo_id == "my_collection"
        assert args.repo_type == "collection"


# ---------------------------------------------------------------------------
# Include/Exclude patterns
# ---------------------------------------------------------------------------
class TestPatternAction:
    def test_multi_value(self, parser):
        """--include a b c"""
        args = parser.parse_args(["download", "owner/repo", "--include", "*.bin", "*.json"])
        assert args.allow_patterns == ["*.bin", "*.json"]

    def test_repeated(self, parser):
        """--include a --include b"""
        args = parser.parse_args([
            "download", "owner/repo", "--include", "*.bin", "--include", "*.json"
        ])
        assert "*.bin" in args.allow_patterns
        assert "*.json" in args.allow_patterns

    def test_normalize_patterns_none(self):
        assert normalize_patterns(None) is None

    def test_normalize_patterns_string(self):
        assert normalize_patterns("*.bin") == ["*.bin"]

    def test_normalize_patterns_flat_list(self):
        assert normalize_patterns(["a", "b"]) == ["a", "b"]

    def test_normalize_patterns_nested(self):
        assert normalize_patterns([["a", "b"], "c"]) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Top-level aliases
# ---------------------------------------------------------------------------
class TestAliases:
    def test_scan_cache_alias(self, parser):
        args = parser.parse_args(["scan-cache"])
        assert hasattr(args, "_command")

    def test_clear_cache_alias(self, parser):
        args = parser.parse_args(["clear-cache", "--model", "owner/repo"])
        assert args.model == "owner/repo"
        assert hasattr(args, "cache_dir")

    def test_clear_cache_with_cache_dir(self, parser):
        args = parser.parse_args(["clear-cache", "--model", "owner/repo", "--cache-dir", "/tmp"])
        assert args.cache_dir == "/tmp"

    def test_clear_cache_dataset(self, parser):
        args = parser.parse_args(["clear-cache", "--dataset", "owner/data"])
        assert args.dataset == "owner/data"

    def test_create_alias(self, parser):
        args = parser.parse_args(["create", "owner/repo", "--repo-type", "model"])
        assert args.repo_id == "owner/repo"
        assert args.repo_type == "model"

    def test_create_exist_ok(self, parser):
        args = parser.parse_args([
            "create", "owner/repo", "--repo-type", "model", "--exist-ok"
        ])
        assert args.exist_ok is True

    def test_create_exist_ok_underscore(self, parser):
        args = parser.parse_args([
            "create", "owner/repo", "--repo-type", "model", "--exist_ok"
        ])
        assert args.exist_ok is True


# ---------------------------------------------------------------------------
# Upload backward compat
# ---------------------------------------------------------------------------
class TestUploadCompat:
    def test_commit_description(self, parser):
        args = parser.parse_args([
            "upload", "owner/repo", ".", "--commit-description", "some desc"
        ])
        assert args.commit_description == "some desc"

    def test_include_multi(self, parser):
        args = parser.parse_args([
            "upload", "owner/repo", ".", "--include", "*.py", "*.txt"
        ])
        assert args.allow_patterns == ["*.py", "*.txt"]

    def test_subcmd_token(self, parser):
        args = parser.parse_args([
            "upload", "owner/repo", ".",
            "--token", "ms-abc", "--endpoint", "https://x.cn",
        ])
        assert args.subcmd_token == "ms-abc"
        assert args.subcmd_endpoint == "https://x.cn"


# ---------------------------------------------------------------------------
# Version flag
# ---------------------------------------------------------------------------
class TestVersionFlag:
    def test_version_short(self, parser, capsys):
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["-V"])
        assert exc_info.value.code == 0

    def test_version_long(self, parser, capsys):
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Repo create --exist-ok
# ---------------------------------------------------------------------------
class TestRepoCreateExistOk:
    def test_exist_ok_in_repo_create(self, parser):
        args = parser.parse_args([
            "repo", "create", "owner/repo", "--repo-type", "model", "--exist-ok"
        ])
        assert args.exist_ok is True

    def test_exist_ok_underscore(self, parser):
        args = parser.parse_args([
            "repo", "create", "owner/repo", "--repo-type", "model", "--exist_ok"
        ])
        assert args.exist_ok is True
