"""Unit tests for CLI backward compatibility.

Tests the normalization layer (normalize_download_args, normalize_patterns)
and edge cases in legacy argument handling. Per-command parser and execution
tests live in their respective ``test_<command>.py`` files.
"""

from __future__ import annotations

import warnings

import pytest

from modelscope_hub.cli.compat import normalize_download_args, normalize_patterns


# ---------------------------------------------------------------------------
# Download: legacy argument edge cases
# ---------------------------------------------------------------------------
class TestDownloadLegacyEdgeCases:
    def test_legacy_dataset_with_local_dir(self, parser):
        """ms download --dataset owner/repo --local_dir ./temp (regression test)"""
        args = parser.parse_args([
            "download", "--dataset", "wangxingjun778/self_cog_data",
            "--local_dir", "./temp",
        ])
        assert args.dataset == "wangxingjun778/self_cog_data"
        assert args.local_dir_legacy == "./temp"


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
        """ms download --model owner/repo file.bin -> file.bin is a file to download."""
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
# Include/Exclude patterns — normalize_patterns function
# ---------------------------------------------------------------------------
class TestNormalizePatterns:
    def test_none(self):
        assert normalize_patterns(None) is None

    def test_string(self):
        assert normalize_patterns("*.bin") == ["*.bin"]

    def test_flat_list(self):
        assert normalize_patterns(["a", "b"]) == ["a", "b"]

    def test_nested(self):
        assert normalize_patterns([["a", "b"], "c"]) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Version flag (short -V form, long --version is tested in test_main.py)
# ---------------------------------------------------------------------------
class TestVersionFlag:
    def test_version_short(self, parser):
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["-V"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Cross-cutting backward compat: --repo_type (underscore) in multiple commands
# ---------------------------------------------------------------------------
class TestRepoTypeUnderscore:
    @pytest.mark.parametrize("cmd,expected_type", [
        (["info", "o/r", "--repo_type", "dataset"], "dataset"),
        (["list", "--repo_type", "model"], "model"),
        (["delete", "o/r", "--repo_type", "model"], "model"),
    ])
    def test_repo_type_underscore_in_all_commands(self, parser, cmd, expected_type):
        """--repo_type (underscore) works in info, list, delete."""
        args = parser.parse_args(cmd)
        assert args.repo_type == expected_type
