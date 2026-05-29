"""Tests for ``ms download`` command — single file and snapshot modes."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import call

import pytest

from modelscope_hub.errors import NotFoundError


class TestDownloadSingleFile:
    """Verify single-file download paths."""

    def test_download_single_file(self, mock_api, run_cli):
        """Download a single file successfully."""
        exit_code, out, err = run_cli(
            ["download", "owner/model", "config.json"]
        )
        assert exit_code == 0
        assert "config.json" in out
        mock_api.download_file.assert_called_once_with(
            "owner/model",
            "model",
            "config.json",
            revision=None,
            cache_dir=None,
            force=False,
        )

    def test_download_multiple_files(self, mock_api, run_cli):
        """Download multiple files in one invocation."""
        mock_api.download_file.return_value = Path("/tmp/cached/file.bin")
        exit_code, out, err = run_cli(
            ["download", "owner/model", "config.json", "model.safetensors"]
        )
        assert exit_code == 0
        assert mock_api.download_file.call_count == 2
        calls = mock_api.download_file.call_args_list
        assert calls[0][0][2] == "config.json"
        assert calls[1][0][2] == "model.safetensors"

    def test_download_with_revision(self, mock_api, run_cli):
        """Download with --revision flag."""
        exit_code, out, err = run_cli(
            ["download", "owner/model", "file.bin", "--revision", "v1.0"]
        )
        assert exit_code == 0
        mock_api.download_file.assert_called_once_with(
            "owner/model",
            "model",
            "file.bin",
            revision="v1.0",
            cache_dir=None,
            force=False,
        )

    def test_download_with_cache_dir(self, mock_api, run_cli):
        """Download with --cache-dir flag."""
        exit_code, out, err = run_cli(
            ["download", "owner/model", "f.bin", "--cache-dir", "/my/cache"]
        )
        assert exit_code == 0
        mock_api.download_file.assert_called_once_with(
            "owner/model",
            "model",
            "f.bin",
            revision=None,
            cache_dir=Path("/my/cache"),
            force=False,
        )


class TestDownloadSnapshot:
    """Verify full-repo snapshot download."""

    def test_download_full_repo(self, mock_api, run_cli):
        """Download full repo snapshot when no files specified."""
        exit_code, out, err = run_cli(["download", "owner/model"])
        assert exit_code == 0
        assert "Snapshot ready" in out
        mock_api.download_repo.assert_called_once_with(
            "owner/model",
            "model",
            revision=None,
            cache_dir=None,
            allow_patterns=None,
            ignore_patterns=None,
            max_workers=4,
        )

    def test_download_with_patterns(self, mock_api, run_cli):
        """Download with --include and --exclude patterns."""
        exit_code, out, err = run_cli([
            "download", "owner/model",
            "--include", "*.safetensors",
            "--exclude", "*.bin",
        ])
        assert exit_code == 0
        mock_api.download_repo.assert_called_once_with(
            "owner/model",
            "model",
            revision=None,
            cache_dir=None,
            allow_patterns=["*.safetensors"],
            ignore_patterns=["*.bin"],
            max_workers=4,
        )

    def test_download_api_error(self, mock_api, run_cli):
        """Download exits 1 on API NotFoundError."""
        mock_api.download_repo.side_effect = NotFoundError(
            "Repo not found", status_code=404
        )
        exit_code, out, err = run_cli(["download", "owner/ghost"])
        assert exit_code == 1
        assert "not found" in err.lower() or "Repo" in err

    def test_download_dataset_type(self, mock_api, run_cli):
        """Download with --repo-type dataset."""
        exit_code, out, err = run_cli(
            ["download", "owner/data", "--repo-type", "dataset"]
        )
        assert exit_code == 0
        mock_api.download_repo.assert_called_once()
        assert mock_api.download_repo.call_args[0][1] == "dataset"
