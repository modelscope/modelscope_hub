"""Tests for ``ms upload`` command — file and folder modes."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from modelscope_hub.errors import HubError


class TestUploadFile:
    """Verify single-file upload path."""

    def test_upload_file(self, mock_api, run_cli, tmp_path):
        """Upload a single file successfully."""
        test_file = tmp_path / "model.bin"
        test_file.write_bytes(b"fake model data")

        exit_code, out, err = run_cli(
            ["upload", "owner/repo", str(test_file)]
        )
        assert exit_code == 0
        assert "Upload complete" in out
        mock_api.upload_file.assert_called_once()
        call_args = mock_api.upload_file.call_args
        assert call_args[0][0] == "owner/repo"
        assert call_args[0][2] == str(test_file)
        assert call_args[0][3] == "model.bin"  # basename as path_in_repo

    def test_upload_with_commit_message(self, mock_api, run_cli, tmp_path):
        """Upload with --commit-message flag."""
        test_file = tmp_path / "weights.safetensors"
        test_file.write_bytes(b"data")

        exit_code, out, err = run_cli([
            "upload", "owner/repo", str(test_file),
            "--commit-message", "Add weights",
        ])
        assert exit_code == 0
        call_kwargs = mock_api.upload_file.call_args[1]
        assert call_kwargs["commit_message"] == "Add weights"

    def test_upload_path_not_exists(self, mock_api, run_cli):
        """Upload exits 2 when local path does not exist."""
        exit_code, out, err = run_cli(
            ["upload", "owner/repo", "/nonexistent/path/file.bin"]
        )
        assert exit_code == 2
        assert "not found" in err.lower()


class TestUploadDirectory:
    """Verify folder upload path."""

    def test_upload_directory(self, mock_api, run_cli, tmp_path):
        """Upload a directory successfully."""
        (tmp_path / "file1.txt").write_text("a")
        (tmp_path / "file2.txt").write_text("b")

        exit_code, out, err = run_cli(
            ["upload", "owner/repo", str(tmp_path)]
        )
        assert exit_code == 0
        assert "Folder upload complete" in out
        mock_api.upload_folder.assert_called_once()
        call_args = mock_api.upload_folder.call_args
        assert call_args[0][0] == "owner/repo"
        assert call_args[0][2] == str(tmp_path)

    def test_upload_with_patterns(self, mock_api, run_cli, tmp_path):
        """Upload folder with --include and --exclude patterns."""
        (tmp_path / "file.txt").write_text("a")

        exit_code, out, err = run_cli([
            "upload", "owner/repo", str(tmp_path),
            "--include", "*.py",
            "--exclude", "__pycache__",
        ])
        assert exit_code == 0
        call_kwargs = mock_api.upload_folder.call_args[1]
        assert call_kwargs["allow_patterns"] == ["*.py"]
        assert call_kwargs["ignore_patterns"] == ["__pycache__"]

    def test_upload_with_path_in_repo(self, mock_api, run_cli, tmp_path):
        """Upload folder with explicit path_in_repo."""
        (tmp_path / "file.txt").write_text("a")

        exit_code, out, err = run_cli(
            ["upload", "owner/repo", str(tmp_path), "subdir/"]
        )
        assert exit_code == 0
        call_kwargs = mock_api.upload_folder.call_args[1]
        assert call_kwargs["path_in_repo"] == "subdir/"

    def test_upload_api_error(self, mock_api, run_cli, tmp_path):
        """Upload exits 1 on API HubError."""
        test_file = tmp_path / "f.bin"
        test_file.write_bytes(b"data")
        mock_api.upload_file.side_effect = HubError("upload failed")
        exit_code, out, err = run_cli(
            ["upload", "owner/repo", str(test_file)]
        )
        assert exit_code == 1
        assert "upload failed" in err
