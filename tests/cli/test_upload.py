"""Tests for ``ms upload`` command.

Includes:
- Parser tests: all flags and choices
- Execution tests: mock HubApi for file/folder upload logic
- Remote tests: real API upload (existing)
"""
from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import patch

import pytest

from modelscope_hub.cli.upload import UploadCommand

from .conftest import run_cli


# ===================================================================
# Parser tests
# ===================================================================
class TestUploadParser:
    """``ms upload`` argument parsing."""

    def test_basic(self, parser):
        args = parser.parse_args(["upload", "owner/repo", "./weights.bin"])
        assert args.repo_id == "owner/repo"
        assert args.local_path == "./weights.bin"

    def test_with_path_in_repo(self, parser):
        args = parser.parse_args(["upload", "owner/repo", "./output", "models/"])
        assert args.local_path == "./output"
        assert args.path_in_repo == "models/"

    def test_repo_type_default_model(self, parser):
        args = parser.parse_args(["upload", "owner/repo", "."])
        assert args.repo_type == "model"

    @pytest.mark.parametrize("repo_type", ["model", "dataset"])
    def test_repo_type_choices(self, parser, repo_type):
        args = parser.parse_args(["upload", "o/r", ".", "--repo-type", repo_type])
        assert args.repo_type == repo_type

    def test_invalid_repo_type_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["upload", "o/r", ".", "--repo-type", "studio"])

    def test_commit_message(self, parser):
        args = parser.parse_args([
            "upload", "o/r", ".", "--commit-message", "add weights",
        ])
        assert args.commit_message == "add weights"

    def test_commit_description(self, parser):
        args = parser.parse_args([
            "upload", "o/r", ".", "--commit-description", "extended desc",
        ])
        assert args.commit_description == "extended desc"

    def test_revision(self, parser):
        args = parser.parse_args(["upload", "o/r", ".", "--revision", "dev"])
        assert args.revision == "dev"

    def test_include_patterns(self, parser):
        args = parser.parse_args(["upload", "o/r", ".", "--include", "*.py", "*.txt"])
        assert args.allow_patterns == ["*.py", "*.txt"]

    def test_exclude_patterns(self, parser):
        args = parser.parse_args(["upload", "o/r", ".", "--exclude", "*.ckpt", "*.bin"])
        assert args.ignore_patterns == ["*.ckpt", "*.bin"]

    def test_max_workers(self, parser):
        args = parser.parse_args(["upload", "o/r", ".", "--max-workers", "8"])
        assert args.max_workers == 8

    def test_max_workers_default_none(self, parser):
        args = parser.parse_args(["upload", "o/r", "."])
        assert args.max_workers is None

    def test_use_cache_default_true(self, parser):
        args = parser.parse_args(["upload", "o/r", "."])
        assert args.use_cache is True

    def test_no_cache(self, parser):
        args = parser.parse_args(["upload", "o/r", ".", "--no-cache"])
        assert args.use_cache is False

    def test_use_cache_explicit(self, parser):
        args = parser.parse_args(["upload", "o/r", ".", "--use-cache"])
        assert args.use_cache is True

    def test_disable_tqdm(self, parser):
        args = parser.parse_args(["upload", "o/r", ".", "--disable-tqdm"])
        assert args.disable_tqdm is True

    def test_disable_tqdm_default_false(self, parser):
        args = parser.parse_args(["upload", "o/r", "."])
        assert args.disable_tqdm is False

    def test_subcmd_token_endpoint(self, parser):
        args = parser.parse_args([
            "upload", "o/r", ".",
            "--token", "ms-tok", "--endpoint", "https://x.cn",
        ])
        assert args.subcmd_token == "ms-tok"
        assert args.subcmd_endpoint == "https://x.cn"

    def test_all_options_combined(self, parser):
        args = parser.parse_args([
            "upload", "my-org/my-model", "./output", "weights/",
            "--repo-type", "dataset",
            "--commit-message", "v2",
            "--commit-description", "retrained",
            "--revision", "dev",
            "--include", "*.safetensors",
            "--exclude", "*.ckpt",
            "--max-workers", "4",
            "--no-cache",
            "--disable-tqdm",
        ])
        assert args.repo_id == "my-org/my-model"
        assert args.local_path == "./output"
        assert args.path_in_repo == "weights/"
        assert args.repo_type == "dataset"
        assert args.use_cache is False
        assert args.disable_tqdm is True


# ===================================================================
# Execution tests — mock HubApi
# ===================================================================
class TestUploadExecute:
    """UploadCommand.execute() logic with mocked API."""

    def test_file_upload(self, parser, mock_api, tmp_path, capsys):
        test_file = tmp_path / "weights.bin"
        test_file.write_text("content")
        args = parser.parse_args(["upload", "owner/repo", str(test_file)])
        with patch("modelscope_hub.cli.upload.make_api", return_value=mock_api):
            UploadCommand(args).execute()
        mock_api.upload_file.assert_called_once()
        call_args = mock_api.upload_file.call_args
        assert call_args.args[0] == "owner/repo"
        assert call_args.args[3] == "weights.bin"
        out = capsys.readouterr().out
        assert "Upload complete" in out

    def test_folder_upload(self, parser, mock_api, tmp_path, capsys):
        upload_dir = tmp_path / "output"
        upload_dir.mkdir()
        (upload_dir / "a.txt").write_text("a")
        args = parser.parse_args(["upload", "owner/repo", str(upload_dir)])
        with patch("modelscope_hub.cli.upload.make_api", return_value=mock_api):
            UploadCommand(args).execute()
        mock_api.upload_folder.assert_called_once()
        out = capsys.readouterr().out
        assert "Folder upload complete" in out

    def test_folder_upload_nothing_to_upload(self, parser, mock_api, tmp_path, capsys):
        upload_dir = tmp_path / "empty_dir"
        upload_dir.mkdir()
        (upload_dir / "a.txt").write_text("a")
        mock_api.upload_folder.return_value = None
        args = parser.parse_args(["upload", "owner/repo", str(upload_dir)])
        with patch("modelscope_hub.cli.upload.make_api", return_value=mock_api):
            UploadCommand(args).execute()
        out = capsys.readouterr().out
        assert "nothing to upload" in out

    def test_nonexistent_path_exits_2(self, parser, mock_api, capsys):
        args = parser.parse_args(["upload", "owner/repo", "/nonexistent/path"])
        with patch("modelscope_hub.cli.upload.make_api", return_value=mock_api):
            with pytest.raises(SystemExit) as exc_info:
                UploadCommand(args).execute()
            assert exc_info.value.code == 2

    def test_no_cache_forwarded(self, parser, mock_api, tmp_path, capsys):
        upload_dir = tmp_path / "dir"
        upload_dir.mkdir()
        (upload_dir / "f.txt").write_text("x")
        args = parser.parse_args(["upload", "o/r", str(upload_dir), "--no-cache"])
        with patch("modelscope_hub.cli.upload.make_api", return_value=mock_api):
            UploadCommand(args).execute()
        assert mock_api.upload_folder.call_args.kwargs["use_cache"] is False

    def test_commit_message_forwarded(self, parser, mock_api, tmp_path, capsys):
        test_file = tmp_path / "f.bin"
        test_file.write_text("data")
        args = parser.parse_args([
            "upload", "o/r", str(test_file),
            "--commit-message", "add weights",
            "--commit-description", "desc",
        ])
        with patch("modelscope_hub.cli.upload.make_api", return_value=mock_api):
            UploadCommand(args).execute()
        kw = mock_api.upload_file.call_args.kwargs
        assert kw["commit_message"] == "add weights"
        assert kw["commit_description"] == "desc"

    def test_dataset_upload(self, parser, mock_api, tmp_path, capsys):
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b\n1,2")
        args = parser.parse_args([
            "upload", "o/r", str(test_file), "--repo-type", "dataset",
        ])
        with patch("modelscope_hub.cli.upload.make_api", return_value=mock_api):
            UploadCommand(args).execute()
        assert mock_api.upload_file.call_args.args[1] == "dataset"


# ===================================================================
# Remote integration tests (existing)
# ===================================================================
@pytest.mark.remote
class TestUploadLifecycle:
    """Test file upload with real API on a temporary repo."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_repo(self, api, test_owner, repo_name):
        """Create a model repo for upload testing."""
        cls = type(self)
        cls.repo_id = f"{test_owner}/{repo_name}_upload"
        api.create_repo(cls.repo_id, "model", visibility="private")
        cls.api = api
        yield
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                api.delete_repo(cls.repo_id, "model")
            except Exception:
                pass

    def test_01_upload_file(self, test_token, test_endpoint, tmp_path):
        """Upload a single file successfully."""
        test_file = tmp_path / "test_upload.txt"
        test_file.write_text("hello modelscope upload test")

        exit_code, out, err = run_cli(
            ["upload", self.repo_id, str(test_file), "--repo-type", "model"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [upload file] repo_id={self.repo_id}, file={test_file}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Upload complete" in out

    def test_02_upload_folder(self, test_token, test_endpoint, tmp_path):
        """Upload a directory successfully."""
        upload_dir = tmp_path / "upload_folder"
        upload_dir.mkdir()
        (upload_dir / "file_a.txt").write_text("content a")
        (upload_dir / "file_b.txt").write_text("content b")

        exit_code, out, err = run_cli(
            ["upload", self.repo_id, str(upload_dir), "--repo-type", "model"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [upload folder] repo_id={self.repo_id}, dir={upload_dir}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Folder upload complete" in out
