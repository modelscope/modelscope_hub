"""Tests for ``ms download`` command.

Includes:
- Parser tests: all flags, repo types, legacy args
- Execution tests: mock HubApi for file/snapshot download logic
- Remote tests: real API file download (existing)
"""
from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from modelscope_hub.cli.download import DownloadCommand

from pathlib import Path

from .conftest import run_cli


# ===================================================================
# Parser tests
# ===================================================================
class TestDownloadParser:
    """``ms download`` argument parsing."""

    def test_positional_repo_id(self, parser):
        args = parser.parse_args(["download", "owner/repo"])
        assert args.repo_id == "owner/repo"

    def test_positional_with_files(self, parser):
        args = parser.parse_args(["download", "owner/repo", "a.bin", "b.json"])
        assert args.repo_id == "owner/repo"
        assert args.files == ["a.bin", "b.json"]

    def test_repo_type_default_model(self, parser):
        args = parser.parse_args(["download", "owner/repo"])
        assert args.repo_type == "model"

    @pytest.mark.parametrize("repo_type", ["model", "dataset", "studio"])
    def test_repo_type_choices(self, parser, repo_type):
        args = parser.parse_args(["download", "o/r", "--repo-type", repo_type])
        assert args.repo_type == repo_type

    def test_invalid_repo_type_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["download", "o/r", "--repo-type", "skill"])

    def test_revision_flag(self, parser):
        args = parser.parse_args(["download", "o/r", "--revision", "v2"])
        assert args.revision == "v2"

    def test_cache_dir(self, parser):
        args = parser.parse_args(["download", "o/r", "--cache-dir", "/data/cache"])
        assert args.cache_dir == "/data/cache"

    def test_local_dir(self, parser):
        args = parser.parse_args(["download", "o/r", "--local-dir", "./out"])
        assert args.local_dir == "./out"

    def test_max_workers(self, parser):
        args = parser.parse_args(["download", "o/r", "--max-workers", "8"])
        assert args.max_workers == 8

    def test_max_workers_default(self, parser):
        args = parser.parse_args(["download", "o/r"])
        assert args.max_workers == 4

    def test_force_flag(self, parser):
        args = parser.parse_args(["download", "o/r", "--force"])
        assert args.force is True

    def test_force_default_false(self, parser):
        args = parser.parse_args(["download", "o/r"])
        assert args.force is False

    def test_include_single(self, parser):
        args = parser.parse_args(["download", "o/r", "--include", "*.safetensors"])
        assert args.allow_patterns == ["*.safetensors"]

    def test_include_multi(self, parser):
        args = parser.parse_args(["download", "o/r", "--include", "*.bin", "*.json"])
        assert args.allow_patterns == ["*.bin", "*.json"]

    def test_include_repeated(self, parser):
        args = parser.parse_args([
            "download", "o/r", "--include", "*.bin", "--include", "*.json",
        ])
        assert "*.bin" in args.allow_patterns
        assert "*.json" in args.allow_patterns

    def test_exclude_single(self, parser):
        args = parser.parse_args(["download", "o/r", "--exclude", "*.gguf"])
        assert args.ignore_patterns == ["*.gguf"]

    def test_exclude_multi(self, parser):
        args = parser.parse_args(["download", "o/r", "--exclude", "*.bin", "*.gguf"])
        assert args.ignore_patterns == ["*.bin", "*.gguf"]

    def test_include_and_exclude(self, parser):
        args = parser.parse_args([
            "download", "o/r",
            "--include", "*.safetensors",
            "--exclude", "*.bin", "*.gguf",
        ])
        assert args.allow_patterns == ["*.safetensors"]
        assert args.ignore_patterns == ["*.bin", "*.gguf"]

    def test_all_options_combined(self, parser):
        args = parser.parse_args([
            "download", "Qwen/Qwen3-0.6B",
            "--repo-type", "model",
            "--revision", "main",
            "--cache-dir", "/cache",
            "--local-dir", "./out",
            "--max-workers", "16",
            "--include", "*.safetensors",
            "--exclude", "*.bin",
            "--force",
        ])
        assert args.repo_id == "Qwen/Qwen3-0.6B"
        assert args.repo_type == "model"
        assert args.revision == "main"
        assert args.cache_dir == "/cache"
        assert args.local_dir == "./out"
        assert args.max_workers == 16
        assert args.force is True

    def test_repo_type_underscore(self, parser):
        args = parser.parse_args(["download", "o/r", "--repo_type", "dataset"])
        assert args.repo_type == "dataset"


class TestDownloadLegacyParser:
    """Legacy ``--model``/``--dataset``/``--collection`` flags."""

    def test_model_flag(self, parser):
        args = parser.parse_args(["download", "--model", "owner/repo"])
        assert args.model == "owner/repo"

    def test_dataset_flag(self, parser):
        args = parser.parse_args(["download", "--dataset", "owner/data"])
        assert args.dataset == "owner/data"

    def test_collection_flag(self, parser):
        args = parser.parse_args(["download", "--collection", "my-collection"])
        assert args.collection == "my-collection"

    def test_model_dataset_mutually_exclusive(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["download", "--model", "a", "--dataset", "b"])

    def test_legacy_local_dir_underscore(self, parser):
        args = parser.parse_args(["download", "--model", "o/r", "--local_dir", "/tmp"])
        assert args.local_dir_legacy == "/tmp"

    def test_legacy_cache_dir_underscore(self, parser):
        args = parser.parse_args(["download", "--model", "o/r", "--cache_dir", "/cache"])
        assert args.cache_dir_legacy == "/cache"

    def test_subcmd_token_endpoint(self, parser):
        args = parser.parse_args([
            "download", "--model", "o/r",
            "--token", "ms-xxx", "--endpoint", "https://custom.cn",
        ])
        assert args.subcmd_token == "ms-xxx"
        assert args.subcmd_endpoint == "https://custom.cn"


# ===================================================================
# Execution tests — mock HubApi
# ===================================================================
class TestDownloadExecute:
    """DownloadCommand.execute() logic with mocked API."""

    def _patch_download_api(self, mock_api):
        """Patch both ``make_api`` and ``HubApi`` so ``_make_api_with_endpoint``
        resolves through its natural code path without hitting the network."""
        return (
            patch("modelscope_hub.cli.download.make_api", return_value=mock_api),
            patch("modelscope_hub.cli.download.HubApi", return_value=mock_api),
        )

    def test_single_file(self, parser, mock_api, capsys):
        args = parser.parse_args([
            "download", "owner/repo", "config.json", "--cache-dir", "/tmp/cache",
        ])
        p1, p2 = self._patch_download_api(mock_api)
        with p1, p2:
            DownloadCommand(args).execute()
        mock_api.download_file.assert_called_once()
        call_args = mock_api.download_file.call_args
        assert call_args.args[0] == "owner/repo"
        assert call_args.args[2] == "config.json"
        out = capsys.readouterr().out
        assert "config.json" in out

    def test_multiple_files(self, parser, mock_api, capsys):
        args = parser.parse_args([
            "download", "owner/repo", "a.bin", "b.json",
        ])
        p1, p2 = self._patch_download_api(mock_api)
        with p1, p2:
            DownloadCommand(args).execute()
        assert mock_api.download_file.call_count == 2

    def test_snapshot(self, parser, mock_api, capsys):
        args = parser.parse_args(["download", "owner/repo"])
        p1, p2 = self._patch_download_api(mock_api)
        with p1, p2:
            DownloadCommand(args).execute()
        mock_api.download_repo.assert_called_once()
        out = capsys.readouterr().out
        assert "Snapshot ready" in out

    def test_snapshot_with_patterns(self, parser, mock_api, capsys):
        args = parser.parse_args([
            "download", "owner/repo",
            "--include", "*.safetensors",
            "--exclude", "*.bin",
        ])
        p1, p2 = self._patch_download_api(mock_api)
        with p1, p2:
            DownloadCommand(args).execute()
        call_kwargs = mock_api.download_repo.call_args.kwargs
        assert call_kwargs["allow_patterns"] == ["*.safetensors"]
        assert call_kwargs["ignore_patterns"] == ["*.bin"]

    def test_force_forwarded(self, parser, mock_api, capsys):
        args = parser.parse_args(["download", "owner/repo", "f.txt", "--force"])
        p1, p2 = self._patch_download_api(mock_api)
        with p1, p2:
            DownloadCommand(args).execute()
        assert mock_api.download_file.call_args.kwargs["force"] is True

    def test_dataset_repo_type(self, parser, mock_api, capsys):
        args = parser.parse_args([
            "download", "org/data", "--repo-type", "dataset",
        ])
        p1, p2 = self._patch_download_api(mock_api)
        with p1, p2:
            DownloadCommand(args).execute()
        assert mock_api.download_repo.call_args.args[1] == "dataset"

    def test_local_dir_forwarded(self, parser, mock_api, capsys):
        args = parser.parse_args(["download", "o/r", "--local-dir", "./out"])
        p1, p2 = self._patch_download_api(mock_api)
        with p1, p2:
            DownloadCommand(args).execute()
        assert mock_api.download_repo.call_args.kwargs["local_dir"] == Path("./out")

    def test_cache_dir_forwarded(self, parser, mock_api, capsys):
        args = parser.parse_args(["download", "o/r", "--cache-dir", "/cache"])
        p1, p2 = self._patch_download_api(mock_api)
        with p1, p2:
            DownloadCommand(args).execute()
        assert mock_api.download_repo.call_args.kwargs["cache_dir"] == Path("/cache")

    def test_revision_forwarded(self, parser, mock_api, capsys):
        args = parser.parse_args(["download", "o/r", "--revision", "v2"])
        p1, p2 = self._patch_download_api(mock_api)
        with p1, p2:
            DownloadCommand(args).execute()
        assert mock_api.download_repo.call_args.kwargs["revision"] == "v2"

    def test_max_workers_forwarded(self, parser, mock_api, capsys):
        args = parser.parse_args(["download", "o/r", "--max-workers", "16"])
        p1, p2 = self._patch_download_api(mock_api)
        with p1, p2:
            DownloadCommand(args).execute()
        assert mock_api.download_repo.call_args.kwargs["max_workers"] == 16

    def test_studio_repo_type(self, parser, mock_api, capsys):
        args = parser.parse_args(["download", "o/r", "--repo-type", "studio"])
        p1, p2 = self._patch_download_api(mock_api)
        with p1, p2:
            DownloadCommand(args).execute()
        assert mock_api.download_repo.call_args.args[1] == "studio"

    def test_collection_download(self, parser, mock_api, capsys):
        args = parser.parse_args(["download", "--collection", "my-col"])
        mock_api.legacy.get_collection.return_value = {
            "CollectionElements": {
                "CollectionElementVoList": [
                    {"ElementPath": "org", "ElementName": "skill1"},
                ],
            },
        }
        with patch("modelscope_hub.cli.download.make_api", return_value=mock_api):
            DownloadCommand(args).execute()
        mock_api.legacy.get_collection.assert_called_once_with("my-col")
        assert mock_api.download_repo.call_count == 1


# ===================================================================
# Remote integration tests (existing)
# ===================================================================
@pytest.mark.remote
class TestDownloadLifecycle:
    """Test file download with real API.

    Creates a repo, uploads a file, then verifies download works.
    """

    @pytest.fixture(autouse=True, scope="class")
    def setup_repo(self, api, test_owner, test_endpoint, repo_name):
        """Create a model repo and upload a test file for download tests."""
        cls = type(self)
        cls.repo_id = f"{test_owner}/{repo_name}_download"
        print(f"\n** Setup: creating repo {cls.repo_id}")
        print(f"** endpoint: {test_endpoint}")
        try:
            api.create_repo(cls.repo_id, "model", visibility="private")
        except Exception as exc:
            print(f"** Setup FAILED: {exc}")
            if hasattr(exc, "response_body"):
                print(f"** response_body: {exc.response_body}")
            raise

        import base64

        print(f"** Setup: committing test_data.txt to {cls.repo_id}")
        file_bytes = b"download test content"
        content_b64 = base64.b64encode(file_bytes).decode()
        api.legacy.create_commit(
            repo_id=cls.repo_id,
            repo_type="model",
            operations=[{
                "action": "create",
                "path": "test_data.txt",
                "type": "normal",
                "size": len(file_bytes),
                "sha256": "",
                "content": content_b64,
                "encoding": "base64",
            }],
            commit_message="Add test file",
            revision="master",
        )

        cls.api = api
        print("** Setup: done")
        yield
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                api.delete_repo(cls.repo_id, "model")
                print(f"** Teardown: deleted repo {cls.repo_id}")
            except Exception as e:
                print(f"** Teardown: cleanup via web console: {cls.repo_id} ({e})")

    def test_01_download_single_file(self, test_token, test_endpoint, tmp_path):
        """Download a single file by name."""
        exit_code, out, err = run_cli(
            ["download", self.repo_id, "test_data.txt", "--cache-dir", str(tmp_path)],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [download single] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "test_data.txt" in out

    def test_02_download_full_snapshot(self, test_token, test_endpoint, tmp_path):
        """Download full repo snapshot."""
        exit_code, out, err = run_cli(
            ["download", self.repo_id, "--cache-dir", str(tmp_path)],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [download snapshot] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Snapshot ready" in out
