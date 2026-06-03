"""Integration tests for dataset-specific operations.

These tests verify that dataset file listing, download, and CLI operations
work correctly against the ModelScope API. Datasets use a different file
listing endpoint (/repo/tree) than models (/repo/files), so they need
dedicated test coverage.

Uses a known small public dataset — no auth required for read operations.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from modelscope_hub import HubApi

PUBLIC_DATASET_ID = "tencent-community/CL-bench"
PUBLIC_DATASET_SMALL_FILE = "dataset_infos.json"
PUBLIC_DATASET_SMALL_FILE_SIZE = 123


@pytest.mark.remote
class TestDatasetFileOperations:
    """Test list_repo_files, download_file, download_repo for datasets."""

    @pytest.fixture(autouse=True)
    def setup_api(self, test_endpoint):
        self.api = HubApi(endpoint=test_endpoint)

    def test_list_repo_files_dataset(self):
        """list_repo_files works for dataset repos (uses /repo/tree endpoint)."""
        files = self.api.list_repo_files(PUBLIC_DATASET_ID, "dataset")
        paths = [f.path for f in files]
        assert len(paths) > 0
        assert PUBLIC_DATASET_SMALL_FILE in paths

    def test_download_file_dataset(self, tmp_path):
        """download_file can fetch a single file from a dataset repo."""
        local = self.api.download_file(
            PUBLIC_DATASET_ID,
            "dataset",
            PUBLIC_DATASET_SMALL_FILE,
            local_dir=str(tmp_path),
            force=True,
        )
        assert local.exists()
        assert local.stat().st_size == PUBLIC_DATASET_SMALL_FILE_SIZE

    def test_download_repo_dataset(self, tmp_path):
        """download_repo can fetch a dataset snapshot (with pattern filter)."""
        output = self.api.download_repo(
            PUBLIC_DATASET_ID,
            "dataset",
            local_dir=str(tmp_path / "out"),
            allow_patterns=["*.json"],
        )
        assert output.is_dir()
        files = [p.name for p in output.rglob("*") if p.is_file()]
        assert PUBLIC_DATASET_SMALL_FILE in files


@pytest.mark.remote
class TestDatasetCLI:
    """Test CLI download commands with --repo-type dataset."""

    def test_cli_download_dataset_single_file(self, test_endpoint, tmp_path):
        """ms download <dataset> <file> --repo-type dataset works."""
        from tests.cli.conftest import run_cli

        exit_code, out, err = run_cli(
            [
                "download", PUBLIC_DATASET_ID,
                PUBLIC_DATASET_SMALL_FILE,
                "--repo-type", "dataset",
                "--local-dir", str(tmp_path),
            ],
            endpoint=test_endpoint,
        )
        assert exit_code == 0, f"CLI failed: {out} {err}"
        assert (tmp_path / PUBLIC_DATASET_SMALL_FILE).exists()

    def test_cli_download_dataset_snapshot(self, test_endpoint, tmp_path):
        """ms download <dataset> --repo-type dataset with pattern filter."""
        from tests.cli.conftest import run_cli

        exit_code, out, err = run_cli(
            [
                "download", PUBLIC_DATASET_ID,
                "--repo-type", "dataset",
                "--local-dir", str(tmp_path),
                "--include", "*.json",
            ],
            endpoint=test_endpoint,
        )
        assert exit_code == 0, f"CLI failed: {out} {err}"
        assert "Snapshot ready" in out
        assert (tmp_path / PUBLIC_DATASET_SMALL_FILE).exists()
