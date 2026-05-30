"""Tests for ``ms upload`` command — real API file and folder upload."""
from __future__ import annotations

import pytest

from .conftest import run_cli


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
