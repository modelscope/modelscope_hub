"""Tests for ``ms download`` command — real API file download."""
from __future__ import annotations

import pytest

from .conftest import run_cli


@pytest.mark.remote
class TestDownloadLifecycle:
    """Test file download with real API.

    Creates a repo, uploads a file, then verifies download works.
    """

    @pytest.fixture(autouse=True, scope="class")
    def setup_repo(self, api, test_owner, repo_name):
        """Create a model repo and upload a test file for download tests."""
        cls = type(self)
        cls.repo_id = f"{test_owner}/{repo_name}_download"
        api.create_repo(cls.repo_id, "model", visibility="private")
        # Upload a small test file for download verification
        import tempfile
        import os

        fd, path = tempfile.mkstemp(suffix=".txt")
        try:
            os.write(fd, b"download test content")
            os.close(fd)
            api.upload_file(cls.repo_id, "model", path, "test_data.txt")
        finally:
            os.unlink(path)
        cls.api = api
        yield
        try:
            api.delete_repo(cls.repo_id, "model")
        except Exception:
            pass

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
