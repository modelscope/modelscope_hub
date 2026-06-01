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
        try:
            api.delete_repo(cls.repo_id, "model")
            print(f"** Teardown: deleted repo {cls.repo_id}")
        except Exception as e:
            print(f"** Teardown: failed to delete repo {cls.repo_id}: {e}")

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
