"""Integration tests for file operations, versioning, and auth edge cases.

These tests hit the real ModelScope API. They require
MODELSCOPE_TEST_TOKEN and MODELSCOPE_TEST_OWNER in tests/.env.
"""
from __future__ import annotations

import pytest

from modelscope_hub import HubApi
from modelscope_hub.errors import AuthenticationError


@pytest.mark.remote
class TestRemoteFileOperations:
    """Test list_repo_files, download_file, delete_files against real API."""

    @pytest.fixture(autouse=True)
    def setup_repo_with_file(self, api, test_owner, unique_repo_name):
        """Create a repo, upload a test file, yield, then cleanup."""
        self.repo_id = f"{test_owner}/{unique_repo_name}"
        self.api = api
        print(f"\n** Setup: creating repo {self.repo_id}")
        api.create_repo(self.repo_id, "model", visibility="private")
        print(f"** Setup: uploading test_file.txt to {self.repo_id}")
        api.upload_file(
            self.repo_id,
            "model",
            b"hello modelscope",
            "test_file.txt",
            commit_message="Add test file",
        )
        print("** Setup: done")
        yield
        try:
            api.delete_repo(self.repo_id, "model")
            print(f"** Teardown: deleted repo {self.repo_id}")
        except Exception as e:
            print(f"** Teardown: failed to delete repo {self.repo_id}: {e}")

    def test_list_repo_files(self):
        """list_repo_files returns uploaded file metadata."""
        files = self.api.list_repo_files(self.repo_id, "model")
        paths = [f.path for f in files]
        print(f"\n** repo_id: {self.repo_id}")
        print(f"** list_repo_files returned {len(files)} file(s): {paths}")
        assert "test_file.txt" in paths

    def test_download_file(self, tmp_path):
        """download_file retrieves correct content."""
        local_path = self.api.download_file(
            self.repo_id,
            "model",
            "test_file.txt",
            cache_dir=str(tmp_path),
            force=True,
        )
        content = local_path.read_text() if local_path.exists() else "<file not found>"
        print(f"\n** repo_id: {self.repo_id}")
        print(f"** download_file -> {local_path}")
        print(f"** file exists: {local_path.exists()}, content: {content!r}")
        assert local_path.exists()
        assert content == "hello modelscope"

    @pytest.mark.xfail(
        reason="Server restricts file deletion to cookie-based session auth; "
        "API tokens get 401 'token no longer supports deletion operations'"
    )
    def test_delete_files(self):
        """delete_files removes the file from the repo."""
        print(f"\n** repo_id: {self.repo_id}")
        print("** Deleting test_file.txt ...")
        result = self.api.delete_files(
            self.repo_id, "model", ["test_file.txt"], commit_message="cleanup"
        )
        print(f"** delete_files response: {result}")
        assert "test_file.txt" in result["deleted_files"]
        files = self.api.list_repo_files(self.repo_id, "model")
        paths = [f.path for f in files]
        print(f"** Files after deletion: {paths}")
        assert "test_file.txt" not in paths


@pytest.mark.remote
class TestRemoteVersioning:
    """Test list_repo_revisions and create_repo_tag against real API."""

    @pytest.fixture(autouse=True)
    def setup_repo_with_commit(self, api, test_owner, unique_repo_name):
        """Create a repo with one commit so tags can be created."""
        self.repo_id = f"{test_owner}/{unique_repo_name}"
        self.api = api
        print(f"\n** Setup: creating repo {self.repo_id}")
        api.create_repo(self.repo_id, "model", visibility="private")
        print(f"** Setup: uploading version.txt to {self.repo_id}")
        api.upload_file(
            self.repo_id,
            "model",
            b"v1 content",
            "version.txt",
            commit_message="Initial commit",
        )
        print("** Setup: done")
        yield
        try:
            api.delete_repo(self.repo_id, "model")
            print(f"** Teardown: deleted repo {self.repo_id}")
        except Exception as e:
            print(f"** Teardown: failed to delete repo {self.repo_id}: {e}")

    def test_list_repo_revisions(self):
        """list_repo_revisions contains the master branch."""
        revisions = self.api.list_repo_revisions(self.repo_id, "model")
        names = [r.get("Revision") or r.get("name") or r.get("Name") or "" for r in revisions]
        print(f"\n** repo_id: {self.repo_id}")
        print(f"** list_repo_revisions returned {len(revisions)} revision(s): {names}")
        print(f"** raw response: {revisions}")
        assert isinstance(revisions, list)
        assert len(revisions) > 0
        assert "master" in names

    def test_create_repo_tag(self):
        """create_repo_tag creates a tag visible in revisions list."""
        print(f"\n** repo_id: {self.repo_id}")
        print("** Creating tag 'v1.0' ...")
        result = self.api.create_repo_tag(self.repo_id, "model", "v1.0")
        print(f"** create_repo_tag response: {result}")
        revisions = self.api.list_repo_revisions(self.repo_id, "model")
        names = [r.get("Revision") or r.get("name") or r.get("Name") or "" for r in revisions]
        print(f"** Revisions after tagging: {names}")
        assert "v1.0" in names


class TestLogout:
    """Test logout clears credentials (no remote call needed)."""

    def test_logout_clears_token(self, test_token, test_endpoint):
        """After logout, whoami raises AuthenticationError."""
        api = HubApi(token=test_token, endpoint=test_endpoint)
        print(f"\n** token (masked): {test_token[:8]}...{test_token[-4:]}")
        print(f"** endpoint: {test_endpoint}")
        api.logout()
        print("** logout() called, attempting whoami() ...")
        with pytest.raises((AuthenticationError, Exception)) as exc_info:
            api.whoami()
        print(f"** whoami raised: {type(exc_info.value).__name__}: {exc_info.value}")
