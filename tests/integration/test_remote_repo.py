"""Integration tests for remote repo operations.

These tests create real repos on ModelScope and clean up after themselves.
Requires MODELSCOPE_TEST_TOKEN and MODELSCOPE_TEST_OWNER in .env.
"""
from __future__ import annotations

import pytest


@pytest.mark.remote
class TestRemoteRepoLifecycle:
    """Test full repo lifecycle: create → info → exists → delete."""

    def test_create_and_delete_model_repo(self, api, test_owner, unique_repo_name):
        """Create a private model repo, verify it exists, then delete it."""
        repo_id = f"{test_owner}/{unique_repo_name}"
        try:
            # Create
            info = api.create_repo(repo_id, "model", visibility="private")
            assert info is not None
            assert info.name == unique_repo_name or info.repo_id == repo_id

            # Verify exists
            assert api.repo_exists(repo_id, "model")

            # Get info
            repo_info = api.get_repo(repo_id, "model")
            assert repo_info.owner == test_owner
        finally:
            # Cleanup
            try:
                api.delete_repo(repo_id, "model")
            except Exception:
                pass

        # Verify deleted
        assert not api.repo_exists(repo_id, "model")

    def test_create_and_delete_dataset_repo(self, api, test_owner, unique_repo_name):
        """Create a private dataset repo, verify it exists, then delete it."""
        repo_id = f"{test_owner}/{unique_repo_name}"
        try:
            # Create
            info = api.create_repo(repo_id, "dataset", visibility="private")
            assert info is not None

            # Verify exists
            assert api.repo_exists(repo_id, "dataset")
        finally:
            # Cleanup
            try:
                api.delete_repo(repo_id, "dataset")
            except Exception:
                pass

        # Verify deleted
        assert not api.repo_exists(repo_id, "dataset")


@pytest.mark.remote
class TestRemoteRepoList:
    """Test listing repos."""

    def test_list_models(self, api, test_owner):
        """List models for a user returns a PagedResult."""
        result = api.list_repos("model", owner=test_owner, page_size=5)
        # Just verify the structure — may be empty for a fresh account
        assert hasattr(result, "items")
        assert hasattr(result, "total_count")
        assert isinstance(result.items, list)


@pytest.mark.remote
class TestRemoteRepoWithManagedFixture:
    """Test using the managed_repo fixture for automatic cleanup."""

    def test_managed_repo_create(self, managed_repo):
        """managed_repo fixture creates and auto-cleans repos."""
        info, repo_id = managed_repo("model")
        assert info is not None
        # The fixture will delete this repo in teardown
