"""Integration tests for remote repo operations.

These tests create real repos on ModelScope and clean up after themselves.
Requires MODELSCOPE_TEST_TOKEN and MODELSCOPE_TEST_OWNER in tests/.env.
"""
from __future__ import annotations

import pytest


@pytest.mark.remote
class TestRemoteRepoLifecycle:
    """Test full repo lifecycle: create → info → exists → delete."""

    def test_create_and_delete_model_repo(self, api, test_owner, unique_repo_name):
        """Create a private model repo, verify it exists, then delete it."""
        repo_id = f"{test_owner}/{unique_repo_name}"
        print(f"\n** repo_id: {repo_id}")
        try:
            # Create
            info = api.create_repo(repo_id, "model", visibility="private")
            print(f"** create_repo response: {info}")
            assert info is not None
            assert info.name == unique_repo_name or info.repo_id == repo_id

            # Verify exists
            exists = api.repo_exists(repo_id, "model")
            print(f"** repo_exists: {exists}")
            assert exists

            # Get info
            repo_info = api.get_repo(repo_id, "model")
            print(f"** get_repo: owner={repo_info.owner}, name={repo_info.name}")
            assert repo_info.owner == test_owner
        finally:
            # Cleanup
            try:
                api.delete_repo(repo_id, "model")
                print(f"** delete_repo: success")
            except Exception as e:
                print(f"** delete_repo: failed - {e}")

        # Verify deleted
        exists_after = api.repo_exists(repo_id, "model")
        print(f"** repo_exists after delete: {exists_after}")
        assert not exists_after

    def test_create_and_delete_dataset_repo(self, api, test_owner, unique_repo_name):
        """Create a private dataset repo, verify it exists, then delete it."""
        repo_id = f"{test_owner}/{unique_repo_name}"
        print(f"\n** repo_id: {repo_id}")
        try:
            # Create
            info = api.create_repo(repo_id, "dataset", visibility="private")
            print(f"** create_repo (dataset) response: {info}")
            assert info is not None

            # Verify exists
            exists = api.repo_exists(repo_id, "dataset")
            print(f"** repo_exists: {exists}")
            assert exists
        finally:
            # Cleanup
            try:
                api.delete_repo(repo_id, "dataset")
                print(f"** delete_repo: success")
            except Exception as e:
                print(f"** delete_repo: failed - {e}")

        # Verify deleted
        exists_after = api.repo_exists(repo_id, "dataset")
        print(f"** repo_exists after delete: {exists_after}")
        assert not exists_after


@pytest.mark.remote
class TestRemoteRepoList:
    """Test listing repos."""

    def test_list_models(self, api, test_owner):
        """List models for a user returns a PagedResult."""
        result = api.list_repos("model", owner=test_owner, page_size=5)
        print(f"\n** list_repos(model, owner={test_owner})")
        print(f"** total_count: {result.total_count}, items: {len(result.items)}")
        print(f"** first items: {[getattr(i, 'repo_id', i) for i in result.items[:3]]}")
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
        print(f"\n** managed_repo created: {repo_id}")
        print(f"** info: {info}")
        assert info is not None
        # The fixture will delete this repo in teardown
