"""Integration test fixtures — real API calls with cleanup."""
from __future__ import annotations

import pytest

from modelscope_hub import HubApi


@pytest.fixture
def api(test_token, test_endpoint):
    """Create a real HubApi instance for integration tests."""
    return HubApi(token=test_token, endpoint=test_endpoint)


@pytest.fixture
def managed_repo(api, test_owner, unique_repo_name):
    """Create a repo and ensure cleanup after test.

    Usage::

        def test_something(managed_repo):
            info, repo_id = managed_repo("model", description="test")
            # ... test the repo ...
            # Cleanup happens automatically
    """
    repo_id = f"{test_owner}/{unique_repo_name}"
    created: list[tuple[str, str]] = []

    def _create(repo_type: str = "model", **kwargs):
        info = api.create_repo(repo_id, repo_type, visibility="private", **kwargs)
        created.append((repo_id, repo_type))
        return info, repo_id

    yield _create

    # Teardown: delete all created repos
    for rid, rtype in created:
        try:
            api.delete_repo(rid, rtype)
        except Exception:
            pass
