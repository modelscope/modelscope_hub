"""CLI test fixtures — mock HubApi and CLI runner."""
from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modelscope_hub.types import (
    CachedRepoInfo,
    CacheInfo,
    PagedResult,
    RepoInfo,
    UserInfo,
)


@pytest.fixture
def mock_api():
    """Create a mock HubApi instance with common return values pre-configured."""
    api = MagicMock()
    api.whoami.return_value = UserInfo(
        id="123", username="test_user", email="test@example.com"
    )
    api.login.return_value = UserInfo(
        id="123", username="test_user", email="test@example.com"
    )
    api.create_repo.return_value = RepoInfo(
        id="456", owner="test_owner", name="test_repo", repo_type="model"
    )
    api.get_repo.return_value = RepoInfo(
        id="456",
        owner="test_owner",
        name="test_repo",
        repo_type="model",
        visibility=1,
        license="Apache-2.0",
        downloads=100,
        likes=10,
    )
    api.list_repos.return_value = PagedResult(
        items=[
            RepoInfo(owner="test_owner", name="repo1", visibility=1, downloads=50, likes=5, license="MIT"),
            RepoInfo(owner="test_owner", name="repo2", visibility=3, downloads=10, likes=2, license="Apache-2.0"),
        ],
        total_count=2,
        page_number=1,
        page_size=10,
    )
    api.delete_repo.return_value = None
    api.deploy_repo.return_value = {"status": "deploying"}
    api.stop_repo.return_value = {"status": "stopping"}
    api.get_repo_logs.return_value = {"logs": ["line1", "line2", "line3"]}
    api.update_repo_settings.return_value = {"ok": True}
    api.download_file.return_value = Path("/tmp/cached/file.bin")
    api.download_repo.return_value = Path("/tmp/cached/repo_snapshot")
    api.upload_file.return_value = {"sha": "abc123"}
    api.upload_folder.return_value = {"sha": "def456"}
    api.list_secrets.return_value = [
        {"key": "MY_KEY", "description": "test secret", "updated_at": "2025-01-01"},
    ]
    api.add_secret.return_value = {"ok": True}
    api.update_secret.return_value = {"ok": True}
    api.delete_secret.return_value = {"ok": True}
    api.list_mcp_servers.return_value = PagedResult(
        items=[
            {"id": "srv1", "name": "mcp-server-1", "status": "running", "description": "desc1"},
        ],
        total_count=1,
        page_number=1,
        page_size=20,
    )
    api.get_mcp_server.return_value = {"id": "srv1", "name": "mcp-server-1", "status": "running"}
    api.deploy_mcp_server.return_value = {"status": "deploying"}
    api.undeploy_mcp_server.return_value = {"status": "stopped"}
    api.scan_cache.return_value = CacheInfo(
        repos=[
            CachedRepoInfo(
                repo_id="owner/model1",
                repo_type="model",
                revision="master",
                size_on_disk=1024 * 1024 * 50,
                nb_files=10,
                local_path="/tmp/cache/models/owner/model1",
            ),
        ],
        total_size=1024 * 1024 * 50,
        cache_dir="/tmp/cache",
    )
    api.clear_cache.return_value = 1024 * 1024 * 50
    return api


# All CLI modules that import make_api from .base
_MAKE_API_TARGETS = [
    "modelscope_hub.cli.base.make_api",
    "modelscope_hub.cli.login.make_api",
    "modelscope_hub.cli.repo.make_api",
    "modelscope_hub.cli.download.make_api",
    "modelscope_hub.cli.upload.make_api",
    "modelscope_hub.cli.deploy.make_api",
    "modelscope_hub.cli.secret.make_api",
    "modelscope_hub.cli.mcp.make_api",
    "modelscope_hub.cli.cache.make_api",
]


@pytest.fixture
def patch_make_api(mock_api):
    """Patch make_api in all CLI modules to return mock_api."""
    from contextlib import ExitStack

    with ExitStack() as stack:
        for target in _MAKE_API_TARGETS:
            stack.enter_context(patch(target, return_value=mock_api))
        yield mock_api


@pytest.fixture
def run_cli(patch_make_api):
    """Run CLI commands and capture output.

    Uses ``run_cmd(argv)`` to avoid manipulating sys.argv. Captures
    stdout/stderr through temporary StringIO objects.
    """
    def _run(args: list[str], input_text: str | None = None):
        from modelscope_hub.cli.main import run_cmd

        captured_out = io.StringIO()
        captured_err = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_stdin = sys.stdin
        try:
            sys.stdout = captured_out
            sys.stderr = captured_err
            if input_text is not None:
                sys.stdin = io.StringIO(input_text)
            exit_code = run_cmd(args)
        except SystemExit as e:
            exit_code = int(e.code) if e.code else 0
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sys.stdin = old_stdin
        return exit_code, captured_out.getvalue(), captured_err.getvalue()

    return _run
