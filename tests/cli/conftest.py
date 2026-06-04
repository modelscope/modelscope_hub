"""CLI test fixtures — shared across unit and remote integration tests."""
from __future__ import annotations

import io
import sys
import uuid
from unittest.mock import MagicMock

import pytest

from modelscope_hub.api import HubApi
from modelscope_hub.cli.main import _build_parser, run_cmd
from modelscope_hub.types import CacheInfo, CachedRepoInfo, PagedResult, RepoInfo, UserInfo


# ---------------------------------------------------------------------------
# Shared parser fixture (used by ALL parser tests)
# ---------------------------------------------------------------------------
@pytest.fixture
def parser():
    """Build the full CLI parser for argument-parsing tests."""
    return _build_parser()


# ---------------------------------------------------------------------------
# Mock API fixture for unit-level execution tests
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_api():
    """Return a ``MagicMock`` mimicking :class:`HubApi`.

    Common return values are pre-configured so tests don't need to repeat
    boilerplate. Override specific methods in individual tests as needed.
    """
    api = MagicMock(spec=HubApi)
    api.create_repo.return_value = RepoInfo(
        id=1, owner="owner", name="repo", repo_type="model",
    )
    api.get_repo.return_value = RepoInfo(
        id=1, owner="owner", name="repo", repo_type="model",
        visibility=None, license="apache-2.0", downloads=100, likes=5,
    )
    api.list_repos.return_value = PagedResult(
        items=[
            RepoInfo(id=1, owner="owner", name="model1", repo_type="model",
                     visibility=None, downloads=100, likes=5),
        ],
        total_count=1, page_number=1, page_size=10,
    )
    api.download_file.return_value = "/cache/owner/repo/file.txt"
    api.download_repo.return_value = "/cache/owner/repo"
    api.upload_folder.return_value = "commit_sha"
    api.whoami.return_value = UserInfo(
        username="testuser", email="test@example.com", id=42, description="",
    )
    api.login.return_value = UserInfo(
        username="testuser", email="test@example.com", id=42,
    )
    api.scan_cache.return_value = CacheInfo(
        cache_dir="/tmp/cache", total_size=1024,
        repos=[CachedRepoInfo(
            repo_id="owner/repo", repo_type="model", revision="master",
            nb_files=3, size_on_disk=1024, local_path="/tmp/cache/owner/repo",
        )],
    )
    api.clear_cache.return_value = 2048
    api.list_secrets.return_value = [
        {"key": "API_KEY", "description": "test", "updated_at": "2000-01-01T00:00:00Z"},
    ]
    api.list_mcp_servers.return_value = PagedResult(
        items=[{"id": "mcp-1", "name": "weather", "status": "running", "description": "Weather MCP"}],
        total_count=1, page_number=1, page_size=20,
    )
    api.get_mcp_server.return_value = {"id": "mcp-1", "name": "weather"}
    api.get_repo_logs.return_value = {"logs": ["line1", "line2"]}
    api.resolve_endpoint_for_read.return_value = "https://modelscope.cn"
    return api


# ---------------------------------------------------------------------------
# Real API fixtures (used by @pytest.mark.remote tests)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="class")
def api(test_token, test_endpoint) -> HubApi:
    """Create a real HubApi instance for class-scoped tests."""
    return HubApi(token=test_token, endpoint=test_endpoint)


@pytest.fixture(scope="class")
def repo_name() -> str:
    """Generate a unique repo name for the test class."""
    return f"unittest_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# CLI runner helper (module-level function, usable by all tests)
# ---------------------------------------------------------------------------
def run_cli(
    args: list[str],
    token: str | None = None,
    endpoint: str | None = None,
    input_text: str | None = None,
) -> tuple[int, str, str]:
    """Execute a CLI command with real credentials and capture output.

    Token and endpoint are injected via the global ``--token`` / ``--endpoint``
    flags that :func:`make_api` reads from the parsed namespace.
    """
    full_args: list[str] = []
    if token:
        full_args += ["--token", token]
    if endpoint:
        full_args += ["--endpoint", endpoint]
    full_args += args

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
        exit_code = run_cmd(full_args)
    except SystemExit as e:
        exit_code = int(e.code) if isinstance(e.code, int) else (0 if e.code is None else 1)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.stdin = old_stdin
    return exit_code, captured_out.getvalue(), captured_err.getvalue()
