"""Global test configuration and fixtures."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# .env loader (simple parser, no third-party dependencies)
# ---------------------------------------------------------------------------
def _load_dotenv(path: Path) -> None:
    """Load .env file into os.environ (simple parser, no third-party deps)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(Path(__file__).parent / ".env")


# ---------------------------------------------------------------------------
# Remote mode detection
# ---------------------------------------------------------------------------
def is_remote_enabled() -> bool:
    """Check if remote tests should run.

    Returns True (run real API tests) unless MODELSCOPE_RUN_REMOTE_TESTS=false.
    When the flag is unset, we auto-detect based on valid credentials.
    """
    flag = os.environ.get("MODELSCOPE_RUN_REMOTE_TESTS", "").lower()
    if flag == "false":
        return False
    # Explicit opt-in
    if flag in ("true", "1", "yes"):
        token = os.environ.get("MODELSCOPE_TEST_TOKEN", "")
        return bool(token and token != "your_token_here")
    # Auto-detect: has valid credentials → remote enabled
    token = os.environ.get("MODELSCOPE_TEST_TOKEN", "")
    return bool(token and token != "your_token_here")


# ---------------------------------------------------------------------------
# Pytest hooks
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line("markers", "remote: tests requiring remote API access")
    config.addinivalue_line("markers", "mock_only: tests using mock API (only run when MODELSCOPE_RUN_REMOTE_TESTS=false)")


def pytest_collection_modifyitems(config, items):
    """Conditionally skip tests based on remote mode.

    - remote_enabled=True  → skip mock_only tests, run remote tests
    - remote_enabled=False → skip remote tests, run mock_only tests
    """
    remote_enabled = is_remote_enabled()

    if remote_enabled:
        # Skip mock-only tests when real API is available
        skip_mock = pytest.mark.skip(
            reason="Mock-only tests skipped (remote mode active)"
        )
        for item in items:
            if "mock_only" in item.keywords:
                item.add_marker(skip_mock)
    else:
        # Skip remote tests when in mock/local mode
        skip_remote = pytest.mark.skip(
            reason="Remote tests disabled (set MODELSCOPE_RUN_REMOTE_TESTS=true with valid credentials)"
        )
        for item in items:
            if "remote" in item.keywords:
                item.add_marker(skip_remote)


# ---------------------------------------------------------------------------
# Global fixtures (session-scoped for integration/remote tests)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def test_token() -> str:
    return os.environ.get("MODELSCOPE_TEST_TOKEN", "fake_token_for_unit_tests")


@pytest.fixture(scope="session")
def test_owner() -> str:
    return os.environ.get("MODELSCOPE_TEST_OWNER", "test_owner")


@pytest.fixture(scope="session")
def test_endpoint() -> str:
    return os.environ.get("MODELSCOPE_TEST_ENDPOINT", "https://modelscope.cn")


@pytest.fixture
def unique_repo_name() -> str:
    """Generate a unique repo name with unittest_ prefix."""
    return f"unittest_{uuid.uuid4().hex[:8]}"
