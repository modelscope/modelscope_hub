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
# Pytest hooks
# ---------------------------------------------------------------------------
def pytest_configure(config):
    config.addinivalue_line("markers", "remote: tests requiring remote API access")


def pytest_collection_modifyitems(config, items):
    """Auto-skip remote tests unless explicitly enabled with credentials."""
    run_remote = os.environ.get("MODELSCOPE_RUN_REMOTE_TESTS", "").lower() in ("true", "1", "yes")
    token = os.environ.get("MODELSCOPE_TEST_TOKEN")
    owner = os.environ.get("MODELSCOPE_TEST_OWNER")
    if run_remote and token and owner and token != "your_token_here":
        return
    skip = pytest.mark.skip(
        reason="Remote tests disabled (set MODELSCOPE_RUN_REMOTE_TESTS=true with valid credentials)"
    )
    for item in items:
        if "remote" in item.keywords:
            item.add_marker(skip)


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
