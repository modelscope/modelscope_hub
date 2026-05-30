"""CLI test fixtures for real API integration tests."""
from __future__ import annotations

import io
import os
import sys
import uuid

import pytest

from modelscope_hub.api import HubApi
from modelscope_hub.cli.main import run_cmd


# ---------------------------------------------------------------------------
# Real API fixtures
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
