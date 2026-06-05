#!/usr/bin/env python3
"""Run all integration tests (requires network and API credentials).

Usage:
    python tests/integration/run_all.py              # from project root
    python tests/integration/run_all.py --quick      # skip slow tests (file ops)
    python tests/integration/run_all.py --dry-run    # collect only, don't execute

Environment:
    MODELSCOPE_TEST_TOKEN    — API token for authenticated operations
    MODELSCOPE_TEST_OWNER    — Owner username for repo creation/listing
    MODELSCOPE_TEST_ENDPOINT — API endpoint (default: https://modelscope.cn)

The tests are organized by operation type:
    test_openapi.py         — Raw OpenAPI client surface (models, datasets, MCP, skills)
    test_sdk_api.py         — HubApi facade: repo CRUD, file ops, versioning, cache, compat
    test_remote_repo.py     — Repo lifecycle with cleanup
    test_remote_file_ops.py — File upload/download/delete with cleanup
    test_dataset_ops.py     — Dataset-specific file listing and download
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent.parent


def main() -> int:
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    quick = "--quick" in args

    cmd = [
        sys.executable, "-m", "pytest",
        str(_TESTS_DIR),
        "-v",
        "--tb=short",
        "-m", "remote",
    ]

    if dry_run:
        cmd.append("--collect-only")

    if quick:
        cmd.extend(["--ignore", str(_TESTS_DIR / "test_remote_file_ops.py")])

    for arg in args:
        if arg not in ("--dry-run", "--quick"):
            cmd.append(arg)

    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {_PROJECT_ROOT}")
    print()
    return subprocess.call(cmd, cwd=str(_PROJECT_ROOT))


if __name__ == "__main__":
    sys.exit(main())
