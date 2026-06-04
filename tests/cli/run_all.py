#!/usr/bin/env python3
"""Run all CLI unit tests (no network / API credentials required).

Usage:
    python tests/cli/run_all.py          # from project root
    python -m tests.cli.run_all          # as module
    make test-cli                        # via Makefile

This script discovers and runs every ``test_*.py`` under ``tests/cli/``,
excluding tests marked ``@pytest.mark.remote`` which require live API access.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _TESTS_DIR.parent.parent


def main() -> int:
    cmd = [
        sys.executable, "-m", "pytest",
        str(_TESTS_DIR),
        "-k", "not remote",
        "-v",
        "--tb=short",
    ]
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(_PROJECT_ROOT))


if __name__ == "__main__":
    sys.exit(main())
