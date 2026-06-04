"""Base contracts and shared helpers for ``modelscope`` CLI subcommands.

Every concrete subcommand is a small, self-contained class that:

1. Registers its argparse parser via :meth:`CLICommand.register`.
2. Receives the parsed ``argparse.Namespace`` in its constructor.
3. Performs its work in :meth:`CLICommand.execute`.

This keeps :mod:`.main` free of subcommand-specific logic and makes adding
a new command a single-file change.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from argparse import Action, ArgumentParser, Namespace
from typing import Any, Iterable, Sequence

from ..api import HubApi
from ..constants import RepoType


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------
class CLICommand(ABC):
    """Abstract contract every CLI subcommand implements."""

    def __init__(self, args: Namespace) -> None:
        self.args = args

    @staticmethod
    @abstractmethod
    def register(subparsers: Action) -> None:
        """Attach this command's argparse parser to ``subparsers``."""

    @abstractmethod
    def execute(self) -> None:
        """Run the command. Raise on failure; print on success."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def make_api(args: Namespace) -> HubApi:
    """Construct a :class:`HubApi` honouring global and subcommand ``--token`` / ``--endpoint``.

    Automatically merges subcommand-level ``subcmd_token``/``subcmd_endpoint``
    into the namespace (subcommand values take precedence over global values).
    """
    subcmd_token = getattr(args, "subcmd_token", None)
    if subcmd_token:
        args.token = subcmd_token
    subcmd_endpoint = getattr(args, "subcmd_endpoint", None)
    if subcmd_endpoint:
        args.endpoint = subcmd_endpoint

    return HubApi(
        token=getattr(args, "token", None),
        endpoint=getattr(args, "endpoint", None),
    )


def add_repo_type_arg(
    parser: ArgumentParser,
    *,
    choices: Sequence[str] | None = None,
    default: str | None = None,
    required: bool = True,
    help: str = "Repository type.",
) -> None:
    """Attach a uniform ``--repo-type`` argument to ``parser``.

    Also accepts the legacy ``--repo_type`` (underscore) form for backward
    compatibility with the old ``modelscope`` CLI.
    """
    valid = list(choices) if choices else [t.value for t in RepoType]
    parser.add_argument(
        "--repo-type", "--repo_type",
        dest="repo_type",
        choices=valid,
        default=default,
        required=required and default is None,
        help=help,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def info(message: str) -> None:
    """Print a neutral status line."""
    print(message)


def success(message: str) -> None:
    """Print a success line (prefixed with ✓ when stdout is a TTY)."""
    prefix = "✓ " if sys.stdout.isatty() else ""
    print(f"{prefix}{message}")


def warn(message: str) -> None:
    """Print a warning line to stderr."""
    print(f"warning: {message}", file=sys.stderr)


def error(message: str) -> None:
    """Print an error line to stderr."""
    print(f"error: {message}", file=sys.stderr)


def render_table(rows: Iterable[Sequence[Any]], headers: Sequence[str]) -> str:
    """Format ``rows`` as a fixed-width text table.

    Pure stdlib — no third-party dependencies. Truncates very long cells so a
    runaway field cannot destroy the layout.
    """
    str_rows: list[list[str]] = [
        [_truncate(str(c) if c is not None else "-") for c in row] for row in rows
    ]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(cell))

    sep = "  "
    out = [sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    out.append(sep.join("-" * w for w in widths))
    for row in str_rows:
        out.append(sep.join(
            (row[i] if i < len(row) else "").ljust(widths[i]) for i in range(len(headers))
        ))
    return "\n".join(out)


def _truncate(text: str, limit: int = 80) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def parse_kv_pairs(values: Iterable[str]) -> dict[str, str]:
    """Parse ``key=value`` argument tokens into a dict.

    Raises :class:`ValueError` when a token does not contain ``=``.
    """
    result: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(
                f"Invalid setting {raw!r}: expected 'key=value' format."
            )
        key, _, value = raw.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid setting {raw!r}: empty key.")
        result[key] = value
    return result


__all__ = [
    "CLICommand",
    "add_repo_type_arg",
    "error",
    "info",
    "make_api",
    "parse_kv_pairs",
    "render_table",
    "success",
    "warn",
]
