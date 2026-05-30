"""Entry point for the ``modelscope`` / ``ms`` console scripts.

Subcommands live in dedicated modules and are wired in via their
:meth:`CLICommand.register` static method. :func:`run_cmd` is intentionally
small: it builds the argparse tree, dispatches to the chosen subcommand,
and translates SDK exceptions into friendly, machine-parseable output.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Sequence

from .. import __version__
from ..errors import HubError, NetworkError
from .base import error
from .cache import CacheCommand
from .deploy import DeployCommand, LogsCommand, SettingsCommand, StopCommand
from .download import DownloadCommand
from .login import LoginCommand, WhoamiCommand
from .mcp import McpCommand
from .repo import RepoCommand
from .secret import SecretCommand
from .upload import UploadCommand

# All top-level commands in registration order. Adding a new command means
# importing it above and appending it here — that's it.
_COMMANDS = [
    LoginCommand,
    WhoamiCommand,
    RepoCommand,
    DownloadCommand,
    UploadCommand,
    DeployCommand,
    StopCommand,
    LogsCommand,
    SettingsCommand,
    SecretCommand,
    McpCommand,
    CacheCommand,
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ms",
        description="ModelScope Hub command-line interface.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"modelscope_hub {__version__}",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="API token (overrides MODELSCOPE_API_TOKEN and the persisted token).",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="API endpoint (overrides MODELSCOPE_ENDPOINT).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True
    for cmd in _COMMANDS:
        cmd.register(subparsers)

    return parser


def run_cmd(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point referenced by ``[project.scripts]``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    command_cls = getattr(args, "_command", None)
    if command_cls is None:
        parser.print_help(sys.stderr)
        return 2

    try:
        command_cls(args).execute()
    except KeyboardInterrupt:
        error("Interrupted.")
        return 130
    except SystemExit as exc:  # honour explicit SystemExit from subcommands
        return int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except NetworkError as exc:
        error(f"Network error: {exc}")
        return 1
    except HubError as exc:
        error(str(exc))
        return 1
    except ValueError as exc:
        error(str(exc))
        return 2
    except NotImplementedError as exc:
        error(str(exc))
        return 2
    except Exception as exc:  # pragma: no cover - unexpected
        error(f"Unexpected error: {exc.__class__.__name__}: {exc}")
        if getattr(args, "verbose", False):
            raise
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_cmd())
