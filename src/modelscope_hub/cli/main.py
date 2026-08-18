"""Entry point for every ModelScope console script.

This package owns all four aliases -- ``modelscope``, ``ms``,
``modelscope-hub`` and ``ms-hub`` -- so exactly one distribution writes those
files and no cross-package overwrite can leave a user without a working CLI.
The umbrella ``modelscope`` SDK contributes its commands as plugins instead of
shipping a competing script.

Because a single parser serves all four aliases, the program name,
``--version`` output and help epilog are derived from ``sys.argv[0]`` rather
than hard-coded, so each alias reports itself accurately. Only the two
``*-hub`` aliases are treated as hub-only; anything else (including
``python -m``) gets the brand view, which always names the hub version too and
so cannot misreport this package.

Subcommands live in dedicated modules and are wired in via their
:meth:`CLICommand.register` static method. :func:`run_cmd` is intentionally
small: it builds the argparse tree, dispatches to the chosen subcommand,
and translates SDK exceptions into friendly, machine-parseable output.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import sys
from argparse import SUPPRESS, Action
from collections.abc import Sequence
from functools import cache
from pathlib import Path

from .. import __version__
from ..constants import MODELSCOPE_ASCII
from ..errors import HubError, InvalidParameter, NotSupportedError
from .agent import AgentCommand
from .base import CLICommand, error, info
from .cache import CacheCommand, _CacheClear, _CacheScan
from .deploy import DeployCommand, LogsCommand, SettingsCommand, StopCommand
from .download import DownloadCommand
from .login import LoginCommand, WhoamiCommand
from .mcp import McpCommand
from .repo import CreateCommand, DeleteCommand, InfoCommand, ListCommand, RepoCommand
from .secret import SecretCommand
from .upload import UploadCommand

# All top-level commands in registration order. Adding a new command means
# importing it above and appending it here — that's it.
_COMMANDS = [
    LoginCommand,
    WhoamiCommand,
    CreateCommand,
    InfoCommand,
    ListCommand,
    DeleteCommand,
    DownloadCommand,
    UploadCommand,
    DeployCommand,
    StopCommand,
    LogsCommand,
    SettingsCommand,
    SecretCommand,
    McpCommand,
    CacheCommand,
    AgentCommand,
]

# Plugin entry-point group name
_PLUGIN_GROUP = "modelscope_hub.cli_plugins"


# ---------------------------------------------------------------------------
# Invocation identity
# ---------------------------------------------------------------------------
# Aliases that address this package alone. The other two (``modelscope`` /
# ``ms``) front the whole brand, so they lead with the umbrella SDK version
# whenever it is installed alongside.
_HUB_ONLY_PROGS = frozenset({"modelscope-hub", "ms-hub"})

# Distribution name of the umbrella SDK that contributes the plugin commands.
_SDK_DIST = "modelscope"


def _invoked_as() -> str:
    """Basename of the console script that started this process."""
    return Path(sys.argv[0]).name


@cache
def _sdk_version() -> str | None:
    """Version of the umbrella ``modelscope`` SDK, or ``None`` when absent.

    Cached because resolving it walks ``sys.path`` looking for installed
    distribution metadata, and both the version line and the help epilog ask
    the same question.
    """
    try:
        return importlib.metadata.version(_SDK_DIST)
    except importlib.metadata.PackageNotFoundError:
        return None


def _version_text() -> str:
    """Build the ``--version`` line for the alias actually invoked."""
    hub = f"modelscope-hub {__version__}"
    if _invoked_as() in _HUB_ONLY_PROGS:
        return hub
    sdk = _sdk_version()
    if sdk is None:
        return f"{hub} (full ModelScope SDK not installed)"
    return f"modelscope {sdk} ({hub})"


def _brand_epilog() -> str | None:
    """Tell brand-alias users why SDK subcommands are missing, if they are.

    ``modelscope`` / ``ms`` ship with this package, so they exist even on a
    lightweight hub-only install -- in which case the SDK subcommands are
    genuinely absent and the user deserves to know how to get them.
    """
    if _invoked_as() in _HUB_ONLY_PROGS or _sdk_version() is not None:
        return None
    return (
        "Commands from the full ModelScope SDK (pipeline, server, studio,\n"
        "modelcard, ...) are not installed. Add them with:\n"
        "    pip install modelscope"
    )


class _VersionAction(argparse.Action):
    """``--version`` resolved on use rather than at parser-build time.

    The stock ``version`` action needs its string up front, which would make
    every invocation pay for a distribution-metadata lookup just to run an
    unrelated subcommand.
    """

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str = SUPPRESS,
        default: str = SUPPRESS,
        help: str | None = None,
    ) -> None:
        super().__init__(
            option_strings=list(option_strings),
            dest=dest,
            default=default,
            nargs=0,
            help=help,
        )

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: str | None = None,
    ) -> None:
        print(_version_text())
        parser.exit()


def _build_parser() -> argparse.ArgumentParser:
    # ``prog`` is intentionally left unset so argparse derives it from
    # ``sys.argv[0]``: one parser backs all four console scripts, so usage and
    # help output must reflect whichever command the user actually ran.
    parser = argparse.ArgumentParser(
        description="ModelScope Hub command-line interface.",
        epilog=_brand_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-V",
        "--version",
        action=_VersionAction,
        help="Show the version of the invoked command and exit.",
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
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging and print the full error cause chain.",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    for cmd in _COMMANDS:
        cmd.register(subparsers)

    # Register top-level aliases for backward compatibility
    _register_aliases(subparsers)

    # Discover and register CLI plugins from other packages
    _discover_plugins(subparsers)

    return parser


# ---------------------------------------------------------------------------
# Aliases — backward compat with old `modelscope create`, `scan-cache`, etc.
# ---------------------------------------------------------------------------
def _register_aliases(subparsers) -> None:
    """Register top-level command aliases for legacy CLI compatibility."""
    RepoCommand.register(subparsers)
    _register_scan_cache_alias(subparsers)
    _register_clear_cache_alias(subparsers)


def _register_scan_cache_alias(subparsers) -> None:
    """``ms-hub scan-cache`` → alias for ``ms-hub cache scan``."""
    p = subparsers.add_parser("scan-cache", help="[Alias] Show cached repos and disk usage.")
    p.add_argument("--dir", "--cache-dir", dest="cache_dir", default=None)
    p.set_defaults(_command=_ScanCacheAlias)


def _register_clear_cache_alias(subparsers) -> None:
    """``ms-hub clear-cache`` → alias for ``ms-hub cache clear``."""

    p = subparsers.add_parser("clear-cache", help="[Alias] Remove cached files.")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--model", type=str, default=None, help=SUPPRESS)
    group.add_argument("--dataset", type=str, default=None, help=SUPPRESS)
    p.add_argument("--cache-dir", dest="cache_dir", default=None, help="Override cache directory.")
    p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation.")
    p.set_defaults(_command=_ClearCacheAlias)


class _ScanCacheAlias(CLICommand):
    """Adapter: top-level ``scan-cache`` → ``cache scan``."""

    @staticmethod
    def register(subparsers) -> None:
        pass

    def execute(self) -> None:
        _CacheScan(self.args).execute()


class _ClearCacheAlias(CLICommand):
    """Adapter: top-level ``clear-cache`` → ``cache clear``.

    Maps legacy --model/--dataset to repo_type + repo_id.
    """

    @staticmethod
    def register(subparsers) -> None:
        pass

    def execute(self) -> None:
        model = getattr(self.args, "model", None)
        dataset = getattr(self.args, "dataset", None)
        if model:
            self.args.repo_type = "model"
            self.args.repo_id = model
        elif dataset:
            self.args.repo_type = "dataset"
            self.args.repo_id = dataset
        else:
            self.args.repo_type = None
            self.args.repo_id = None
        _CacheClear(self.args).execute()


# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------
def _discover_plugins(subparsers: Action) -> None:
    """Discover CLI plugins registered via entry_points.

    A plugin must never be able to break the whole CLI, so every failure is
    contained. Failure severity differs though: a plugin that cannot even be
    imported is usually an optional extra the user simply did not install
    (``server`` needs the HTTP stack, for instance), which would make a warning
    on every invocation pure noise -- whereas a name collision is always a
    packaging mistake and stays silent forever unless we say so.
    """
    # ``entry_points(group=...)`` is available on all supported Pythons (3.10+).
    eps = importlib.metadata.entry_points(group=_PLUGIN_GROUP)
    log = logging.getLogger(__name__)
    # Live reference: it also grows as plugins register, so plugin-vs-plugin
    # collisions are caught too, not just plugin-vs-built-in.
    registered = getattr(subparsers, "choices", None)

    for ep in eps:
        try:
            cmd_cls = ep.load()
            name = getattr(cmd_cls, "name", ep.name)
            if registered is not None and name in registered:
                log.warning(
                    "Skipping CLI plugin %r: command %r is already registered.",
                    ep.name,
                    name,
                )
                continue
            if hasattr(cmd_cls, "register"):
                cmd_cls.register(subparsers)
            elif hasattr(cmd_cls, "define_args"):
                cmd_cls.define_args(subparsers)
        except Exception as exc:
            log.debug("Failed to load CLI plugin %r: %s", ep.name, exc)


# ---------------------------------------------------------------------------
# Error reporting
# ---------------------------------------------------------------------------
def _next_cause(exc: BaseException) -> BaseException | None:
    """Return what *exc* was raised from, honouring ``raise ... from None``."""
    if exc.__cause__ is not None:
        return exc.__cause__
    if exc.__suppress_context__:
        return None
    return exc.__context__


def _report_hub_error(exc: HubError, *, verbose: bool, max_depth: int = 5) -> None:
    """Print a structured report for an SDK error.

    ``str(exc)`` already carries the error code, HTTP status, request id and --
    for API errors -- the request/response detail. Verbose mode additionally
    unwinds the cause chain: wrapping an exception is convenient for callers but
    otherwise hides the originating failure from whoever has to diagnose it.

    The walk is bounded by *max_depth* and skips exceptions already visited, so
    a self-referential chain cannot stall the error path.
    """
    error(str(exc))
    if exc.suggestion and exc.error_code != "E9001":
        info(f"Suggestion: {exc.suggestion}")
    if not verbose:
        return

    seen = {id(exc)}
    cause = _next_cause(exc)
    depth = 1
    while cause is not None and id(cause) not in seen and depth <= max_depth:
        info(f"{'  ' * depth}Caused by: {cause.__class__.__name__}: {cause}")
        seen.add(id(cause))
        cause = _next_cause(cause)
        depth += 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_cmd(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point referenced by ``[project.scripts]``."""
    print(MODELSCOPE_ASCII, file=sys.stderr)
    parser = _build_parser()
    args = parser.parse_args(argv)

    verbose = bool(getattr(args, "verbose", False))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
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
    except (InvalidParameter, NotSupportedError) as exc:
        _report_hub_error(exc, verbose=verbose)
        return 2
    except HubError as exc:
        _report_hub_error(exc, verbose=verbose)
        return 1
    except ValueError as exc:
        error(str(exc))
        return 2
    except NotImplementedError as exc:
        error(str(exc))
        return 2
    except Exception as exc:  # pragma: no cover - unexpected
        error(f"Unexpected error: {exc.__class__.__name__}: {exc}")
        if verbose:
            raise
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_cmd())
