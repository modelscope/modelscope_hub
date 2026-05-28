"""``ms cache`` group — scan and clear the local cache."""

from __future__ import annotations

import sys
from argparse import _SubParsersAction

from ..constants import RepoType
from .base import CLICommand, info, make_api, render_table, success


class CacheCommand(CLICommand):
    """Top-level dispatcher for the ``cache`` subcommands."""

    @staticmethod
    def register(subparsers: _SubParsersAction) -> None:
        parser = subparsers.add_parser("cache", help="Inspect or clear the local cache.")
        sub = parser.add_subparsers(dest="cache_action", metavar="ACTION")
        sub.required = True

        _CacheScan.register(sub)
        _CacheClear.register(sub)

        parser.set_defaults(_command=CacheCommand)

    def execute(self) -> None:
        leaf = getattr(self.args, "_cache_leaf", None)
        if leaf is None:  # pragma: no cover
            raise SystemExit("No cache action given. See `ms cache --help`.")
        leaf(self.args).execute()


def _human_size(num: int) -> str:
    """Format a byte count using IEC suffixes."""
    value = float(num)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"  # pragma: no cover


class _CacheScan(CLICommand):
    @staticmethod
    def register(subparsers: _SubParsersAction) -> None:
        p = subparsers.add_parser("scan", help="Show cached repositories and disk usage.")
        p.add_argument("--cache-dir", dest="cache_dir", default=None)
        p.set_defaults(_command=CacheCommand, _cache_leaf=_CacheScan)

    def execute(self) -> None:
        api = make_api(self.args)
        report = api.scan_cache(self.args.cache_dir)
        info(f"cache_dir: {report.cache_dir or '-'}")
        info(f"total    : {_human_size(report.total_size)} across {report.total_repos} repo(s)")
        if not report.repos:
            return
        rows = [
            (
                r.repo_id or "-",
                getattr(r.repo_type, "value", r.repo_type) or "-",
                r.revision or "-",
                r.nb_files,
                _human_size(r.size_on_disk),
                r.local_path or "-",
            )
            for r in report.repos
        ]
        info("")
        info(render_table(
            rows,
            headers=["repo_id", "repo_type", "revision", "files", "size", "path"],
        ))


class _CacheClear(CLICommand):
    @staticmethod
    def register(subparsers: _SubParsersAction) -> None:
        p = subparsers.add_parser("clear", help="Remove cached files.")
        p.add_argument("--cache-dir", dest="cache_dir", default=None)
        p.add_argument(
            "--repo-type",
            dest="repo_type",
            choices=[t.value for t in RepoType],
            default=None,
        )
        p.add_argument("--repo-id", dest="repo_id", default=None)
        p.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation prompt.")
        p.set_defaults(_command=CacheCommand, _cache_leaf=_CacheClear)

    def execute(self) -> None:
        # Reject ambiguous "--repo-id without --repo-type" early so we never
        # accidentally fall through to the "clear everything" branch.
        if self.args.repo_id and not self.args.repo_type:
            print(
                "Error: --repo-type is required when using --repo-id",
                file=sys.stderr,
            )
            raise SystemExit(2)

        scope = self.args.repo_id or self.args.repo_type or "the entire cache"
        if not self.args.yes:
            answer = input(f"Clear {scope}? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                info("Aborted.")
                return
        api = make_api(self.args)
        freed = api.clear_cache(
            cache_dir=self.args.cache_dir,
            repo_type=self.args.repo_type,
            repo_id=self.args.repo_id,
        )
        success(f"Freed {_human_size(freed)} from cache.")
