"""``ms download`` command — fetch a single file or a full repo snapshot."""

from __future__ import annotations

from argparse import Action
from pathlib import Path

from ..constants import RepoType
from .base import CLICommand, add_repo_type_arg, info, make_api, success


class DownloadCommand(CLICommand):
    """Download files or whole repositories from ModelScope Hub."""

    @staticmethod
    def register(subparsers: Action) -> None:
        p = subparsers.add_parser(
            "download",
            help="Download a file or full snapshot of a repository.",
        )
        p.add_argument("repo_id", help="Canonical 'owner/name' identifier.")
        p.add_argument(
            "files",
            nargs="*",
            help="Optional list of file paths to download. Empty = full snapshot.",
        )
        add_repo_type_arg(
            p,
            choices=[RepoType.MODEL.value, RepoType.DATASET.value],
            default=RepoType.MODEL.value,
            required=False,
        )
        p.add_argument("--revision", default=None, help="Branch / tag / commit (default: master).")
        p.add_argument("--cache-dir", dest="cache_dir", default=None, help="Override cache directory.")
        p.add_argument(
            "--max-workers",
            dest="max_workers",
            type=int,
            default=4,
            help="Concurrency for full-repo snapshot downloads.",
        )
        p.add_argument(
            "--include",
            dest="allow_patterns",
            action="append",
            default=None,
            help="Glob to include (snapshot mode). Repeatable.",
        )
        p.add_argument(
            "--exclude",
            dest="ignore_patterns",
            action="append",
            default=None,
            help="Glob to exclude (snapshot mode). Repeatable.",
        )
        p.add_argument("--force", action="store_true", help="Re-download even if cached.")
        p.set_defaults(_command=DownloadCommand)

    def execute(self) -> None:
        api = make_api(self.args)
        cache_dir: Path | None = Path(self.args.cache_dir) if self.args.cache_dir else None

        if self.args.files:
            for file_path in self.args.files:
                local = api.download_file(
                    self.args.repo_id,
                    self.args.repo_type,
                    file_path,
                    revision=self.args.revision,
                    cache_dir=cache_dir,
                    force=self.args.force,
                )
                success(f"{file_path} → {local}")
            return

        info(f"Downloading snapshot of {self.args.repo_id} ({self.args.repo_type})…")
        local_dir = api.download_repo(
            self.args.repo_id,
            self.args.repo_type,
            revision=self.args.revision,
            cache_dir=cache_dir,
            allow_patterns=self.args.allow_patterns,
            ignore_patterns=self.args.ignore_patterns,
            max_workers=self.args.max_workers,
        )
        success(f"Snapshot ready at {local_dir}")
