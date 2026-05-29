"""``ms upload`` command — upload a single file or a folder."""

from __future__ import annotations

from argparse import Action
from pathlib import Path

from ..constants import RepoType
from .base import CLICommand, add_repo_type_arg, error, info, make_api, success


class UploadCommand(CLICommand):
    """Upload a local path to a repository.

    The local path is auto-detected as a file or directory. ``path_in_repo``
    defaults to the basename for files, or to the repo root for folders.
    """

    @staticmethod
    def register(subparsers: Action) -> None:
        p = subparsers.add_parser(
            "upload",
            help="Upload a file or folder to a repository.",
        )
        p.add_argument("repo_id", help="Canonical 'owner/name' identifier.")
        p.add_argument(
            "local_path",
            nargs="?",
            default=".",
            help="Local file or folder to upload (default: current directory).",
        )
        p.add_argument(
            "path_in_repo",
            nargs="?",
            default=None,
            help="Destination path inside the repo. Defaults to basename / root.",
        )
        add_repo_type_arg(
            p,
            choices=[RepoType.MODEL.value, RepoType.DATASET.value],
            default=RepoType.MODEL.value,
            required=False,
        )
        p.add_argument("--commit-message", dest="commit_message", default=None)
        p.add_argument("--revision", default=None, help="Target branch (default: master).")
        p.add_argument(
            "--include",
            dest="allow_patterns",
            action="append",
            default=None,
            help="Glob to include (folder mode). Repeatable.",
        )
        p.add_argument(
            "--exclude",
            dest="ignore_patterns",
            action="append",
            default=None,
            help="Glob to exclude (folder mode). Repeatable.",
        )
        p.add_argument(
            "--max-workers",
            dest="max_workers",
            type=int,
            default=4,
            help="Concurrency for folder uploads.",
        )
        p.set_defaults(_command=UploadCommand)

    def execute(self) -> None:
        api = make_api(self.args)
        local = Path(self.args.local_path).expanduser()
        if not local.exists():
            error(f"Local path not found: {local}")
            raise SystemExit(2)

        if local.is_file():
            path_in_repo = self.args.path_in_repo or local.name
            info(f"Uploading file {local} → {self.args.repo_id}:{path_in_repo}")
            api.upload_file(
                self.args.repo_id,
                self.args.repo_type,
                str(local),
                path_in_repo,
                commit_message=self.args.commit_message,
                revision=self.args.revision,
            )
            success("Upload complete.")
            return

        path_in_repo = self.args.path_in_repo or ""
        info(f"Uploading folder {local} → {self.args.repo_id}:{path_in_repo or '/'}")
        api.upload_folder(
            self.args.repo_id,
            self.args.repo_type,
            str(local),
            path_in_repo=path_in_repo,
            commit_message=self.args.commit_message,
            revision=self.args.revision,
            allow_patterns=self.args.allow_patterns,
            ignore_patterns=self.args.ignore_patterns,
            max_workers=self.args.max_workers,
        )
        success("Folder upload complete.")
