"""``ms repo`` group — create / info / delete / list."""

from __future__ import annotations

from argparse import _SubParsersAction

from ..constants import RepoType
from ..types import RepoInfo
from .base import CLICommand, add_repo_type_arg, info, make_api, render_table, success


# ---------------------------------------------------------------------------
# Group dispatcher
# ---------------------------------------------------------------------------
class RepoCommand(CLICommand):
    """Top-level dispatcher for the ``repo`` subcommands."""

    @staticmethod
    def register(subparsers: _SubParsersAction) -> None:
        parser = subparsers.add_parser(
            "repo",
            help="Manage repositories (model, dataset, studio, skill).",
        )
        sub = parser.add_subparsers(dest="repo_action", metavar="ACTION")
        sub.required = True

        _RepoCreate.register(sub)
        _RepoInfo.register(sub)
        _RepoDelete.register(sub)
        _RepoList.register(sub)

        parser.set_defaults(_command=RepoCommand)

    def execute(self) -> None:
        # Each leaf command sets its own ``_command`` via ``set_defaults``.
        leaf = getattr(self.args, "_repo_leaf", None)
        if leaf is None:  # pragma: no cover - argparse ensures coverage
            raise SystemExit("No repo action given. See `ms repo --help`.")
        leaf(self.args).execute()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_visibility(value: object) -> str:
    if value is None:
        return "-"
    return getattr(value, "label", None) or getattr(value, "name", None) or str(value)


def _print_repo_info(repo: RepoInfo) -> None:
    info(f"id         : {repo.id if repo.id is not None else '-'}")
    info(f"repo_id    : {repo.repo_id or '-'}")
    info(f"repo_type  : {getattr(repo.repo_type, 'value', repo.repo_type) or '-'}")
    info(f"visibility : {_format_visibility(repo.visibility)}")
    info(f"license    : {repo.license or '-'}")
    info(f"downloads  : {repo.downloads}")
    info(f"likes      : {repo.likes}")
    if repo.description:
        info(f"description: {repo.description}")
    if repo.tags:
        info(f"tags       : {', '.join(repo.tags)}")


# ---------------------------------------------------------------------------
# Leaf commands
# ---------------------------------------------------------------------------
class _RepoCreate(CLICommand):
    @staticmethod
    def register(subparsers: _SubParsersAction) -> None:
        p = subparsers.add_parser("create", help="Create a new repository.")
        p.add_argument("repo_id", help="Canonical 'owner/name' identifier.")
        add_repo_type_arg(p)
        p.add_argument("--visibility", choices=["public", "private", "internal"], default=None)
        p.add_argument("--license", dest="license", default=None)
        p.add_argument("--chinese-name", dest="chinese_name", default=None)
        p.add_argument("--description", dest="description", default=None)
        # Studio-specific options. They are accepted for any repo type for a
        # uniform CLI surface and forwarded as extra kwargs — the API layer
        # only emits them for Studios.
        p.add_argument(
            "--sdk-type",
            dest="sdk_type",
            choices=["gradio", "streamlit", "docker", "static"],
            default=None,
            help="Studio SDK type.",
        )
        p.add_argument("--sdk-version", dest="sdk_version", default=None, help="Studio SDK version.")
        p.add_argument("--base-image", dest="base_image", default=None, help="Studio base image.")
        p.add_argument("--cover-image", dest="cover_image", default=None, help="Studio cover image URL.")
        p.add_argument("--hardware", dest="hardware", default=None, help="Studio hardware spec.")
        p.set_defaults(_command=RepoCommand, _repo_leaf=_RepoCreate)

    def execute(self) -> None:
        api = make_api(self.args)
        extra: dict[str, object] = {}
        for key in ("sdk_type", "sdk_version", "base_image", "cover_image", "hardware"):
            value = getattr(self.args, key, None)
            if value is not None:
                extra[key] = value
        repo = api.create_repo(
            self.args.repo_id,
            self.args.repo_type,
            visibility=self.args.visibility,
            license=self.args.license,
            chinese_name=self.args.chinese_name,
            description=self.args.description,
            **extra,
        )
        success(f"Created {self.args.repo_type}: {repo.repo_id or self.args.repo_id}")


class _RepoInfo(CLICommand):
    @staticmethod
    def register(subparsers: _SubParsersAction) -> None:
        p = subparsers.add_parser("info", help="Show metadata for a repository.")
        p.add_argument("repo_id")
        add_repo_type_arg(p)
        p.set_defaults(_command=RepoCommand, _repo_leaf=_RepoInfo)

    def execute(self) -> None:
        api = make_api(self.args)
        repo = api.get_repo(self.args.repo_id, self.args.repo_type)
        _print_repo_info(repo)


class _RepoDelete(CLICommand):
    @staticmethod
    def register(subparsers: _SubParsersAction) -> None:
        p = subparsers.add_parser("delete", help="Delete a repository (model or dataset).")
        p.add_argument("repo_id")
        add_repo_type_arg(p, choices=[RepoType.MODEL.value, RepoType.DATASET.value])
        p.add_argument("--yes", "-y", action="store_true", help="Skip the confirmation prompt.")
        p.set_defaults(_command=RepoCommand, _repo_leaf=_RepoDelete)

    def execute(self) -> None:
        if not self.args.yes:
            answer = input(
                f"Delete {self.args.repo_type} {self.args.repo_id!r}? This cannot be undone. [y/N] "
            ).strip().lower()
            if answer not in ("y", "yes"):
                info("Aborted.")
                return
        api = make_api(self.args)
        api.delete_repo(self.args.repo_id, self.args.repo_type)
        success(f"Deleted {self.args.repo_type}: {self.args.repo_id}")


class _RepoList(CLICommand):
    @staticmethod
    def register(subparsers: _SubParsersAction) -> None:
        p = subparsers.add_parser("list", help="List repositories of a given type.")
        add_repo_type_arg(
            p,
            choices=[
                RepoType.MODEL.value,
                RepoType.DATASET.value,
                RepoType.SKILL.value,
                RepoType.MCP.value,
            ],
        )
        p.add_argument("--owner", default=None)
        p.add_argument("--search", default=None)
        p.add_argument("--page", dest="page_number", type=int, default=1)
        p.add_argument("--page-size", dest="page_size", type=int, default=10)
        p.set_defaults(_command=RepoCommand, _repo_leaf=_RepoList)

    def execute(self) -> None:
        api = make_api(self.args)
        result = api.list_repos(
            self.args.repo_type,
            owner=self.args.owner,
            search=self.args.search,
            page_number=self.args.page_number,
            page_size=self.args.page_size,
        )
        if not result.items:
            info("(no repositories found)")
            return
        rows = [
            (
                r.repo_id or "-",
                _format_visibility(r.visibility),
                r.downloads,
                r.likes,
                r.license or "-",
            )
            for r in result.items
        ]
        info(render_table(rows, headers=["repo_id", "visibility", "downloads", "likes", "license"]))
        info(
            f"\npage {result.page_number} / total {result.total_count} "
            f"(page_size={result.page_size})"
        )
