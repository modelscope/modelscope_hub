"""``ms studio`` command group — manage Studio runtime resources.

Historically this command group lived in the umbrella ``modelscope`` SDK and was
registered into this CLI as a plugin.  The console scripts are now owned by
``modelscope-hub`` itself, so the Studio management surface must be available
from the hub package too (including hub-only installs such as ``ms-hub``).
"""

from __future__ import annotations

import json
from argparse import SUPPRESS
from typing import Any

from ..constants import RepoType, StudioVisibility
from .base import CLICommand, SubParsers, info, make_api, parse_kv_pairs, render_table, success

_LOG_TYPES = ("run", "build")
_STUDIO_SDK_TYPES = ("gradio", "streamlit", "docker", "static")
_STUDIO_SORTS = ("default", "last_modified", "view_num", "likes")
_STUDIO_STATUS_FILTERS = ("running", "all")
_STUDIO_HARDWARE_TYPES = ("xgpu", "amd")
_SETTINGS_FIELDS = (
    "display_name",
    "description",
    "license",
    "cover_image",
    "sdk_type",
    "sdk_version",
    "base_image",
    "hardware",
    "visibility",
    "private",
)


class StudioCommand(CLICommand):
    """Top-level dispatcher for ``studio`` subcommands."""

    # The other built-in commands do not need this, because the parser name is
    # the string handed to ``add_parser``. This one does: it is the single command
    # that crosses the package boundary (the umbrella SDK re-exports it as
    # ``StudioCMD``, and its plugin classes are all identified by ``name``), so
    # plugin-collision detection can key off the class instead of the entry-point
    # label it happens to be registered under.
    name = "studio"

    @staticmethod
    def register(subparsers: SubParsers) -> None:
        parser = subparsers.add_parser("studio", help="Manage ModelScope Studio spaces.")
        _add_visible_auth_args(parser)
        actions = parser.add_subparsers(dest="studio_action", metavar="ACTION")
        actions.required = True

        _StudioList.register(actions)
        _StudioDeploy.register(actions)
        _StudioStop.register(actions)
        _StudioLogs.register(actions)
        _StudioSettings.register(actions)
        _StudioSecret.register(actions)
        _StudioVariable.register(actions)
        _StudioHardware.register(actions)
        _StudioBaseImages.register(actions)
        _StudioSdkVersions.register(actions)

        parser.set_defaults(_command=StudioCommand)

    def execute(self) -> None:
        leaf = getattr(self.args, "_studio_leaf", None)
        if leaf is None:
            # Namespaces built by hand -- as embedders and the umbrella SDK's
            # tests do -- carry only the action name, because ``_studio_leaf`` is
            # set by argparse during dispatch. Fall back to the action so a
            # programmatic caller is not forced to know that internal detail.
            leaf = _STUDIO_LEAVES.get(getattr(self.args, "studio_action", None) or "")
        if leaf is None:
            raise SystemExit("No studio subcommand specified. Run 'modelscope studio --help'.")
        leaf(self.args).execute()


# Backward-compatible name used by the umbrella SDK's historical module.
StudioCMD = StudioCommand


def _add_visible_auth_args(parser) -> None:
    parser.add_argument("--token", dest="subcmd_token", default=None, help="Optional access token.")
    parser.add_argument("--endpoint", dest="subcmd_endpoint", default=None, help="ModelScope server endpoint.")


def _add_studio_id(parser) -> None:
    parser.add_argument("studio_id", help="Studio ID in the form 'owner/name'.")
    _add_leaf_auth_args(parser)


def _add_leaf_auth_args(parser) -> None:
    """Accept auth flags after a studio action without overwriting group flags."""
    parser.add_argument("--token", dest="subcmd_token", default=SUPPRESS, help=SUPPRESS)
    parser.add_argument("--endpoint", dest="subcmd_endpoint", default=SUPPRESS, help=SUPPRESS)


class _StudioList(CLICommand):
    @staticmethod
    def register(subparsers: SubParsers) -> None:
        p = subparsers.add_parser("list", help="List Studio spaces.")
        p.add_argument("--search", default=None, help="Match name, display name or author.")
        p.add_argument("--owner", default=None, help="Restrict to spaces owned by this user/org.")
        p.add_argument("--sort", choices=_STUDIO_SORTS, default=None)
        p.add_argument(
            "--status",
            choices=_STUDIO_STATUS_FILTERS,
            default=None,
            help="Runtime status filter. Defaults to 'all' when --owner is given, else to the server default.",
        )
        mcp = p.add_mutually_exclusive_group()
        mcp.add_argument("--mcp-support", dest="mcp_support", action="store_const", const=True, default=None)
        mcp.add_argument("--no-mcp-support", dest="mcp_support", action="store_const", const=False)
        p.add_argument("--hardware-type", dest="hardware_type", choices=_STUDIO_HARDWARE_TYPES, default=None)
        p.add_argument("--page", dest="page_number", type=int, default=1)
        p.add_argument("--page-size", dest="page_size", type=int, default=10)
        _add_leaf_auth_args(p)
        p.set_defaults(_command=StudioCommand, _studio_leaf=_StudioList)

    def execute(self) -> None:
        api = make_api(self.args)
        result = api.list_repos(
            RepoType.STUDIO,
            search=self.args.search,
            owner=self.args.owner,
            sort=self.args.sort,
            page_number=self.args.page_number,
            page_size=self.args.page_size,
            status=self._status(),
            mcp_support=self.args.mcp_support,
            hardware_type=self.args.hardware_type,
        )
        if not result.items:
            info("(no studios found)")
            return
        rows = [
            (
                r.repo_id or "-",
                _visibility_label(r.visibility),
                r.sdk_type or "-",
                r.hardware or "-",
                _runtime_status(r),
                r.likes,
            )
            for r in result.items
        ]
        info(render_table(rows, headers=["repo_id", "visibility", "sdk_type", "hardware", "status", "likes"]))
        info(f"\npage {result.page_number} / total {result.total_count} (page_size={result.page_size})")

    def _status(self) -> str | None:
        """Resolve the status filter, defaulting to ``all`` when listing an owner.

        The endpoint is documented as switching its own default to ``all`` once
        ``owner`` is set, but it does not: it keeps filtering to running spaces,
        so "list my spaces" answered with nothing at all. Asking for ``all``
        explicitly gives the command the meaning a user expects, while leaving
        the SDK faithful to whatever the server actually does.
        """
        if self.args.status:
            return str(self.args.status)
        return "all" if self.args.owner else None


class _StudioDeploy(CLICommand):
    @staticmethod
    def register(subparsers: SubParsers) -> None:
        p = subparsers.add_parser("deploy", help="Deploy (re-pull and rebuild) a Studio space.")
        _add_studio_id(p)
        p.set_defaults(_command=StudioCommand, _studio_leaf=_StudioDeploy)

    def execute(self) -> None:
        api = make_api(self.args)
        data = api.deploy_repo(self.args.studio_id, RepoType.STUDIO)
        success(f"Deploy triggered for studio {self.args.studio_id}.")
        _print_status(data)


class _StudioStop(CLICommand):
    @staticmethod
    def register(subparsers: SubParsers) -> None:
        p = subparsers.add_parser("stop", help="Stop a running Studio space.")
        _add_studio_id(p)
        p.set_defaults(_command=StudioCommand, _studio_leaf=_StudioStop)

    def execute(self) -> None:
        api = make_api(self.args)
        data = api.stop_repo(self.args.studio_id, RepoType.STUDIO)
        success(f"Stop triggered for studio {self.args.studio_id}.")
        _print_status(data)


class _StudioLogs(CLICommand):
    @staticmethod
    def register(subparsers: SubParsers) -> None:
        p = subparsers.add_parser("logs", help="Fetch Studio runtime or build logs.")
        _add_studio_id(p)
        p.add_argument("--type", "--log-type", dest="log_type", choices=_LOG_TYPES, default="run")
        p.add_argument("--keyword", default=None, help="Optional keyword to filter log lines.")
        p.add_argument("--page", "--page-num", dest="page_num", type=int, default=1)
        p.add_argument("--page-size", dest="page_size", type=int, default=100)
        p.add_argument("--start-timestamp", dest="start_timestamp", type=int, default=None)
        p.add_argument("--end-timestamp", dest="end_timestamp", type=int, default=None)
        p.set_defaults(_command=StudioCommand, _studio_leaf=_StudioLogs)

    def execute(self) -> None:
        api = make_api(self.args)
        payload = api.get_repo_logs(
            self.args.studio_id,
            RepoType.STUDIO,
            log_type=self.args.log_type,
            page_num=self.args.page_num,
            page_size=self.args.page_size,
            keyword=self.args.keyword,
            start_timestamp=self.args.start_timestamp,
            end_timestamp=self.args.end_timestamp,
        )
        _print_logs(payload, page_num=self.args.page_num, page_size=self.args.page_size)


class _StudioSettings(CLICommand):
    @staticmethod
    def register(subparsers: SubParsers) -> None:
        p = subparsers.add_parser("settings", help="Update Studio settings.")
        _add_studio_id(p)
        p.add_argument("settings", nargs="*", help="Optional key=value settings.")
        p.add_argument("--display-name", dest="display_name", default=None, help="Studio display name.")
        p.add_argument("--description", default=None, help="Studio description.")
        p.add_argument("--license", default=None, help="Studio license.")
        p.add_argument("--cover-image", dest="cover_image", default=None, help="Studio cover image URL.")
        p.add_argument("--sdk-type", dest="sdk_type", choices=_STUDIO_SDK_TYPES, default=None)
        p.add_argument("--sdk-version", dest="sdk_version", default=None)
        p.add_argument("--base-image", dest="base_image", default=None)
        p.add_argument("--hardware", default=None)
        visibility = p.add_mutually_exclusive_group()
        visibility.add_argument(
            "--visibility",
            choices=[v.value for v in StudioVisibility],
            default=None,
            help="public (code and app public), protected (app public, code hidden) or private.",
        )
        visibility.add_argument("--private", dest="private", action="store_const", const=True, default=None)
        visibility.add_argument("--public", dest="private", action="store_const", const=False)
        p.set_defaults(_command=StudioCommand, _studio_leaf=_StudioSettings)

    def execute(self) -> None:
        settings = _collect_settings(self.args)
        if not settings:
            raise ValueError(
                "No setting specified. Provide key=value or one of: --display-name, --description, --license, "
                "--cover-image, --sdk-type, --sdk-version, --base-image, --hardware, --visibility, "
                "--private/--public."
            )
        api = make_api(self.args)
        data = api.update_repo_settings(self.args.studio_id, RepoType.STUDIO, **settings)
        success(f"Updated settings for studio {self.args.studio_id}: {', '.join(sorted(settings))}.")
        if data:
            info(json.dumps(data, ensure_ascii=False, indent=2, default=str))


class _StudioSecret(CLICommand):
    @staticmethod
    def register(subparsers: SubParsers) -> None:
        parser = subparsers.add_parser("secret", help="Manage Studio environment variables (secrets).")
        actions = parser.add_subparsers(dest="secret_action", metavar="ACTION")
        actions.required = True

        list_p = actions.add_parser("list", help="List secret keys.")
        _add_studio_id(list_p)
        list_p.set_defaults(_command=StudioCommand, _studio_leaf=_StudioSecret)

        add_p = actions.add_parser("add", help="Add a secret.")
        _add_studio_id(add_p)
        add_p.add_argument("key")
        add_p.add_argument("value")
        add_p.set_defaults(_command=StudioCommand, _studio_leaf=_StudioSecret)

        update_p = actions.add_parser("update", help="Update an existing secret.")
        _add_studio_id(update_p)
        update_p.add_argument("key")
        update_p.add_argument("value")
        update_p.set_defaults(_command=StudioCommand, _studio_leaf=_StudioSecret)

        delete_p = actions.add_parser("delete", help="Delete a secret.")
        _add_studio_id(delete_p)
        delete_p.add_argument("key")
        delete_p.set_defaults(_command=StudioCommand, _studio_leaf=_StudioSecret)

    def execute(self) -> None:
        api = make_api(self.args)
        action = self.args.secret_action
        if action == "list":
            secrets = api.list_secrets(self.args.studio_id, RepoType.STUDIO)
            if not secrets:
                info("(no secrets)")
                return
            for item in secrets:
                info(str(item.get("key") if isinstance(item, dict) else item))
            return
        if action == "add":
            api.add_secret(self.args.studio_id, self.args.key, self.args.value, RepoType.STUDIO)
            success(f"Secret {self.args.key!r} added.")
            return
        if action == "update":
            api.update_secret(self.args.studio_id, self.args.key, self.args.value, RepoType.STUDIO)
            success(f"Secret {self.args.key!r} updated.")
            return
        if action == "delete":
            api.delete_secret(self.args.studio_id, self.args.key, RepoType.STUDIO)
            success(f"Secret {self.args.key!r} deleted.")
            return
        raise ValueError(f"Unknown secret subcommand: {action}")


class _StudioVariable(CLICommand):
    """``studio variable`` — plaintext environment variables.

    Deliberately separate from ``studio secret``: a variable's value is publicly
    visible, so ``list`` prints it, whereas a secret's value is never disclosed.
    """

    @staticmethod
    def register(subparsers: SubParsers) -> None:
        parser = subparsers.add_parser(
            "variable",
            help="Manage Studio plaintext environment variables (values are public).",
        )
        actions = parser.add_subparsers(dest="variable_action", metavar="ACTION")
        actions.required = True

        list_p = actions.add_parser("list", help="List variable keys and values.")
        _add_studio_id(list_p)
        list_p.set_defaults(_command=StudioCommand, _studio_leaf=_StudioVariable)

        add_p = actions.add_parser("add", help="Add a plaintext variable.")
        _add_studio_id(add_p)
        add_p.add_argument("key")
        add_p.add_argument("value")
        add_p.set_defaults(_command=StudioCommand, _studio_leaf=_StudioVariable)

        update_p = actions.add_parser("update", help="Update an existing plaintext variable.")
        _add_studio_id(update_p)
        update_p.add_argument("key")
        update_p.add_argument("value")
        update_p.set_defaults(_command=StudioCommand, _studio_leaf=_StudioVariable)

        delete_p = actions.add_parser("delete", help="Delete a plaintext variable.")
        _add_studio_id(delete_p)
        delete_p.add_argument("key")
        delete_p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation.")
        delete_p.set_defaults(_command=StudioCommand, _studio_leaf=_StudioVariable)

    def execute(self) -> None:
        action = self.args.variable_action
        if action == "delete" and not getattr(self.args, "yes", False):
            answer = input(f"Delete variable {self.args.key!r} from {self.args.studio_id}? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                info("Aborted.")
                return
        api = make_api(self.args)
        if action == "list":
            variables = api.list_variables(self.args.studio_id, RepoType.STUDIO)
            if not variables:
                info("(no variables)")
                return
            rows = [
                (item.get("key") or "-", item.get("value") or "") if isinstance(item, dict) else (str(item), "")
                for item in variables
            ]
            info(render_table(rows, headers=["key", "value"]))
            return
        if action == "add":
            api.add_variable(self.args.studio_id, self.args.key, self.args.value, RepoType.STUDIO)
            success(f"Variable {self.args.key!r} added.")
            return
        if action == "update":
            api.update_variable(self.args.studio_id, self.args.key, self.args.value, RepoType.STUDIO)
            success(f"Variable {self.args.key!r} updated.")
            return
        if action == "delete":
            api.delete_variable(self.args.studio_id, self.args.key, RepoType.STUDIO)
            success(f"Variable {self.args.key!r} deleted.")
            return
        raise ValueError(f"Unknown variable subcommand: {action}")


class _StudioHardware(CLICommand):
    @staticmethod
    def register(subparsers: SubParsers) -> None:
        p = subparsers.add_parser("hardware", help="List hardware tiers a Studio can be deployed on.")
        p.add_argument("--sdk-type", dest="sdk_type", choices=_STUDIO_SDK_TYPES, default=None)
        p.add_argument(
            "--studio",
            dest="studio_id",
            default=None,
            help="Scope free tiers to what this 'owner/name' space may use.",
        )
        _add_leaf_auth_args(p)
        p.set_defaults(_command=StudioCommand, _studio_leaf=_StudioHardware)

    def execute(self) -> None:
        api = make_api(self.args)
        items = api.list_studio_hardware(sdk_type=self.args.sdk_type, repo_id=self.args.studio_id)
        if not items:
            info("(no hardware available)")
            return
        rows = [
            (
                item.get("name") or "-",
                item.get("instance_type") or "-",
                item.get("resource_type") or "-",
                item.get("gpu_type") or "-",
                _stock_label(item),
                _cost_label(item),
            )
            for item in items
        ]
        info(
            render_table(
                rows,
                headers=["name", "instance_type", "resource_type", "gpu", "stock", "cost"],
            )
        )
        info("\nPass a paid tier as --hardware paid/<instance_type>.")


class _StudioBaseImages(CLICommand):
    @staticmethod
    def register(subparsers: SubParsers) -> None:
        p = subparsers.add_parser("base-images", help="List base images available to Studio spaces.")
        _add_leaf_auth_args(p)
        p.set_defaults(_command=StudioCommand, _studio_leaf=_StudioBaseImages)

    def execute(self) -> None:
        api = make_api(self.args)
        items = api.list_studio_base_images()
        if not items:
            info("(no base images available)")
            return
        rows = [(item.get("name") or "-", item.get("tag") or "-") for item in items]
        info(render_table(rows, headers=["name", "tag"]))


class _StudioSdkVersions(CLICommand):
    @staticmethod
    def register(subparsers: SubParsers) -> None:
        p = subparsers.add_parser("sdk-versions", help="List SDK versions available to Studio spaces.")
        p.add_argument(
            "--sdk-type",
            dest="sdk_type",
            choices=_STUDIO_SDK_TYPES,
            default="gradio",
            help="Only 'gradio' publishes versions (default: gradio).",
        )
        _add_leaf_auth_args(p)
        p.set_defaults(_command=StudioCommand, _studio_leaf=_StudioSdkVersions)

    def execute(self) -> None:
        api = make_api(self.args)
        items = api.list_studio_sdk_versions(sdk_type=self.args.sdk_type)
        if not items:
            info(f"(no SDK versions published for sdk_type={self.args.sdk_type!r})")
            return
        rows = [(item.get("sdk_type") or "-", item.get("version") or "-", item.get("tag") or "-") for item in items]
        info(render_table(rows, headers=["sdk_type", "version", "tag"]))


def _collect_settings(args) -> dict[str, Any]:
    settings: dict[str, Any] = parse_kv_pairs(getattr(args, "settings", []) or [])
    for field in _SETTINGS_FIELDS:
        value = getattr(args, field, None)
        if value is not None:
            settings[field] = value
    return settings


# Action name -> leaf handler, for callers that dispatch without argparse.
# Declared after the leaf classes because it references them.
_STUDIO_LEAVES: dict[str, type[CLICommand]] = {
    "list": _StudioList,
    "deploy": _StudioDeploy,
    "stop": _StudioStop,
    "logs": _StudioLogs,
    "settings": _StudioSettings,
    "secret": _StudioSecret,
    "variable": _StudioVariable,
    "hardware": _StudioHardware,
    "base-images": _StudioBaseImages,
    "sdk-versions": _StudioSdkVersions,
}


def _visibility_label(value: object) -> str:
    """Render a visibility that may be an enum or the raw ``protected`` string."""
    if value is None:
        return "-"
    return getattr(value, "label", None) or getattr(value, "name", None) or str(value)


def _runtime_status(repo: object) -> str:
    runtime = getattr(repo, "runtime", None)
    if isinstance(runtime, dict):
        return str(runtime.get("status") or "-")
    return "-"


def _stock_label(item: dict) -> str:
    """Summarise availability, distinguishing "none left" from "not reported"."""
    if item.get("has_stock") is False:
        return "out of stock"
    stock = item.get("stock")
    if stock is None:
        return "-"
    return str(stock)


def _cost_label(item: dict) -> str:
    """Show the discounted price, noting the original when it differs."""
    cost = item.get("cost_after_discount")
    original = item.get("original_cost")
    if cost is None and original is None:
        return "free"
    if cost is None:
        return str(original)
    if original in (None, cost):
        return str(cost)
    return f"{cost} (was {original})"


def _print_status(data: object) -> None:
    if not data:
        return
    if isinstance(data, dict):
        status = data.get("status") or data.get("Status")
        if status:
            info(f"Status: {status}")
            return
    info(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _print_logs(payload: object, *, page_num: int, page_size: int) -> None:
    if not isinstance(payload, dict):
        info(str(payload))
        return
    logs = payload.get("logs")
    if logs is None:
        info(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return
    for entry in logs:
        if isinstance(entry, dict):
            ts = entry.get("timestamp") or entry.get("time") or ""
            msg = entry.get("content") or entry.get("message") or ""
            info(f"[{ts}] {msg}" if ts else str(msg))
        else:
            info(str(entry))
    # The response reports ``total_count`` / ``total_page_num``; the older
    # ``total`` spelling this used to read never existed, so the footer was
    # always suppressed.
    total = payload.get("total_count")
    if total is None:
        total = payload.get("total")
    if total is None:
        return
    footer = f"-- page {page_num} (size {page_size}), total {total}"
    total_pages = payload.get("total_page_num")
    if total_pages is not None:
        footer += f" across {total_pages} page(s)"
    info(f"{footer} --")
