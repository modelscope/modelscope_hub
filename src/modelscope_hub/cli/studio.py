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

from ..constants import RepoType
from .base import CLICommand, SubParsers, info, make_api, parse_kv_pairs, success

_LOG_TYPES = ("run", "build")
_STUDIO_SDK_TYPES = ("gradio", "streamlit", "docker", "static")
_SETTINGS_FIELDS = (
    "display_name",
    "description",
    "license",
    "cover_image",
    "sdk_type",
    "sdk_version",
    "base_image",
    "hardware",
    "private",
)


class StudioCommand(CLICommand):
    """Top-level dispatcher for ``studio`` subcommands."""

    @staticmethod
    def register(subparsers: SubParsers) -> None:
        parser = subparsers.add_parser("studio", help="Manage ModelScope Studio spaces.")
        _add_visible_auth_args(parser)
        actions = parser.add_subparsers(dest="studio_action", metavar="ACTION")
        actions.required = True

        _StudioDeploy.register(actions)
        _StudioStop.register(actions)
        _StudioLogs.register(actions)
        _StudioSettings.register(actions)
        _StudioSecret.register(actions)

        parser.set_defaults(_command=StudioCommand)

    def execute(self) -> None:
        leaf = getattr(self.args, "_studio_leaf", None)
        if leaf is None:  # pragma: no cover - argparse enforces this
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
        visibility.add_argument("--private", dest="private", action="store_const", const=True, default=None)
        visibility.add_argument("--public", dest="private", action="store_const", const=False)
        p.set_defaults(_command=StudioCommand, _studio_leaf=_StudioSettings)

    def execute(self) -> None:
        settings = _collect_settings(self.args)
        if not settings:
            raise ValueError(
                "No setting specified. Provide key=value or one of: --display-name, --description, --license, "
                "--cover-image, --sdk-type, --sdk-version, --base-image, --hardware, --private/--public."
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


def _collect_settings(args) -> dict[str, Any]:
    settings: dict[str, Any] = parse_kv_pairs(getattr(args, "settings", []) or [])
    for field in _SETTINGS_FIELDS:
        value = getattr(args, field, None)
        if value is not None:
            settings[field] = value
    return settings


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
    total = payload.get("total")
    if total is not None:
        info(f"-- page {page_num} (size {page_size}), total {total} --")
