# Copyright (c) Alibaba, Inc. and its affiliates.
"""``ms agent`` command -- manage agent workspace files."""

from __future__ import annotations

from argparse import Action

from ..agent import (
    FRAMEWORK_REGISTRY,
    available_frameworks,
    cmd_convert,
    cmd_download,
    cmd_recover,
    cmd_status,
    cmd_stop,
    cmd_upload,
    cmd_watch,
)
from .base import CLICommand


class AgentCommand(CLICommand):
    """Agent workspace management: upload, download, convert, watch, stop, recover."""

    @staticmethod
    def register(subparsers: Action) -> None:
        agent_parser = subparsers.add_parser(
            "agent",
            help="Manage agent workspace files (upload/download/convert/watch).",
        )
        agent_sub = agent_parser.add_subparsers(dest="agent_command", metavar="ACTION")
        agent_sub.required = True

        # --- Common arguments factory ---
        def _add_common(p, *, needs_framework=True, needs_name=True):
            if needs_framework:
                p.add_argument(
                    "-f", "--framework",
                    required=True,
                    help=f"Agent framework ({available_frameworks()}).",
                )
            if needs_name:
                p.add_argument(
                    "-n", "--name",
                    default=None,
                    help="Sub-agent name (auto-detected if omitted).",
                )
            p.add_argument(
                "--local_dir",
                default=None,
                help="Override local workspace root path.",
            )

        # --- upload ---
        p_up = agent_sub.add_parser("upload", help="Upload local agent files to remote.")
        _add_common(p_up)
        p_up.add_argument("--repo", default=None, help="Remote repo name (derived if omitted).")
        p_up.add_argument("--dry-run", action="store_true", help="Show files without uploading.")

        # --- download ---
        p_dl = agent_sub.add_parser("download", help="Download remote agent files to local.")
        _add_common(p_dl)
        p_dl.add_argument("--repo", required=True, help="Remote repo name.")
        p_dl.add_argument("--target", default=None, help="Target framework for conversion.")
        p_dl.add_argument("--dry-run", action="store_true", help="Show files without writing.")

        # --- convert ---
        p_cv = agent_sub.add_parser("convert", help="Convert workspace between frameworks locally.")
        p_cv.add_argument("--from", dest="source", required=True, help="Source framework.")
        p_cv.add_argument("--to", dest="target", required=True, help="Target framework.")
        p_cv.add_argument("-n", "--name", default=None, help="Sub-agent name.")
        p_cv.add_argument("--local_dir", default=None, help="Source workspace root.")
        p_cv.add_argument("--out_dir", default=None, help="Target output root.")
        p_cv.add_argument("--dry-run", action="store_true", help="Show result without writing.")

        # --- watch ---
        p_wa = agent_sub.add_parser("watch", help="Start background sync daemon.")
        _add_common(p_wa)
        p_wa.add_argument("--repo", default=None, help="Remote repo name (derived if omitted).")
        p_wa.add_argument("--pull", action="store_true", help="Enable bidirectional sync.")

        # --- stop ---
        agent_sub.add_parser("stop", help="Stop background watch daemon.")

        # --- status ---
        p_st = agent_sub.add_parser("status", help="List discoverable sub-agents.")
        _add_common(p_st, needs_name=False)

        # --- recover ---
        p_rc = agent_sub.add_parser("recover", help="Restore files from backup.")
        p_rc.add_argument("target", nargs="?", default=None, help="'last' or a backup filename.")
        p_rc.add_argument("-f", "--framework", default=None, help="Framework to restore.")
        p_rc.add_argument("-n", "--name", default=None, help="Agent name filter.")
        p_rc.add_argument("--local_dir", default=None, help="Override workspace root.")
        p_rc.add_argument("--list", action="store_true", help="List available backups.")

    def execute(self) -> None:
        args = self.args
        action = args.agent_command

        # Resolve credentials from global args (token/endpoint from parent parser).
        token = getattr(args, "token", None)
        endpoint = getattr(args, "endpoint", None)

        # For commands that need auth, resolve username from the API.
        # AgentApi falls back to HubConfig (ms login credentials) when args are None.
        username = None
        if action in ("upload", "download", "watch"):
            from ..agent import AgentApi
            try:
                client = AgentApi(endpoint=endpoint, token=token)
                username = client.get_username()
                token = token or client.token
                endpoint = endpoint or client.server
            except Exception:
                pass

        if action == "upload":
            rc = cmd_upload(
                framework=args.framework,
                name=args.name,
                local_dir=args.local_dir,
                repo=args.repo,
                dry_run=args.dry_run,
                endpoint=endpoint,
                token=token,
                username=username,
            )
        elif action == "download":
            rc = cmd_download(
                framework=args.framework,
                repo=args.repo,
                name=args.name,
                target=args.target,
                local_dir=args.local_dir,
                dry_run=args.dry_run,
                endpoint=endpoint,
                token=token,
                username=username,
            )
        elif action == "convert":
            rc = cmd_convert(
                source_fw=args.source,
                target_fw=args.target,
                name=args.name,
                local_dir=args.local_dir,
                out_dir=args.out_dir,
                dry_run=args.dry_run,
            )
        elif action == "watch":
            rc = cmd_watch(
                framework=args.framework,
                name=args.name,
                local_dir=args.local_dir,
                repo=args.repo,
                pull=args.pull,
                endpoint=endpoint,
                token=token,
                username=username,
            )
        elif action == "stop":
            rc = cmd_stop()
        elif action == "status":
            rc = cmd_status(
                framework=args.framework,
                local_dir=args.local_dir,
            )
        elif action == "recover":
            rc = cmd_recover(
                target=args.target,
                framework=getattr(args, "framework", None),
                name=getattr(args, "name", None),
                local_dir=getattr(args, "local_dir", None),
                list_backups=args.list,
            )
        else:
            print(f"Unknown agent action: {action}")
            rc = 1

        if rc != 0:
            raise SystemExit(rc)
