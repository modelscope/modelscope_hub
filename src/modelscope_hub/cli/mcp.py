"""``ms mcp`` group — list / info / deploy / undeploy MCP servers."""

from __future__ import annotations

import json

from .base import CLICommand, SubParsers, info, make_api, parse_kv_pairs, render_table, success
from .compat import add_subcmd_token_endpoint


class McpCommand(CLICommand):
    """Top-level dispatcher for the ``mcp`` subcommands."""

    @staticmethod
    def register(subparsers: SubParsers) -> None:
        parser = subparsers.add_parser("mcp", help="Manage MCP servers.")
        sub = parser.add_subparsers(dest="mcp_action", metavar="ACTION")
        sub.required = True

        _McpList.register(sub)
        _McpInfo.register(sub)
        _McpDeploy.register(sub)
        _McpUndeploy.register(sub)

        parser.set_defaults(_command=McpCommand)

    def execute(self) -> None:
        leaf = getattr(self.args, "_mcp_leaf", None)
        if leaf is None:  # pragma: no cover
            raise SystemExit("No mcp action given. See `ms mcp --help`.")
        leaf(self.args).execute()


class _McpList(CLICommand):
    @staticmethod
    def register(subparsers: SubParsers) -> None:
        p = subparsers.add_parser("list", help="List MCP servers.")
        p.add_argument("--search", default=None)
        p.add_argument("--page", dest="page_number", type=int, default=1)
        p.add_argument("--page-size", dest="page_size", type=int, default=20)
        add_subcmd_token_endpoint(p)
        p.set_defaults(_command=McpCommand, _mcp_leaf=_McpList)

    def execute(self) -> None:
        api = make_api(self.args)
        result = api.list_mcp_servers(
            search=self.args.search,
            page_number=self.args.page_number,
            page_size=self.args.page_size,
        )
        if not result.items:
            info("(no MCP servers found)")
            return
        rows = [
            (
                item.get("id") or item.get("Id") or "-",
                item.get("name") or item.get("Name") or "-",
                item.get("status") or item.get("Status") or "-",
                item.get("description") or item.get("Description") or "-",
            )
            for item in result.items
        ]
        info(render_table(rows, headers=["id", "name", "status", "description"]))
        info(f"\npage {result.page_number} / total {result.total_count}")


class _McpInfo(CLICommand):
    @staticmethod
    def register(subparsers: SubParsers) -> None:
        p = subparsers.add_parser("info", help="Show details of an MCP server.")
        p.add_argument("server_id")
        add_subcmd_token_endpoint(p)
        p.set_defaults(_command=McpCommand, _mcp_leaf=_McpInfo)

    def execute(self) -> None:
        api = make_api(self.args)
        data = api.get_mcp_server(self.args.server_id)
        info(json.dumps(data, indent=2, ensure_ascii=False, default=str))


class _McpDeploy(CLICommand):
    @staticmethod
    def register(subparsers: SubParsers) -> None:
        p = subparsers.add_parser("deploy", help="Deploy an MCP server.")
        p.add_argument("server_id")
        p.add_argument(
            "--transport-type",
            dest="transport_type",
            default=None,
            choices=["sse", "streamable_http"],
            help="Transport type (default: sse).",
        )
        p.add_argument(
            "--expiration-minutes",
            dest="expiration_minutes",
            type=int,
            default=None,
            help="Expiration time in minutes (-1 for no expiration).",
        )
        p.add_argument(
            "--auth-check",
            dest="auth_check",
            action="store_true",
            default=None,
            help="Require a ModelScope token when connecting to the remote URL.",
        )
        p.add_argument(
            "--env",
            dest="env",
            action="append",
            default=None,
            metavar="KEY=VALUE",
            help="Environment variable for the MCP server (repeatable). "
            "Available keys are listed in the server's env_schema (see `mcp info`).",
        )
        add_subcmd_token_endpoint(p)
        p.set_defaults(_command=McpCommand, _mcp_leaf=_McpDeploy)

    def execute(self) -> None:
        api = make_api(self.args)
        payload: dict = {}
        if self.args.transport_type:
            payload["transport_type"] = self.args.transport_type
        if self.args.expiration_minutes is not None:
            payload["expiration_minutes"] = self.args.expiration_minutes
        if self.args.auth_check is not None:
            payload["auth_check"] = self.args.auth_check
        if self.args.env:
            payload["env_info"] = parse_kv_pairs(self.args.env)
        result = api.deploy_mcp_server(self.args.server_id, payload=payload or None)
        success(f"Deploy requested for MCP server: {self.args.server_id}")
        _print_operational_url(result)


def _print_operational_url(result: object) -> None:
    """Surface the deployed endpoint (McpOperationalUrl) when present."""
    if not isinstance(result, dict):
        return
    url = result.get("url")
    if not url:
        return
    info(f"  url: {url}")
    if result.get("transport_type"):
        info(f"  transport: {result['transport_type']}")
    if result.get("expiration"):
        info(f"  expires: {result['expiration']}")
    if result.get("auth_required") is not None:
        info(f"  auth required: {result['auth_required']}")


class _McpUndeploy(CLICommand):
    @staticmethod
    def register(subparsers: SubParsers) -> None:
        p = subparsers.add_parser("undeploy", help="Undeploy an MCP server.")
        p.add_argument("server_id")
        add_subcmd_token_endpoint(p)
        p.set_defaults(_command=McpCommand, _mcp_leaf=_McpUndeploy)

    def execute(self) -> None:
        api = make_api(self.args)
        api.undeploy_mcp_server(self.args.server_id)
        success(f"Undeploy requested for MCP server: {self.args.server_id}")
