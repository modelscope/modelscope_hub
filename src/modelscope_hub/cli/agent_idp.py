"""``ms agent-idp`` — manage Agent-IDP identities and signed token issuance.

This command is intentionally separate from ``ms agent``: the latter transfers
raw files to Agent repositories, while this command manages Ed25519 identities,
OIDC discovery, and short-lived JWT issuance.
"""

from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..agent_idp import (
    generate_agent_key_pair,
    load_private_jwk,
    public_jwk_from_private,
    write_private_jwk,
)
from ..errors import InvalidParameter
from ..types import AgentIdConfiguration, AgentIdentity
from .base import CLICommand, SubParsers, info, make_api, render_table

__all__ = ["AgentIdpCommand"]


_EXPIRY_CHOICES = (300, 600, 1800, 3600)
_STATUS_CHOICES = ("active", "paused")


def _read_public_jwk(path: str) -> dict[str, Any]:
    """Read an upload-safe public JWK without accepting a private ``d`` member."""
    target = Path(path).expanduser()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidParameter(f"Public JWK file is not valid JSON: {target}.") from exc
    if not isinstance(payload, Mapping) or "d" in payload:
        raise InvalidParameter("Public JWK file must be an object without private key material ('d').")
    # The low-level client performs the complete protocol validation. Constructing
    # the dataclass here keeps formatting and optional JWK fields consistent.
    return {key: value for key, value in payload.items() if value is not None}


def _public_jwk_from_args(args: Namespace) -> dict[str, Any]:
    if getattr(args, "private_key_file", None):
        return public_jwk_from_private(load_private_jwk(args.private_key_file)).to_dict()
    return _read_public_jwk(args.public_jwk_file)


def _identity_json(identity: AgentIdentity) -> dict[str, Any]:
    """Render identity data while never serialising local private material."""
    result: dict[str, Any] = {
        "agent_id": identity.agent_id,
        "agent_name": identity.agent_name,
        "description": identity.description,
        "token_expire_time": identity.token_expire_time,
        "principal": identity.principal,
        "kid": identity.kid,
        "status": identity.status,
        "create_time": identity.create_time,
        "update_time": identity.update_time,
    }
    if identity.public_key is not None:
        result["public_key"] = identity.public_key.to_dict()
    return {key: value for key, value in result.items() if value is not None and value != ""}


def _configuration_json(configuration: AgentIdConfiguration) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "issuer": configuration.issuer,
            "token_endpoint": configuration.token_endpoint,
            "jwks_uri": configuration.jwks_uri,
            "registration_endpoint": configuration.registration_endpoint,
            "activity_endpoint": configuration.activity_endpoint,
            "id_token_signing_alg_values_supported": configuration.id_token_signing_alg_values_supported,
        }.items()
        if value is not None
    }


class AgentIdpCommand(CLICommand):
    """Manage Agent-IDP identities, keys, OIDC discovery, and JWT issuance."""

    name = "agent-idp"

    @staticmethod
    def register(subparsers: SubParsers) -> None:
        parser = subparsers.add_parser(
            AgentIdpCommand.name,
            help="Manage Agent-IDP identities, signing keys, and OIDC metadata.",
        )
        parser.set_defaults(_command=AgentIdpCommand)
        actions = parser.add_subparsers(dest="agent_idp_action", metavar="ACTION")
        actions.required = True

        keygen = actions.add_parser("keygen", help="Generate an Ed25519 private JWK file (0600).")
        keygen.add_argument("--private-key-out", required=True)
        keygen.add_argument("--kid", default=None)
        keygen.add_argument("--force", action="store_true", help="Replace an existing private-key file.")

        create = actions.add_parser("create", help="Register an Agent-IDP identity.")
        create.add_argument("--agent-name", required=True)
        create.add_argument("--description", default=None)
        create.add_argument("--token-expire-time", type=int, choices=_EXPIRY_CHOICES, default=None)
        AgentIdpCommand._add_key_source(create)

        get = actions.add_parser("get", help="Show one Agent-IDP identity.")
        get.add_argument("agent_id")

        update = actions.add_parser("update", help="Update Agent-IDP identity metadata.")
        update.add_argument("agent_id")
        update.add_argument("--agent-name", default=None)
        update.add_argument("--description", default=None)
        update.add_argument("--token-expire-time", type=int, choices=_EXPIRY_CHOICES, default=None)

        delete = actions.add_parser("delete", help="Delete an Agent-IDP identity.")
        delete.add_argument("agent_id")
        delete.add_argument("--yes", action="store_true", help="Confirm permanent deletion.")

        reset = actions.add_parser("reset-key", help="Replace an Agent-IDP identity public key.")
        reset.add_argument("agent_id")
        AgentIdpCommand._add_key_source(reset)

        pause = actions.add_parser("pause", help="Pause or resume Agent-IDP token issuance.")
        pause.add_argument("agent_id")
        state = pause.add_mutually_exclusive_group(required=True)
        state.add_argument("--paused", action="store_true", help="Pause token issuance.")
        state.add_argument("--active", action="store_true", help="Resume token issuance.")

        listed = actions.add_parser("list", help="List a user's Agent-IDP identities.")
        listed.add_argument("username")
        listed.add_argument("--status", choices=_STATUS_CHOICES, default=None)
        AgentIdpCommand._add_paging(listed)

        records = actions.add_parser("list-tokens", help="List non-sensitive issued-token records.")
        records.add_argument("agent_id")
        AgentIdpCommand._add_paging(records)

        issue = actions.add_parser("issue-token", help="Sign locally and exchange an Agent-IDP JWT request.")
        issue.add_argument("--agent-id", required=True)
        issue.add_argument("--audience", required=True)
        issue.add_argument("--private-key-file", required=True)
        issue.add_argument("--timestamp", type=int, default=None)

        actions.add_parser("configuration", help="Show anonymous Agent-IDP OIDC discovery metadata.")
        actions.add_parser("jwks", help="Show anonymous Agent-IDP JWT verification keys.")

    @staticmethod
    def _add_key_source(parser: Any) -> None:
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--private-key-file", default=None, help="Owner-only Ed25519 private JWK file.")
        source.add_argument("--public-jwk-file", default=None, help="Public Ed25519 JWK file (for HSM-managed keys).")

    @staticmethod
    def _add_paging(parser: Any) -> None:
        parser.add_argument("--page", type=int, default=1)
        parser.add_argument("--page-size", type=int, choices=range(1, 51), default=20)

    def execute(self) -> None:
        action = self.args.agent_idp_action
        if action == "keygen":
            private_jwk, public_jwk = generate_agent_key_pair(self.args.kid)
            write_private_jwk(self.args.private_key_out, private_jwk, overwrite=self.args.force)
            info(json.dumps(public_jwk.to_dict(), ensure_ascii=False, sort_keys=True))
            return

        api = make_api(self.args)
        if action == "create":
            payload = {
                "agent_name": self.args.agent_name,
                "public_key": _public_jwk_from_args(self.args),
                "description": self.args.description,
                "token_expire_time": self.args.token_expire_time,
            }
            info(json.dumps(_identity_json(api.create_agent_identity(payload)), ensure_ascii=False, indent=2))
        elif action == "get":
            info(json.dumps(_identity_json(api.get_agent_identity(self.args.agent_id)), ensure_ascii=False, indent=2))
        elif action == "update":
            payload = {
                key: value
                for key, value in {
                    "agent_name": self.args.agent_name,
                    "description": self.args.description,
                    "token_expire_time": self.args.token_expire_time,
                }.items()
                if value is not None
            }
            identity = api.update_agent_identity(self.args.agent_id, payload)
            info(json.dumps(_identity_json(identity), ensure_ascii=False, indent=2))
        elif action == "delete":
            if not self.args.yes:
                raise InvalidParameter("Deletion requires --yes.")
            api.delete_agent_identity(self.args.agent_id)
            info(f"Deleted Agent-IDP identity {self.args.agent_id}.")
        elif action == "reset-key":
            identity = api.reset_agent_key_pair(
                self.args.agent_id,
                {"public_key": _public_jwk_from_args(self.args)},
            )
            info(json.dumps(_identity_json(identity), ensure_ascii=False, indent=2))
        elif action == "pause":
            api.pause_agent(self.args.agent_id, paused=self.args.paused)
            info(f"Agent-IDP identity {self.args.agent_id} is now {'paused' if self.args.paused else 'active'}.")
        elif action == "list":
            identity_page = api.list_user_agent_identities(
                self.args.username,
                status=self.args.status,
                page=self.args.page,
                page_size=self.args.page_size,
            )
            identity_rows = [
                (
                    item.agent_id,
                    item.agent_name,
                    item.kid or "-",
                    item.status or "-",
                    item.token_expire_time or "-",
                    item.create_time or "-",
                )
                for item in identity_page.items
            ]
            info(render_table(identity_rows, headers=["agent_id", "name", "kid", "status", "expiry", "created"]))
            info(
                f"page {identity_page.page_number} / total {identity_page.total_count} "
                f"(page_size={identity_page.page_size})"
            )
        elif action == "list-tokens":
            token_page = api.list_agent_token_records(
                self.args.agent_id,
                page=self.args.page,
                page_size=self.args.page_size,
            )
            token_rows = [
                (item.token_id, item.audience, item.issued_at or "-", item.expire_at or "-", item.status or "-")
                for item in token_page.items
            ]
            info(render_table(token_rows, headers=["token_id", "audience", "issued", "expires", "status"]))
            info(f"page {token_page.page_number} / total {token_page.total_count} (page_size={token_page.page_size})")
        elif action == "issue-token":
            key = load_private_jwk(self.args.private_key_file)
            token = api.issue_agent_token_with_private_key(
                key,
                agent_id=self.args.agent_id,
                audience=self.args.audience,
                timestamp=self.args.timestamp,
            )
            # Deliberately the only stdout content: callers may pipe this exact
            # credential to another process without parsing a status message.
            print(token.access_token)
        elif action == "configuration":
            info(json.dumps(_configuration_json(api.get_agent_id_configuration()), ensure_ascii=False, indent=2))
        elif action == "jwks":
            keys = [key.to_dict() for key in api.get_agent_id_jwks()]
            info(json.dumps({"keys": keys}, ensure_ascii=False, indent=2))
        else:  # pragma: no cover - argparse makes this defensive only
            raise InvalidParameter(f"Unknown Agent-IDP action: {action}")
