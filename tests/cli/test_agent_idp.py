"""CLI tests for the Agent-IDP command, with no network access."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from modelscope_hub.agent_idp import generate_agent_key_pair, write_private_jwk
from modelscope_hub.cli.agent_idp import AgentIdpCommand
from modelscope_hub.errors import InvalidParameter
from modelscope_hub.types import AgentIdentity, AgentToken


def _execute(parser, arguments: list[str]) -> None:
    args = parser.parse_args(["agent-idp", *arguments])
    assert args._command is AgentIdpCommand
    AgentIdpCommand(args).execute()


def test_parser_registers_agent_idp(parser):
    args = parser.parse_args(["agent-idp", "configuration"])
    assert args.agent_idp_action == "configuration"
    assert args._command is AgentIdpCommand


def test_keygen_prints_only_public_jwk_and_writes_private_file(parser, tmp_path, capsys):
    path = tmp_path / "agent.jwk"
    _execute(parser, ["keygen", "--private-key-out", str(path), "--kid", "key-1"])
    public_jwk = json.loads(capsys.readouterr().out)
    private_jwk = json.loads(path.read_text())
    assert public_jwk["kid"] == "key-1"
    assert "d" not in public_jwk
    assert private_jwk["d"]


def test_create_reads_public_key_without_private_material(parser, tmp_path):
    public_path = tmp_path / "public.jwk"
    public_path.write_text(json.dumps({"kty": "OKP", "crv": "Ed25519", "x": "x", "kid": "key-1"}))
    api = MagicMock()
    api.create_agent_identity.return_value = AgentIdentity(agent_id="agent-1", agent_name="builder")
    with patch("modelscope_hub.cli.agent_idp.make_api", return_value=api):
        _execute(parser, ["create", "--agent-name", "builder", "--public-jwk-file", str(public_path)])
    payload = api.create_agent_identity.call_args.args[0]
    assert payload["public_key"] == {"kty": "OKP", "crv": "Ed25519", "x": "x", "kid": "key-1"}


def test_delete_requires_explicit_confirmation(parser):
    with pytest.raises(InvalidParameter, match="--yes"):
        _execute(parser, ["delete", "agent-1"])


def test_issue_token_stdout_is_only_the_credential(parser, tmp_path, capsys):
    private_jwk, _ = generate_agent_key_pair()
    path = tmp_path / "agent.jwk"
    write_private_jwk(path, private_jwk)
    api = MagicMock()
    api.issue_agent_token_with_private_key.return_value = AgentToken(access_token="issued.jwt")
    with patch("modelscope_hub.cli.agent_idp.make_api", return_value=api):
        _execute(
            parser,
            ["issue-token", "--agent-id", "agent-1", "--audience", "hub", "--private-key-file", str(path)],
        )
    assert capsys.readouterr().out == "issued.jwt\n"


def test_public_oidc_commands_can_run_without_token(parser, capsys):
    api = MagicMock()
    api.get_agent_id_configuration.return_value = MagicMock(
        issuer="https://issuer",
        token_endpoint="https://issuer/token",
        jwks_uri=None,
        registration_endpoint=None,
        activity_endpoint=None,
        id_token_signing_alg_values_supported=None,
    )
    with patch("modelscope_hub.cli.agent_idp.make_api", return_value=api):
        _execute(parser, ["configuration"])
    assert "https://issuer" in capsys.readouterr().out
    api.get_agent_id_configuration.assert_called_once_with()
