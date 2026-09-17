"""Security-sensitive local Ed25519/JWK tests for Agent-IDP."""

from __future__ import annotations

import base64
import os
import stat

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from modelscope_hub.agent_idp import (
    generate_agent_key_pair,
    load_private_jwk,
    public_jwk_from_private,
    sign_agent_token_request,
    write_private_jwk,
)
from modelscope_hub.errors import InvalidParameter


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def test_generated_jwks_round_trip_and_sign():
    private_jwk, public_jwk = generate_agent_key_pair("key-1")
    assert private_jwk.d
    assert "d" not in public_jwk.to_dict()
    assert private_jwk.x == public_jwk.x
    assert "=" not in private_jwk.x

    signed = sign_agent_token_request(private_jwk, agent_id="agent-1", audience="hub", timestamp=100)
    message = b"agent-1|key-1|hub|100"
    Ed25519PublicKey.from_public_bytes(_decode(public_jwk.x)).verify(_decode(signed["signature"]), message)
    with pytest.raises(InvalidSignature):
        Ed25519PublicKey.from_public_bytes(_decode(public_jwk.x)).verify(_decode(signed["signature"]), b"wrong")


def test_private_file_is_owner_only_and_not_overwritten(tmp_path):
    private_jwk, _ = generate_agent_key_pair()
    destination = tmp_path / "agent.jwk"
    write_private_jwk(destination, private_jwk)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    loaded = load_private_jwk(destination)
    assert loaded.to_dict(include_private=True) == private_jwk.to_dict(include_private=True)
    with pytest.raises(InvalidParameter, match="already exists"):
        write_private_jwk(destination, private_jwk)


def test_private_file_rejects_insecure_permissions_on_posix(tmp_path):
    if os.name != "posix":
        pytest.skip("POSIX permission enforcement only")
    private_jwk, _ = generate_agent_key_pair()
    destination = tmp_path / "agent.jwk"
    write_private_jwk(destination, private_jwk)
    destination.chmod(0o644)
    with pytest.raises(InvalidParameter, match="0600"):
        load_private_jwk(destination)


def test_mismatched_key_material_is_rejected_without_leaking_private_value():
    private_jwk, _ = generate_agent_key_pair()
    invalid = private_jwk.to_dict(include_private=True)
    invalid["x"] = "A" * len(invalid["x"])
    with pytest.raises(InvalidParameter) as raised:
        public_jwk_from_private(invalid)
    assert private_jwk.d not in str(raised.value)


def test_signing_rejects_invalid_timestamp():
    private_jwk, _ = generate_agent_key_pair()
    with pytest.raises(InvalidParameter, match="timestamp"):
        sign_agent_token_request(private_jwk, agent_id="agent-1", audience="hub", timestamp=0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("audience", "高德地图"),
        ("agent_id", "agent-高德"),
        ("kid", "kid-高德"),
    ],
)
def test_signing_rejects_non_ascii_canonical_component_without_leaking_encoder_error(field, value):
    private_jwk, _ = generate_agent_key_pair(value if field == "kid" else "key-1")
    kwargs = {"agent_id": "agent-1", "audience": "hub", "timestamp": 100}
    if field != "kid":
        kwargs[field] = value
    with pytest.raises(InvalidParameter) as raised:
        sign_agent_token_request(private_jwk, **kwargs)
    error = raised.value
    assert error.error_code == "E3021"
    assert f"{field} must contain only ASCII characters" in str(error)
    assert "UnicodeEncodeError" not in str(error)
    assert error.__cause__ is None


@pytest.mark.parametrize("field", ["agent_id", "audience", "kid"])
def test_signing_rejects_canonical_message_delimiter(field):
    private_jwk, _ = generate_agent_key_pair("key|1" if field == "kid" else "key-1")
    kwargs = {"agent_id": "agent-1", "audience": "hub", "timestamp": 100}
    if field != "kid":
        kwargs[field] = "value|break"
    with pytest.raises(InvalidParameter, match="must not contain '\\|'") as raised:
        sign_agent_token_request(private_jwk, **kwargs)
    assert raised.value.error_code == "E3021"
