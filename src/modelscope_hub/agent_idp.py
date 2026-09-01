"""Local Ed25519/JWK helpers for the Agent-IDP OpenAPI surface.

This module deliberately owns no HTTP transport and never persists a key unless
an application explicitly calls :func:`write_private_jwk`. Agent repository
transfer lives in :mod:`modelscope_hub.agent` and is unrelated to Agent-IDP
identities, signing keys, OIDC discovery, or JWT issuance.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .errors import InvalidParameter
from .types import AgentJWK, TokenSignPayload

__all__ = [
    "generate_agent_key_pair",
    "load_private_jwk",
    "public_jwk_from_private",
    "sign_agent_token_request",
    "write_private_jwk",
]


_JWK_KEY_TYPE = "OKP"
_JWK_CURVE = "Ed25519"


def _encode_base64url(value: bytes) -> str:
    """Encode bytes as unpadded base64url, the JWK wire representation."""
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode_base64url(value: object, field: str) -> bytes:
    """Decode an unpadded base64url JWK member without accepting aliases."""
    if not isinstance(value, str) or not value:
        raise InvalidParameter(f"Agent private key field {field!r} must be a non-empty base64url string.")
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, UnicodeEncodeError) as exc:
        raise InvalidParameter(f"Agent private key field {field!r} is not valid base64url.") from exc
    if _encode_base64url(decoded) != value:
        raise InvalidParameter(f"Agent private key field {field!r} must be unpadded base64url.")
    return decoded


def _normalise_private_jwk(value: AgentJWK | Mapping[str, Any]) -> AgentJWK:
    """Validate an Ed25519 private JWK and return a safe structured copy."""
    if isinstance(value, AgentJWK):
        key = value
    elif isinstance(value, Mapping):
        key = AgentJWK.from_dict(value)
    else:
        raise InvalidParameter("Agent private key must be an Ed25519 JWK mapping.")

    if key.kty != _JWK_KEY_TYPE or key.crv != _JWK_CURVE:
        raise InvalidParameter("Agent private key must use kty='OKP' and crv='Ed25519'.")
    if not key.kid:
        raise InvalidParameter("Agent private key must contain a non-empty 'kid'.")
    public_bytes = _decode_base64url(key.x, "x")
    private_bytes = _decode_base64url(key.d, "d")
    if len(public_bytes) != 32 or len(private_bytes) != 32:
        raise InvalidParameter("Agent Ed25519 JWK public and private values must each be 32 bytes.")
    derived_public = (
        Ed25519PrivateKey.from_private_bytes(private_bytes)
        .public_key()
        .public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    )
    if derived_public != public_bytes:
        raise InvalidParameter("Agent private JWK public and private values do not form an Ed25519 key pair.")
    return AgentJWK(
        kty=_JWK_KEY_TYPE,
        crv=_JWK_CURVE,
        x=key.x,
        kid=key.kid,
        alg=key.alg,
        use=key.use,
        d=key.d,
    )


def generate_agent_key_pair(kid: str | None = None) -> tuple[AgentJWK, AgentJWK]:
    """Generate a local Ed25519 key pair as ``(private_jwk, public_jwk)``.

    The caller chooses whether and where to persist the private JWK. This helper
    never writes files or returns the private material in a public JWK.
    """
    if kid is not None and not kid.strip():
        raise InvalidParameter("Agent key id 'kid' must not be empty.")
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    resolved_kid = kid.strip() if kid is not None else secrets.token_urlsafe(16)
    private_jwk = AgentJWK(
        kty=_JWK_KEY_TYPE,
        crv=_JWK_CURVE,
        x=_encode_base64url(public_bytes),
        kid=resolved_kid,
        alg="EdDSA",
        use="sig",
        d=_encode_base64url(private_bytes),
    )
    return private_jwk, public_jwk_from_private(private_jwk)


def public_jwk_from_private(private_jwk: AgentJWK | Mapping[str, Any]) -> AgentJWK:
    """Return the upload-safe public JWK corresponding to a private JWK."""
    key = _normalise_private_jwk(private_jwk)
    return AgentJWK(kty=key.kty, crv=key.crv, x=key.x, kid=key.kid, alg=key.alg, use=key.use)


def write_private_jwk(path: str | Path, private_jwk: AgentJWK | Mapping[str, Any], *, overwrite: bool = False) -> Path:
    """Write a validated private JWK with owner-only (``0600``) permissions.

    Existing files are never replaced unless *overwrite* is explicit. Symlinks
    are rejected to keep a CLI invocation from overwriting an unexpected file.
    """
    key = _normalise_private_jwk(private_jwk)
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise InvalidParameter("Refusing to write an Agent private key through a symbolic link.")
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if overwrite else os.O_EXCL)
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError as exc:
        raise InvalidParameter(f"Agent private key file already exists: {target}. Pass --force to replace it.") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(key.to_dict(include_private=True), output, sort_keys=True)
            output.write("\n")
        os.chmod(target, 0o600)
    except BaseException:
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target


def load_private_jwk(path: str | Path) -> AgentJWK:
    """Load and strictly validate an owner-only Ed25519 private JWK file."""
    target = Path(path).expanduser()
    try:
        mode = target.stat().st_mode
    except OSError as exc:
        raise InvalidParameter(f"Unable to read Agent private key file: {target}.") from exc
    if target.is_symlink() or not stat.S_ISREG(mode):
        raise InvalidParameter("Agent private key path must be a regular, non-symbolic-link file.")
    if os.name == "posix" and mode & 0o077:
        raise InvalidParameter("Agent private key file must not be readable by group or other users (mode 0600).")
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidParameter(f"Agent private key file is not valid JSON: {target}.") from exc
    if not isinstance(raw, Mapping):
        raise InvalidParameter("Agent private key file must contain a JSON object.")
    return _normalise_private_jwk(raw)


def sign_agent_token_request(
    private_jwk: AgentJWK | Mapping[str, Any],
    *,
    agent_id: str,
    audience: str,
    timestamp: int,
) -> TokenSignPayload:
    """Build the signed body required by anonymous ``POST /agent_id/token``."""
    if not isinstance(agent_id, str) or not agent_id:
        raise InvalidParameter("agent_id must be a non-empty string.")
    if not isinstance(audience, str) or not audience:
        raise InvalidParameter("audience must be a non-empty string.")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp <= 0:
        raise InvalidParameter("timestamp must be a positive Unix timestamp in seconds.")
    key = _normalise_private_jwk(private_jwk)
    message = f"{agent_id}|{key.kid}|{audience}|{timestamp}".encode("ascii")
    private_bytes = _decode_base64url(key.d, "d")
    signature = Ed25519PrivateKey.from_private_bytes(private_bytes).sign(message)
    return {
        "agent_id": agent_id,
        "kid": key.kid,
        "audience": audience,
        "timestamp": timestamp,
        "signature": _encode_base64url(signature),
    }
