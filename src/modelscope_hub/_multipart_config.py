"""Private settings and URL binding for resumable uploads."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit

from . import constants as c
from .errors import InvalidParameter

_PROTOCOL_KEYS = frozenset({"resumable_multipart", "action", "part_size", "part_number"})


def canonical_origin(url: str) -> str:
    try:
        parsed = urlsplit(url)
        if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username is not None:
            raise ValueError
        port = parsed.port
        host = parsed.hostname.lower()
        if ":" in host:
            host = f"[{host}]"
        return f"https://{host}" + (f":{port}" if port and port != 443 else "")
    except ValueError as exc:
        raise InvalidParameter("Multipart targets require a valid HTTPS origin without credentials.") from exc


def _bounded(value: int, low: int, high: int, default: int, name: str) -> int:
    if not low <= value <= high:
        warnings.warn(f"{name} is outside [{low}, {high}]; using {default}.", UserWarning, stacklevel=3)
        return default
    return value


@dataclass(frozen=True)
class MultipartSettings:
    mode: str
    threshold: int
    part_size: int
    origins: frozenset[str]
    requests: int
    part_workers: int
    hash_workers: int
    chunk_size: int
    response_bytes: int
    lock_timeout: int
    recovery_rounds: int
    recovery_actions: int
    reinitializations: int
    token_refreshes: int
    retry_after: int

    @classmethod
    def load(cls, endpoint: str) -> MultipartSettings:
        mode = c.UPLOAD_MULTIPART_MODE.strip().lower()
        if mode not in {"off", "auto", "required"}:
            raise InvalidParameter("MODELSCOPE_UPLOAD_MULTIPART_MODE must be off, auto or required.")
        origins: set[str] = set()
        part_size = 0
        if mode != "off":
            origins.add(canonical_origin(endpoint))
            for value in c.UPLOAD_MULTIPART_TRUSTED_ORIGINS.split(","):
                if not value.strip():
                    continue
                raw = value.strip()
                parsed = urlsplit(raw)
                if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
                    raise InvalidParameter("MODELSCOPE_UPLOAD_MULTIPART_TRUSTED_ORIGINS accepts origins only.")
                origins.add(canonical_origin(raw))
            try:
                part_size = c._parse_byte_size(str(c.UPLOAD_MULTIPART_PART_SIZE))
            except (ValueError, TypeError) as exc:
                raise InvalidParameter("MODELSCOPE_UPLOAD_MULTIPART_PART_SIZE is invalid.") from exc
            if part_size != 0 and (not 100 * 1024 <= part_size <= 64 * 1024**2 or part_size % 64):
                raise InvalidParameter("Multipart part size must be 100 KiB..64 MiB and divisible by 64, or zero.")
        return cls(
            mode,
            c.UPLOAD_MULTIPART_THRESHOLD_BYTES,
            part_size,
            frozenset(origins),
            c.UPLOAD_MULTIPART_MAX_CONCURRENT_REQUESTS,
            c.UPLOAD_MULTIPART_PART_WORKERS,
            c.UPLOAD_MULTIPART_HASH_WORKERS,
            _bounded(
                c.UPLOAD_MULTIPART_IO_CHUNK_SIZE_BYTES,
                64 * 1024,
                8 * 1024**2,
                1024**2,
                "MODELSCOPE_UPLOAD_MULTIPART_IO_CHUNK_SIZE",
            ),
            _bounded(
                c.UPLOAD_MULTIPART_MAX_RESPONSE_BYTES,
                512 * 1024,
                16 * 1024**2,
                4 * 1024**2,
                "MODELSCOPE_UPLOAD_MULTIPART_MAX_RESPONSE_BYTES",
            ),
            c.UPLOAD_MULTIPART_LOCK_TIMEOUT_SECONDS,
            c.UPLOAD_MULTIPART_MAX_RECOVERY_ROUNDS,
            c.UPLOAD_MULTIPART_MAX_RECOVERY_ACTIONS,
            c.UPLOAD_MULTIPART_MAX_REINITIALIZATIONS,
            c.UPLOAD_MULTIPART_MAX_TOKEN_REFRESHES,
            c.UPLOAD_MULTIPART_MAX_RETRY_AFTER_SECONDS,
        )

    def candidate(self, repo_type: str, size: int, data: object) -> bool:
        return (
            self.mode != "off"
            and repo_type in {"model", "dataset"}
            and size > 0
            and size >= self.threshold
            and isinstance(data, (str, Path, bytes))
        )

    def validate_target(self, url: str, repo_id: str, repo_type: str, sha256: str, size: int) -> str:
        parts = repo_id.split("/")
        if len(parts) != 2 or any(not p or p in {".", ".."} or "\\" in p for p in parts):
            raise InvalidParameter("Multipart repository ID must be owner/name.")
        if not re.fullmatch(r"[0-9a-f]{64}", sha256) or size <= 0:
            raise InvalidParameter("Invalid multipart content binding.")
        parsed = urlsplit(url)
        if canonical_origin(url) not in self.origins or parsed.fragment:
            raise InvalidParameter("Multipart target is outside the trusted origins.")
        if re.search(r"%(?:2f|5c)", parsed.path, re.IGNORECASE):
            raise InvalidParameter("Encoded separators are not allowed in multipart targets.")
        expected = f"/api/v1/repos/{repo_type}s/{'/'.join(parts)}/blobs/{sha256}/{size}"
        if unquote(parsed.path) != expected:
            raise InvalidParameter("Multipart target does not match the current file and repository.")
        if any(key in _PROTOCOL_KEYS for key, _ in parse_qsl(parsed.query, keep_blank_values=True)):
            raise InvalidParameter("Multipart target already contains protocol parameters.")
        return url


def action_url(url: str, action: str, **params: int) -> str:
    query: dict[str, str | int] = {"resumable_multipart": "true", "action": action}
    query.update(params)
    return url + ("&" if urlsplit(url).query else "?") + urlencode(query, quote_via=quote)
