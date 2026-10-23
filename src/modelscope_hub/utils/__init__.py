"""Utility helpers shared across the ModelScope Hub SDK."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from .file_utils import compute_hash, ensure_dir, get_cache_dir, get_file_size
from .format import format_size, format_timesince, tabulate
from .logger import get_logger
from .media import encode_media_to_base64
from .patterns import extract_common_prefix, normalize_patterns
from .time_utils import parse_timestamp

if TYPE_CHECKING:
    from ..config import HubConfig

# Keep retries and transports in one session when credential storage is unavailable.
_FALLBACK_SESSION_ID = uuid.uuid4().hex

__all__ = [
    "append_user_agent",
    "build_user_agent",
    "compute_hash",
    "encode_media_to_base64",
    "ensure_dir",
    "extract_common_prefix",
    "format_size",
    "format_timesince",
    "get_cache_dir",
    "get_file_size",
    "get_logger",
    "normalize_patterns",
    "parse_timestamp",
    "tabulate",
]


def build_user_agent(
    session_id: str | None = None,
    extra: dict | str | None = None,
    *,
    config: HubConfig | None = None,
) -> str:
    """Build the standard ModelScope Hub SDK User-Agent string.

    Parameters
    ----------
    session_id : str, optional
        Stable session UUID (from ``HubConfig.get_session_id()``).
        Falls back to a random UUID if not provided.
    extra : dict, str or None
        Additional key/value pairs or free-form string appended to the UA.
    config : HubConfig, optional
        Resolve the persisted session_id from this config; use a process-stable
        ID if storage is unavailable. An explicit session_id avoids storage access.
    """
    import os
    import platform

    from .. import __version__
    from ..errors import CacheError

    if config is not None and session_id is None:
        try:
            session_id = config.get_session_id()
        except (CacheError, OSError):
            session_id = _FALLBACK_SESSION_ID

    env = os.environ.get("MODELSCOPE_CLOUD_ENVIRONMENT", "custom")
    user_name = os.environ.get("MODELSCOPE_CLOUD_USERNAME", "unknown")
    sid = session_id or uuid.uuid4().hex

    ua = (
        f"modelscope_hub/{__version__}; python/{platform.python_version()}; "
        f"session_id/{sid}; platform/{platform.platform()}; "
        f"processor/{platform.processor()}; env/{env}; user/{user_name}"
    )
    return append_user_agent(ua, extra)


def append_user_agent(base: str, extra: dict | str | None = None) -> str:
    """Append custom UA information without removing the SDK prefix or duplicating fields."""
    if isinstance(extra, dict):
        extra = "; ".join(f"{k}/{v}" for k, v in extra.items())
    parts = base.split("; ")
    if extra:
        for part in extra.split("; "):
            if part and part not in parts:
                parts.append(part)
    return "; ".join(parts)
