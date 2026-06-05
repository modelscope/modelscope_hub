"""Utility helpers shared across the ModelScope Hub SDK."""

from __future__ import annotations

from .file_utils import compute_hash, ensure_dir, get_cache_dir, get_file_size
from .logger import get_logger

__all__ = [
    "build_user_agent",
    "compute_hash",
    "ensure_dir",
    "get_cache_dir",
    "get_file_size",
    "get_logger",
]


def build_user_agent(
    session_id: str | None = None,
    extra: dict | str | None = None,
) -> str:
    """Build the standard ModelScope Hub SDK User-Agent string.

    Parameters
    ----------
    session_id : str, optional
        Stable session UUID (from ``HubConfig.get_session_id()``).
        Falls back to a random UUID if not provided.
    extra : dict, str or None
        Additional key/value pairs or free-form string appended to the UA.
    """
    import os
    import platform
    import uuid

    from .. import __version__

    env = os.environ.get("MODELSCOPE_CLOUD_ENVIRONMENT", "custom")
    user_name = os.environ.get("MODELSCOPE_CLOUD_USERNAME", "unknown")
    sid = session_id or uuid.uuid4().hex

    ua = (
        f"modelscope_hub/{__version__}; python/{platform.python_version()}; "
        f"session_id/{sid}; platform/{platform.platform()}; "
        f"processor/{platform.processor()}; env/{env}; user/{user_name}"
    )
    if isinstance(extra, dict):
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in extra.items())
    elif isinstance(extra, str):
        ua += "; " + extra
    return ua
