"""Project-wide constants and configuration knobs.

All runtime tunables expose an environment-variable override so that the SDK
can be reconfigured without code changes. This keeps the library friendly for
both production deployments and ad-hoc experimentation.
"""

from __future__ import annotations

import os
from enum import Enum, IntEnum


# ---------------------------------------------------------------------------
# StrEnum compatibility shim (Python 3.10 lacks :class:`enum.StrEnum`).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised implicitly by the import path
    from enum import StrEnum  # type: ignore[attr-defined]
except ImportError:  # Python 3.10
    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """Minimal backport of :class:`enum.StrEnum` for Python 3.10."""

        def __str__(self) -> str:  # noqa: D401 - mirror stdlib behaviour
            return str(self.value)


# ---------------------------------------------------------------------------
# Domain enums
# ---------------------------------------------------------------------------
class RepoType(StrEnum):
    """Kinds of repositories hosted on ModelScope Hub."""

    MODEL = "model"
    DATASET = "dataset"
    STUDIO = "studio"
    SKILL = "skill"
    MCP = "mcp"


class Visibility(IntEnum):
    """Repository visibility levels.

    The integer values mirror the encoding used by the ModelScope Hub API
    (1 = public, 3 = internal, 5 = private).
    """

    PUBLIC = 1
    INTERNAL = 3
    PRIVATE = 5

    @property
    def label(self) -> str:
        """Human readable label."""
        return self.name.lower()

    @classmethod
    def from_label(cls, label: str) -> "Visibility":
        """Resolve a visibility from its lowercase label."""
        try:
            return cls[label.upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown visibility label: {label!r}") from exc


class License(StrEnum):
    """Common open-source licenses used on ModelScope Hub."""

    APACHE_2_0 = "Apache-2.0"
    MIT = "MIT"
    BSD_2_CLAUSE = "BSD-2-Clause"
    BSD_3_CLAUSE = "BSD-3-Clause"
    GPL_2_0 = "GPL-2.0"
    GPL_3_0 = "GPL-3.0"
    LGPL_2_1 = "LGPL-2.1"
    LGPL_3_0 = "LGPL-3.0"
    MPL_2_0 = "MPL-2.0"
    CC_BY_4_0 = "CC-BY-4.0"
    CC_BY_SA_4_0 = "CC-BY-SA-4.0"
    CC_BY_NC_4_0 = "CC-BY-NC-4.0"
    CC0_1_0 = "CC0-1.0"
    UNLICENSE = "Unlicense"
    OTHER = "Other"


# ---------------------------------------------------------------------------
# Endpoint configuration
# ---------------------------------------------------------------------------
DEFAULT_ENDPOINT: str = "https://modelscope.cn"
OPENAPI_PREFIX: str = "/openapi/v1"
LEGACY_API_PREFIX: str = "/api/v1"


# ---------------------------------------------------------------------------
# Helpers for environment-driven overrides
# ---------------------------------------------------------------------------
def _env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


# ---------------------------------------------------------------------------
# Network / IO tunables
# ---------------------------------------------------------------------------
API_TIMEOUT: int = _env_int("API_TIMEOUT", 60)
"""Default HTTP timeout (seconds) for API calls."""

API_MAX_RETRIES: int = _env_int("API_MAX_RETRIES", 5)
"""Maximum retry attempts for transient API failures."""

DOWNLOAD_CHUNK_SIZE: int = _env_int("DOWNLOAD_CHUNK_SIZE", 1024 * 1024)
"""Streaming chunk size for downloads (bytes). Defaults to 1 MiB."""

UPLOAD_LFS_THRESHOLD: int = _env_int("UPLOAD_LFS_THRESHOLD", 5 * 1024 * 1024)
"""Files larger than this threshold (bytes) must use the LFS upload path.

Default of 5 MiB aligns with the OpenAPI direct file-upload limit.
"""


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
DEFAULT_CACHE_DIR_NAME: str = "modelscope"
TOKEN_FILE_NAME: str = "token"
CONFIG_DIR_NAME: str = ".modelscope"


__all__ = [
    "API_MAX_RETRIES",
    "API_TIMEOUT",
    "CONFIG_DIR_NAME",
    "DEFAULT_CACHE_DIR_NAME",
    "DEFAULT_ENDPOINT",
    "DOWNLOAD_CHUNK_SIZE",
    "LEGACY_API_PREFIX",
    "License",
    "OPENAPI_PREFIX",
    "RepoType",
    "StrEnum",
    "TOKEN_FILE_NAME",
    "UPLOAD_LFS_THRESHOLD",
    "Visibility",
]
