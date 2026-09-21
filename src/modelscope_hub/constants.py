"""Project-wide constants and configuration knobs.

All runtime tunables expose an environment-variable override so that the SDK
can be reconfigured without code changes. This keeps the library friendly for
both production deployments and ad-hoc experimentation.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path

# ---------------------------------------------------------------------------
# StrEnum compatibility shim (Python 3.10 lacks :class:`enum.StrEnum`).
# ``sys.version_info`` branching (instead of try/except) lets type checkers
# resolve the correct definition statically.
# ---------------------------------------------------------------------------
if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        """Minimal backport of :class:`enum.StrEnum` for Python 3.10."""

        def __str__(self) -> str:  # noqa: D401 - mirror stdlib behaviour
            return str(self.value)


# ---------------------------------------------------------------------------
# Centralised environment-variable registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EnvVar:
    """Metadata for one configurable environment variable."""

    name: str
    default: str
    description: str
    category: str  # Core, Network, Download, Upload, Logging, Deprecated
    deprecated_names: tuple[str, ...] = ()


ENV_REGISTRY: list[EnvVar] = []

CATEGORY_ORDER: tuple[str, ...] = (
    "Core",
    "Network",
    "Download",
    "Upload",
    "Logging",
    "Deprecated",
)


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
    (1 = private, 3 = internal, 5 = public).
    """

    PRIVATE = 1
    INTERNAL = 3
    PUBLIC = 5

    @property
    def label(self) -> str:
        """Human readable label."""
        return self.name.lower()

    @classmethod
    def from_label(cls, label: str) -> Visibility:
        """Resolve a visibility from its lowercase label or numeric string.

        Supports both label strings ('private', 'internal', 'public') and
        numeric strings ('1', '3', '5') for backward compatibility.
        """
        # Support numeric strings for backward compatibility: '1' → PRIVATE, '3' → INTERNAL, '5' → PUBLIC
        if isinstance(label, str) and label.isdigit():
            numeric = int(label)
            for member in cls:
                if member.value == numeric:
                    return member
            raise ValueError(f"Unknown visibility label: {label!r}")
        # Standard label lookup: 'private' → PRIVATE
        try:
            return cls[label.upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown visibility label: {label!r}") from exc


class StudioVisibility(StrEnum):
    """Visibility levels a Studio space can be published under.

    Deliberately separate from :class:`Visibility`: models and datasets encode
    visibility as the integers 1/3/5, whereas the Studio endpoints take a string
    enum and offer a third state the integer encoding cannot express.

    * ``public`` -- both the code and the running app are public.
    * ``protected`` -- the app is public, the code repository is not.
    * ``private`` -- neither is public.
    """

    PUBLIC = "public"
    PROTECTED = "protected"
    PRIVATE = "private"

    @classmethod
    def parse(cls, value: object) -> StudioVisibility | None:
        """Return the matching member, or ``None`` when *value* is not one.

        Returning ``None`` rather than raising lets callers fall back to the
        integer :class:`Visibility` encoding for inputs this enum does not own.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                return None
        return None


class TokenScope(StrEnum):
    """Permission tiers a ModelScope API token can be issued with.

    The Hub grants tokens one of three levels. The OpenAPI specification does
    not model them -- ``securitySchemes`` declares a bare bearer scheme with no
    scopes, and ``GET /users/me`` does not report the caller's level -- so the
    SDK cannot know a token's tier up front and never pre-validates against it.
    These values are used only to annotate what an operation needs, so that a
    403 can name the missing permission instead of leaving the user guessing.
    """

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


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
# Repo-type string aliases and shared repo defaults.
#
# The modern surface models repo kinds as :class:`RepoType`; these string
# aliases and the dataset-revision default are the historical names the
# modelscope SDK consumes. Defined here so ``modelscope_hub.constants`` is the
# single source of truth (``compat.constants`` re-exports them).
# ---------------------------------------------------------------------------
REPO_TYPE_MODEL: str = RepoType.MODEL.value
REPO_TYPE_DATASET: str = RepoType.DATASET.value
REPO_TYPE_STUDIO: str = RepoType.STUDIO.value
REPO_TYPE_SUPPORT: list[str] = [REPO_TYPE_MODEL, REPO_TYPE_DATASET, REPO_TYPE_STUDIO]
DEFAULT_DATASET_REVISION: str = "master"

# ---------------------------------------------------------------------------
# Legacy modelscope domain / endpoint / filesystem constants.
#
# Historical names and shapes (``www.`` domains, the ``damo`` group, a ``Path``
# credentials location) the modelscope SDK consumes. The modern canonical
# endpoint stays :data:`DEFAULT_ENDPOINT` (no ``www``); these coexist for
# backward compatibility and ``compat.constants`` re-exports them.
# ---------------------------------------------------------------------------
MODEL_ID_SEPARATOR: str = "/"
DEFAULT_MODELSCOPE_GROUP: str = "damo"
DEFAULT_MODELSCOPE_DOMAIN: str = "www.modelscope.cn"
DEFAULT_MODELSCOPE_INTL_DOMAIN: str = "www.modelscope.ai"
DEFAULT_MODELSCOPE_DATA_ENDPOINT: str = "https://" + DEFAULT_MODELSCOPE_DOMAIN
DEFAULT_MODELSCOPE_INTL_DATA_ENDPOINT: str = "https://" + DEFAULT_MODELSCOPE_INTL_DOMAIN
DEFAULT_SKILLS_DIR: str = os.path.join(os.path.expanduser("~"), ".agents", "skills")
DEFAULT_CREDENTIALS_PATH: Path = Path.home().joinpath(".modelscope", "credentials")


# ---------------------------------------------------------------------------
# Helpers for environment-driven overrides (auto-registering)
# ---------------------------------------------------------------------------
_REGISTERED_NAMES: set[str] = set()
_DEPRECATED_LOOKUP: dict[str, tuple[str, ...]] = {}


def _warn_deprecated_env(
    old: str,
    name: str,
    *,
    expects_mb: bool = False,
    expects_bytes: bool = False,
    stacklevel: int = 3,
) -> None:
    """Warn that a legacy environment variable remains temporarily supported."""
    message = (
        f"Environment variable {old!r} is deprecated and will be removed in a future version. Use {name!r} instead."
    )
    if expects_mb:
        message += f" {name!r} expects a value in MB."
    if expects_bytes:
        message += f" {name!r} expects bytes, or a value with a unit suffix such as '32KiB'."
    warnings.warn(message, FutureWarning, stacklevel=stacklevel)


def _format_bytes(num_bytes: int) -> str:
    """Render a byte count using the largest binary unit that stays exact."""
    for unit, scale in (("GiB", 1024**3), ("MiB", 1024**2), ("KiB", 1024)):
        if num_bytes and num_bytes % scale == 0:
            return f"{num_bytes // scale}{unit}"
    return f"{num_bytes}"


def _env(name: str, *deprecated_names: str) -> str | None:
    """Read an env var, falling back to deprecated names with a warning."""
    value = os.environ.get(name)
    if value is not None:
        return value
    for old in deprecated_names:
        value = os.environ.get(old)
        if value is not None:
            _warn_deprecated_env(old, name, stacklevel=4)
            return value
    return None


def _warn_invalid_env(name: str, raw: str, reason: str, fallback: object) -> None:
    """Warn that an environment value was rejected, naming the value used instead.

    Silently falling back to the default made misconfiguration invisible: a
    pipeline could export a tuning value, observe none of its effect, and have
    no signal to look at. Every rejected value now says so.
    """
    warnings.warn(
        f"Environment variable {name}={raw!r} is invalid ({reason}); using {fallback!r} instead.",
        UserWarning,
        stacklevel=3,
    )


_BYTE_UNITS: dict[str, int] = {
    "": 1,
    "B": 1,
    "K": 1024,
    "KB": 1000,
    "KIB": 1024,
    "M": 1024**2,
    "MB": 1000**2,
    "MIB": 1024**2,
    "G": 1024**3,
    "GB": 1000**3,
    "GIB": 1024**3,
}


def _parse_byte_size(raw: str, *, bare_unit: int = 1) -> int:
    """Parse a size string into bytes, accepting an optional unit suffix.

    ``bare_unit`` scales a value given without a suffix, which is what lets a
    deprecated ``*_MB`` alias keep its megabyte meaning while the canonical
    name treats a bare number as bytes. Raises :class:`ValueError` so callers
    decide between warning and propagating.
    """
    text = raw.strip()
    if not text:
        raise ValueError("empty value")
    digits = text
    suffix = ""
    while digits and not (digits[-1].isdigit() or digits[-1] == "."):
        suffix = digits[-1] + suffix
        digits = digits[:-1]
    digits = digits.strip()
    suffix = suffix.strip().upper()
    if not digits:
        raise ValueError("no numeric part")
    if suffix not in _BYTE_UNITS:
        raise ValueError(f"unknown size unit {suffix!r}")
    number = float(digits)
    if number != int(number):
        raise ValueError("fractional byte counts are not supported")
    unit = bare_unit if suffix == "" else _BYTE_UNITS[suffix]
    return int(number) * unit


def _env_int(
    name: str,
    default: int,
    description: str = "",
    category: str = "",
    *deprecated_names: str,
    allow_zero: bool = False,
) -> int:
    """Read a positive integer from the environment and register it."""
    all_deprecated = deprecated_names or _DEPRECATED_LOOKUP.get(name, ())
    if description and category:
        _env_register(name, str(default), description, category, deprecated_names=all_deprecated)
    raw = _env(name, *all_deprecated)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        _warn_invalid_env(name, raw, "not an integer", default)
        return default
    if value < 0 or (value == 0 and not allow_zero):
        reason = "must not be negative" if allow_zero else "must be a positive integer"
        _warn_invalid_env(name, raw, reason, default)
        return default
    return value


def _env_int_mb(
    name: str,
    default_mb: int,
    description: str = "",
    category: str = "",
    *deprecated_byte_names: str,
) -> int:
    """Read a size env var (new name in MB, deprecated names in bytes). Returns bytes.

    Handles migration from byte-based deprecated env vars to MB-based new names.
    If a deprecated byte-based name is set, the value is used directly (already bytes).
    If the new name is set, the value is treated as MB and converted to bytes.
    """
    all_deprecated = deprecated_byte_names or _DEPRECATED_LOOKUP.get(name, ())
    if description and category:
        _env_register(name, str(default_mb), description, category, deprecated_names=all_deprecated)
    # Check the new name first (value in MB)
    raw = os.environ.get(name)
    if raw is not None and raw.strip():
        try:
            value = int(raw)
        except ValueError:
            _warn_invalid_env(name, raw, "not an integer", f"{default_mb} MB")
            return default_mb * 1024 * 1024
        if value <= 0:
            _warn_invalid_env(name, raw, "must be a positive integer", f"{default_mb} MB")
            return default_mb * 1024 * 1024
        return value * 1024 * 1024
    # Fall back to deprecated names (value already in bytes)
    for old in all_deprecated:
        raw = os.environ.get(old)
        if raw is not None and raw.strip():
            _warn_deprecated_env(old, name, expects_mb=True, stacklevel=2)
            try:
                value = int(raw)
            except ValueError:
                _warn_invalid_env(old, raw, "not an integer", f"{default_mb} MB")
                return default_mb * 1024 * 1024
            if value <= 0:
                _warn_invalid_env(old, raw, "must be a positive integer", f"{default_mb} MB")
                return default_mb * 1024 * 1024
            return value
    return default_mb * 1024 * 1024


def _env_int_mb_with_deprecated_units(
    name: str,
    default_mb: int,
    description: str,
    category: str,
    *,
    deprecated_mb_names: tuple[str, ...] = (),
    deprecated_byte_names: tuple[str, ...] = (),
) -> int:
    """Read an MB setting while preserving aliases with explicit units."""
    deprecated_names = deprecated_mb_names + deprecated_byte_names
    _env_register(name, str(default_mb), description, category, deprecated_names=deprecated_names)
    default_bytes = default_mb * 1024 * 1024

    raw = os.environ.get(name)
    if raw is not None and raw.strip():
        try:
            value = int(raw)
        except ValueError:
            _warn_invalid_env(name, raw, "not an integer", f"{default_mb} MB")
            return default_bytes
        if value <= 0:
            _warn_invalid_env(name, raw, "must be a positive integer", f"{default_mb} MB")
            return default_bytes
        return value * 1024 * 1024

    for old in deprecated_mb_names:
        raw = os.environ.get(old)
        if raw is not None and raw.strip():
            _warn_deprecated_env(old, name, stacklevel=2)
            try:
                value = int(raw)
            except ValueError:
                _warn_invalid_env(old, raw, "not an integer", f"{default_mb} MB")
                return default_bytes
            if value <= 0:
                _warn_invalid_env(old, raw, "must be a positive integer", f"{default_mb} MB")
                return default_bytes
            return value * 1024 * 1024

    for old in deprecated_byte_names:
        raw = os.environ.get(old)
        if raw is not None and raw.strip():
            _warn_deprecated_env(old, name, expects_mb=True, stacklevel=2)
            try:
                value = int(raw)
            except ValueError:
                _warn_invalid_env(old, raw, "not an integer", f"{default_mb} MB")
                return default_bytes
            if value <= 0:
                _warn_invalid_env(old, raw, "must be a positive integer", f"{default_mb} MB")
                return default_bytes
            return value

    return default_bytes


def _env_bytes(
    name: str,
    default_bytes: int,
    description: str,
    category: str,
    *,
    deprecated_mb_names: tuple[str, ...] = (),
    deprecated_byte_names: tuple[str, ...] = (),
    allow_zero: bool = False,
) -> int:
    """Read a byte-size setting whose canonical name accepts a unit suffix.

    The canonical name treats a bare number as **bytes** and understands the
    suffixes ``B``, ``K``/``KiB``, ``KB``, ``M``/``MiB``, ``MB``, ``G``/``GiB``
    and ``GB`` (binary for the ``iB``/bare-letter forms, decimal for ``KB``/
    ``MB``/``GB``). A megabyte-only knob cannot express thresholds below 1 MB,
    which is exactly the range that matters when deciding whether a small file
    rides inline in a commit or goes to object storage.

    Names in ``deprecated_mb_names`` keep their megabyte meaning for a bare
    number; names in ``deprecated_byte_names`` keep their byte meaning.
    """
    deprecated_names = deprecated_mb_names + deprecated_byte_names
    _env_register(name, _format_bytes(default_bytes), description, category, deprecated_names=deprecated_names)

    def _accept(source: str, raw: str, bare_unit: int) -> int:
        try:
            value = _parse_byte_size(raw, bare_unit=bare_unit)
        except ValueError as exc:
            _warn_invalid_env(source, raw, str(exc), _format_bytes(default_bytes))
            return default_bytes
        if value < 0 or (value == 0 and not allow_zero):
            reason = "must not be negative" if allow_zero else "must be a positive size"
            _warn_invalid_env(source, raw, reason, _format_bytes(default_bytes))
            return default_bytes
        return value

    raw = os.environ.get(name)
    if raw is not None and raw.strip():
        return _accept(name, raw, 1)

    for old in deprecated_mb_names:
        raw = os.environ.get(old)
        if raw is not None and raw.strip():
            _warn_deprecated_env(old, name, expects_bytes=True, stacklevel=2)
            return _accept(old, raw, 1024 * 1024)

    for old in deprecated_byte_names:
        raw = os.environ.get(old)
        if raw is not None and raw.strip():
            _warn_deprecated_env(old, name, expects_bytes=True, stacklevel=2)
            return _accept(old, raw, 1)

    return default_bytes


def _env_csv_frozenset(
    name: str,
    default: str,
    description: str,
    category: str,
    *deprecated_names: str,
) -> frozenset[str]:
    """Read a comma-separated, case-normalised set from the environment."""
    _env_register(name, default, description, category, deprecated_names=deprecated_names)
    raw = _env(name, *deprecated_names) or default
    return frozenset(item.strip().upper() for item in raw.split(",") if item.strip())


def _env_bool(
    name: str,
    default: bool,
    description: str = "",
    category: str = "",
    *deprecated_names: str,
) -> bool:
    """Read a boolean from the environment and register it."""
    if description and category:
        _env_register(name, str(default).lower(), description, category, deprecated_names=deprecated_names)
    all_deprecated = deprecated_names or _DEPRECATED_LOOKUP.get(name, ())
    raw = _env(name, *all_deprecated)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_register(
    name: str,
    default: str,
    description: str,
    category: str,
    *,
    deprecated_names: tuple[str, ...] = (),
) -> None:
    """Register an env var for display only (read logic lives elsewhere)."""
    if category not in CATEGORY_ORDER:
        raise ValueError(f"Unknown env var category {category!r}, must be one of {CATEGORY_ORDER}")
    if name in _REGISTERED_NAMES:
        return
    _REGISTERED_NAMES.add(name)
    if deprecated_names:
        _DEPRECATED_LOOKUP[name] = deprecated_names
    ENV_REGISTRY.append(EnvVar(name, default, description, category, deprecated_names))


# ---------------------------------------------------------------------------
# Core env vars (read logic in config.py / HubConfig)
# ---------------------------------------------------------------------------
_env_register("MODELSCOPE_API_TOKEN", "-", "API authentication token", "Core")
_env_register("MODELSCOPE_ENDPOINT", DEFAULT_ENDPOINT, "API endpoint URL", "Core")
_env_register("MODELSCOPE_CACHE", "~/.cache/modelscope", "Local cache directory", "Core")
ENV_CACHE: str = "MODELSCOPE_CACHE"
_env_register("MODELSCOPE_HOME", "~/.modelscope", "SDK config directory", "Core")


# ---------------------------------------------------------------------------
# Network / IO tunables
# ---------------------------------------------------------------------------
API_TIMEOUT: int = _env_int(
    "MODELSCOPE_API_TIMEOUT",
    60,
    "HTTP request timeout (seconds)",
    "Network",
    "API_TIMEOUT",
)

API_CONNECT_TIMEOUT: int = _env_int(
    "MODELSCOPE_API_CONNECT_TIMEOUT",
    10,
    "HTTP connect timeout (seconds)",
    "Network",
)

API_MAX_RETRIES: int = _env_int(
    "MODELSCOPE_API_MAX_RETRIES",
    5,
    "Max retry attempts for transient failures",
    "Network",
    "API_MAX_RETRIES",
)

API_CONNECTION_POOL_MAXSIZE: int = _env_int(
    "MODELSCOPE_API_CONNECTION_POOL_MAXSIZE",
    32,
    "Per-host HTTP connection pool size",
    "Network",
)
"""Connections kept alive per host, and the concurrency the pool can serve.

urllib3 defaults this to 10. A folder upload runs ``max_workers`` requests at
once -- commonly 16 or more for bulk transfers -- so the default silently
discards the excess connections ("Connection pool is full, discarding
connection") and every discarded one costs a fresh TLS handshake on its next
use. This must be at least as large as the worker count to avoid that churn.
"""

REPO_TREE_PAGE_MAX_ATTEMPTS: int = _env_int(
    "MODELSCOPE_REPO_TREE_PAGE_MAX_ATTEMPTS",
    4,
    "Attempts for one repo-tree page denied on an already-authorized listing",
    "Network",
)
"""Retries for a spurious ``403`` on a single page of a paginated tree listing.

A large dataset listing spans hundreds of pages and an occasional page answers
``403 无权访问该数据集`` on a repository the caller has just read successfully.
Treating that as an authorization result throws away every page already
collected, so it is retried once the credential has been proven by an earlier
page.
"""
REPO_TREE_PAGE_RETRY_MAX_DELAY_SECONDS: int = _env_int(
    "MODELSCOPE_REPO_TREE_PAGE_RETRY_MAX_DELAY_SECONDS",
    8,
    "Maximum backoff between repo-tree page retries (seconds)",
    "Network",
)

REPO_FILES_TRUNCATION_LIMIT: int = 3000
"""Server-side hard cap on a single ``repo/files`` listing.

``GET /api/v1/{type}s/{repo_id}/repo/files`` silently truncates the file tree at
this many entries: the response is ``HTTP 200`` with ``Success: true``, carries
neither ``TotalCount`` nor a truncation flag, and ignores every pagination
parameter. A listing whose length equals this limit therefore means "there may
be more", and the tree has to be re-enumerated with ``Root``-scoped requests.
"""

REPO_TREE_MAX_REQUESTS: int = _env_int(
    "MODELSCOPE_REPO_TREE_MAX_REQUESTS",
    5000,
    "Request budget for walking a truncated repo file tree",
    "Network",
)

REPO_TREE_WALK_WORKERS: int = _env_int(
    "MODELSCOPE_REPO_TREE_WALK_WORKERS",
    8,
    "Concurrent listings when walking a truncated repo file tree",
    "Network",
)

# ---------------------------------------------------------------------------
# Endpoint switching
# ---------------------------------------------------------------------------
ENV_MODELSCOPE_DOMAIN: str = "MODELSCOPE_DOMAIN"
_env_register(ENV_MODELSCOPE_DOMAIN, "-", "Deprecated: use MODELSCOPE_ENDPOINT", "Deprecated")

ENV_PREFER_AI_SITE: str = "MODELSCOPE_PREFER_AI_SITE"
_env_register(ENV_PREFER_AI_SITE, "false", "Prefer modelscope.ai over modelscope.cn", "Core")

DEFAULT_INTL_ENDPOINT: str = "https://www.modelscope.ai"
"""International site endpoint."""

# ---------------------------------------------------------------------------
# Download tunables
# ---------------------------------------------------------------------------
DOWNLOAD_CHUNK_SIZE: int = _env_int_mb(
    "MODELSCOPE_DOWNLOAD_CHUNK_SIZE_MB",
    1,
    "Streaming chunk size (MB)",
    "Download",
    "DOWNLOAD_CHUNK_SIZE",
)

DOWNLOAD_PARALLEL_THRESHOLD: int = (
    _env_int(
        "MODELSCOPE_DOWNLOAD_PARALLEL_THRESHOLD_MB",
        500,
        "Parallel download threshold (MB)",
        "Download",
        "MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB",
    )
    * 1024
    * 1024
)

DOWNLOAD_PARALLELS: int = _env_int(
    "MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS",
    1,
    "Parallel range-download streams",
    "Download",
    "MODELSCOPE_DOWNLOAD_PARALLELS",
)

DOWNLOAD_RETRY_TIMES: int = _env_int(
    "MODELSCOPE_DOWNLOAD_MAX_RETRIES",
    5,
    "Per-file download retry count",
    "Download",
    "DOWNLOAD_RETRY_TIMES",
)

DOWNLOAD_TIMEOUT: int = _env_int(
    "MODELSCOPE_DOWNLOAD_TIMEOUT",
    60,
    "Per-file download timeout (seconds)",
    "Download",
    "DOWNLOAD_TIMEOUT",
)

DOWNLOAD_PART_SIZE: int = _env_int_mb(
    "MODELSCOPE_DOWNLOAD_PART_SIZE_MB",
    160,
    "Parallel range chunk size (MB)",
    "Download",
    "DOWNLOAD_PART_SIZE",
)

TEMPORARY_FOLDER_NAME: str = "._____temp"
"""Temporary folder name used during downloads."""

FILE_HASH_FIELD: str = "Sha256"
"""API response field name for file hash."""

FILE_HASH: str = FILE_HASH_FIELD
"""Legacy alias of :data:`FILE_HASH_FIELD` (modelscope SDK name)."""

ENV_FILE_LOCK: str = "MODELSCOPE_DOWNLOAD_FILE_LOCK"
_env_register(
    ENV_FILE_LOCK,
    "true",
    "File lock for multiprocess download safety",
    "Download",
    deprecated_names=("MODELSCOPE_HUB_FILE_LOCK",),
)

ENV_INTRA_CLOUD_ACCELERATION: str = "MODELSCOPE_DOWNLOAD_INTRA_CLOUD"
_env_register(
    ENV_INTRA_CLOUD_ACCELERATION,
    "true",
    "Alibaba cloud intra-cloud acceleration",
    "Download",
    deprecated_names=("INTRA_CLOUD_ACCELERATION",),
)

ENV_INTRA_CLOUD_REGION: str = "MODELSCOPE_DOWNLOAD_INTRA_CLOUD_REGION"
_env_register(
    ENV_INTRA_CLOUD_REGION,
    "(auto)",
    "Override intra-cloud region ID",
    "Download",
    deprecated_names=("INTRA_CLOUD_ACCELERATION_REGION",),
)

ENV_INTER_CLOUD_REGIONS: str = "MODELSCOPE_DOWNLOAD_INTER_CLOUD_REGIONS"
_env_register(
    ENV_INTER_CLOUD_REGIONS, "", "Comma-separated peer regions for cross-region internal acceleration", "Download"
)

# Upload: blob transport and retries
UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS: int = _env_int(
    "MODELSCOPE_UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS",
    30,
    "Blob upload connection timeout (seconds)",
    "Upload",
    "MODELSCOPE_UPLOAD_CONNECT_TIMEOUT",
    "UPLOAD_BLOB_CONNECT_TIMEOUT",
)
UPLOAD_BLOB_READ_TIMEOUT_SECONDS: int = _env_int(
    "MODELSCOPE_UPLOAD_BLOB_READ_TIMEOUT_SECONDS",
    3600,
    "Blob upload socket read idle timeout (seconds)",
    "Upload",
    "MODELSCOPE_UPLOAD_READ_TIMEOUT",
    "UPLOAD_BLOB_READ_TIMEOUT",
    "UPLOAD_BLOB_TIMEOUT_SECONDS",
)
UPLOAD_BLOB_MAX_ATTEMPTS: int = _env_int(
    "MODELSCOPE_UPLOAD_BLOB_MAX_ATTEMPTS",
    5,
    "Maximum total attempts for one blob upload",
    "Upload",
    "UPLOAD_BLOB_MAX_RETRIES",
)
UPLOAD_BLOB_RETRY_BACKOFF_BASE_SECONDS: int = _env_int(
    "MODELSCOPE_UPLOAD_BLOB_RETRY_BACKOFF_BASE_SECONDS",
    2,
    "Exponential backoff base for blob retries (seconds)",
    "Upload",
    "UPLOAD_BLOB_RETRY_BACKOFF",
)
UPLOAD_BLOB_RETRY_MAX_DELAY_SECONDS: int = _env_int(
    "MODELSCOPE_UPLOAD_BLOB_RETRY_MAX_DELAY_SECONDS",
    60,
    "Maximum delay between blob attempts (seconds)",
    "Upload",
    "UPLOAD_BLOB_RETRY_MAX_WAIT",
)
UPLOAD_BLOB_PROGRESS_THRESHOLD_BYTES: int = _env_int_mb_with_deprecated_units(
    "MODELSCOPE_UPLOAD_BLOB_PROGRESS_THRESHOLD_MB",
    5,
    "Minimum blob size for displaying upload progress (MB)",
    "Upload",
    deprecated_byte_names=("UPLOAD_BLOB_TQDM_DISABLE_THRESHOLD",),
)

# Upload: HTTP transport retries
UPLOAD_HTTP_RETRY_ALLOWED_METHODS: frozenset[str] = _env_csv_frozenset(
    "MODELSCOPE_UPLOAD_HTTP_RETRY_ALLOWED_METHODS",
    "GET,HEAD,DELETE,OPTIONS,TRACE",
    "HTTP methods eligible for automatic transport retries",
    "Upload",
    "UPLOAD_RETRY_ALLOWED_METHODS",
)

# Upload: batching and commit retries
UPLOAD_COMMIT_BATCH_MAX_OPERATIONS: int = _env_int(
    "MODELSCOPE_UPLOAD_COMMIT_BATCH_MAX_OPERATIONS",
    256,
    "Maximum operations in one upload commit batch",
    "Upload",
    "UPLOAD_COMMIT_BATCH_SIZE",
)
UPLOAD_BLOB_VALIDATION_BATCH_MAX_OBJECTS: int = _env_int(
    "MODELSCOPE_UPLOAD_BLOB_VALIDATION_BATCH_MAX_OBJECTS",
    64,
    "Maximum objects in one blob validation request",
    "Upload",
    "UPLOAD_VALIDATE_BLOB_BATCH_SIZE",
)
UPLOAD_ADAPTIVE_BATCHING_ENABLED: bool = _env_bool(
    "MODELSCOPE_UPLOAD_ADAPTIVE_BATCHING_ENABLED",
    True,
    "Enable adaptive upload commit batch sizing",
    "Upload",
    "UPLOAD_ADAPTIVE_BATCH_SIZE",
)
UPLOAD_COMMIT_MAX_INLINE_BYTES: int = _env_bytes(
    "MODELSCOPE_UPLOAD_COMMIT_MAX_INLINE_BYTES",
    8 * 1024 * 1024,
    "Advisory estimated request-body budget for one upload commit",
    "Upload",
)
"""Advisory byte budget for an estimated commit request body.

Normal files contribute their base64-expanded content plus operation JSON; LFS
files contribute only pointer/action metadata. Batching closes on this budget or
the operation target, whichever comes first. A single oversized inline file is
still sent alone so this advisory budget never blocks an upload.
"""
UPLOAD_COMMIT_MAX_PER_HOUR: int = _env_int(
    "MODELSCOPE_UPLOAD_COMMIT_MAX_PER_HOUR",
    0,
    "Client-side commit rate ceiling per hour (0 disables the governor)",
    "Upload",
    allow_zero=True,
)
"""Opt-in client-side commit budget, disabled by default.

The Hub throttles commits per repository. Reacting to a throttle costs a failed
round trip and, when the server holds the connection instead of answering, a
full read timeout. A bulk pipeline that knows its budget can set this to spread
commits out and never trip the limit; interactive uploads stay unthrottled.
"""
UPLOAD_COMMIT_MAX_RETRY_AFTER_SECONDS: int = _env_int(
    "MODELSCOPE_UPLOAD_COMMIT_MAX_RETRY_AFTER_SECONDS",
    1800,
    "Longest server-provided Retry-After a commit will honor (seconds)",
    "Upload",
)
"""Upper bound on an honored ``Retry-After`` for a throttled commit.

A rate limit is a bounded wait the server declares, unlike a transient failure
of unknown duration, so it is budgeted separately from
:data:`UPLOAD_COMMIT_RETRY_TOTAL_WAIT_SECONDS` rather than exhausting it.
"""
COMMIT_MAX_ACTIONS_PER_REQUEST: int = _env_int(
    "MODELSCOPE_COMMIT_MAX_ACTIONS_PER_REQUEST",
    2000,
    "Server-enforced maximum actions in one commit request",
    "Upload",
)
"""Hard ceiling the server puts on a single commit request.

Exceeding it is rejected outright with ``HTTP 422``::

    commit request exceeds actions limit: 3300 > 2000; split the commit into
    smaller batches

This is a server contract, not a tuning preference, so every commit path clamps
to it -- uploads, and deletes, which otherwise put every path in one request.
"""

UPLOAD_PROGRESS_MIN_INTERVAL_SECONDS: int = _env_int(
    "MODELSCOPE_UPLOAD_PROGRESS_MIN_INTERVAL_SECONDS",
    1,
    "Minimum gap between wire-level upload progress events (seconds)",
    "Upload",
)
"""Throttle for per-file upload progress events.

Commits land in lumps tens of seconds apart, so a rate built only from commit
events alternates between a spike and zero and cannot distinguish a slow batch
from a hung one. Blob uploads finish continuously and are the honest source for
a rate -- but there is one per file, so the events are coalesced to this interval
rather than fanning out tens of thousands of callback invocations.
"""
UPLOAD_COMMIT_MAX_ATTEMPTS: int = _env_int(
    "MODELSCOPE_UPLOAD_COMMIT_MAX_ATTEMPTS",
    5,
    "Maximum total attempts for one upload commit",
    "Upload",
    "UPLOAD_COMMIT_MAX_RETRIES",
)
UPLOAD_COMMIT_RETRY_TOTAL_WAIT_SECONDS: int = _env_int(
    "MODELSCOPE_UPLOAD_COMMIT_RETRY_TOTAL_WAIT_SECONDS",
    300,
    "Maximum total wait across upload commit retries (seconds)",
    "Upload",
    "MODELSCOPE_UPLOAD_COMMIT_MAX_TOTAL_WAIT",
)
UPLOAD_COMMIT_MAX_CONSECUTIVE_FAILED_BATCHES: int = _env_int(
    "MODELSCOPE_UPLOAD_COMMIT_MAX_CONSECUTIVE_FAILED_BATCHES",
    3,
    "Maximum consecutive failed upload commit batches",
    "Upload",
    "MODELSCOPE_UPLOAD_BATCH_CONSECUTIVE_FAILURE_LIMIT",
)
UPLOAD_FAILED_FILE_MAX_RETRY_ROUNDS: int = _env_int(
    "MODELSCOPE_UPLOAD_FAILED_FILE_MAX_RETRY_ROUNDS",
    3,
    "Maximum retry rounds for failed upload files",
    "Upload",
    "UPLOAD_FAILED_FILE_MAX_RETRIES",
)
# Upload: progressive recovery
UPLOAD_RECOVERY_ENABLED: bool = _env_bool(
    "MODELSCOPE_UPLOAD_RECOVERY_ENABLED",
    True,
    "Enable progressive upload recovery",
    "Upload",
    "UPLOAD_REACT_ENABLED",
)
UPLOAD_RECOVERY_SERIAL_BACKOFF_BASE_SECONDS: int = _env_int(
    "MODELSCOPE_UPLOAD_RECOVERY_SERIAL_BACKOFF_BASE_SECONDS",
    2,
    "Backoff base for serial upload recovery (seconds)",
    "Upload",
    "UPLOAD_REACT_ROUND2_BASE_DELAY",
)
UPLOAD_RECOVERY_SINGLE_FILE_DELAY_SECONDS: int = _env_int(
    "MODELSCOPE_UPLOAD_RECOVERY_SINGLE_FILE_DELAY_SECONDS",
    5,
    "Delay between single-file recovery attempts (seconds)",
    "Upload",
    "UPLOAD_REACT_ROUND3_FILE_DELAY",
)
UPLOAD_RECOVERY_BACKOFF_MAX_EXPONENT: int = _env_int(
    "MODELSCOPE_UPLOAD_RECOVERY_BACKOFF_MAX_EXPONENT",
    5,
    "Maximum exponent for progressive upload recovery backoff",
    "Upload",
    "UPLOAD_REACT_BACKOFF_MAX_EXPONENT",
)
UPLOAD_RECOVERY_MAX_DELAY_SECONDS: int = _env_int(
    "MODELSCOPE_UPLOAD_RECOVERY_MAX_DELAY_SECONDS",
    120,
    "Maximum progressive upload recovery delay (seconds)",
    "Upload",
    "UPLOAD_REACT_MAX_DELAY",
)

# Upload: workers
UPLOAD_MAX_CONCURRENT_WORKERS: int = _env_int(
    "MODELSCOPE_UPLOAD_MAX_CONCURRENT_WORKERS",
    min(8, (os.cpu_count() or 4) + 4),
    "Maximum concurrent upload workers",
    "Upload",
    "MODELSCOPE_UPLOAD_MAX_WORKERS",
    "DEFAULT_MAX_WORKERS",
)

# Upload: cache / tracker
UPLOAD_CACHE_ENABLED: bool = _env_bool(
    "MODELSCOPE_UPLOAD_CACHE_ENABLED",
    True,
    "Enable resumable upload cache",
    "Upload",
    "MODELSCOPE_UPLOAD_CACHE",
    "UPLOAD_USE_CACHE",
)
_env_register(
    "MODELSCOPE_UPLOAD_IGNORE_FILE_PATTERN",
    "-",
    "File pattern excluded by legacy push_to_hub uploads",
    "Upload",
    deprecated_names=("UPLOAD_IGNORE_FILE_PATTERN",),
)


def get_upload_ignore_file_pattern() -> str | None:
    """Return the optional ignore pattern used by legacy ``push_to_hub`` calls."""
    return _env("MODELSCOPE_UPLOAD_IGNORE_FILE_PATTERN", "UPLOAD_IGNORE_FILE_PATTERN")


UPLOAD_CACHE_FILE: str = ".ms_upload_cache"
UPLOAD_LEGACY_PROGRESS_FILE: str = ".ms_upload_progress"

# Upload: limits
UPLOAD_LFS_FORCE_THRESHOLD_BYTES: int = _env_bytes(
    "MODELSCOPE_UPLOAD_LFS_FORCE_THRESHOLD",
    64 * 1024,
    "File size above which LFS mode is forced (bytes; accepts a unit suffix, 0 forces LFS for every file)",
    "Upload",
    deprecated_mb_names=("MODELSCOPE_UPLOAD_LFS_FORCE_THRESHOLD_MB",),
    deprecated_byte_names=("UPLOAD_LFS_ENFORCE_THRESHOLD", "UPLOAD_SIZE_THRESHOLD_TO_ENFORCE_LFS"),
    allow_zero=True,
)
"""Size above which a file is uploaded as LFS regardless of its suffix.

Files at or below the threshold are committed inline as base64, which puts their
bytes in the commit request body and caps how many of them one commit can carry.
Lowering the threshold moves that content onto the pre-signed object-storage
path instead, leaving the commit with only ``sha256`` plus ``size``.

``0`` forces LFS for every non-empty file. It stays safe because
:data:`UPLOAD_INLINE_METADATA_PATHS` is consulted first, so the repository files
the Hub itself parses are never turned into LFS pointers.
"""
UPLOAD_INLINE_METADATA_PATHS: frozenset[str] = _env_csv_frozenset(
    "MODELSCOPE_UPLOAD_INLINE_METADATA_PATHS",
    "README.md,.gitattributes,.gitignore,configuration.json,configuration.yaml,configuration.yml,"
    "dataset_infos.json,config.json,.msc,.mdl",
    "Repository file names always committed inline, never as LFS",
    "Upload",
)
"""Repository-relative file names that must stay inline in the commit.

The Hub parses these server-side -- the dataset/model card front matter, the
configuration files, the git attribute rules. Stored as an LFS pointer, the
server would read the 130-byte pointer text instead of the real content and the
card or configuration would silently render empty. Matching is on the file name
(case-insensitive), so the rule holds at any depth in the tree, and it is
checked before the size and suffix rules.
"""
UPLOAD_MAX_FILE_SIZE_BYTES: int = _env_int_mb_with_deprecated_units(
    "MODELSCOPE_UPLOAD_MAX_FILE_SIZE_MB",
    100 * 1024,
    "Advisory single upload file size warning threshold (MB, default 100 GB)",
    "Upload",
    deprecated_mb_names=("UPLOAD_MAX_FILE_SIZE_MB",),
    deprecated_byte_names=("UPLOAD_MAX_FILE_SIZE",),
)
UPLOAD_MAX_FILE_COUNT: int = _env_int(
    "MODELSCOPE_UPLOAD_MAX_FILE_COUNT",
    100_000,
    "Advisory total file-count warning threshold per upload",
    "Upload",
    "UPLOAD_MAX_FILE_COUNT",
)
UPLOAD_MAX_FILES_PER_DIRECTORY: int = _env_int(
    "MODELSCOPE_UPLOAD_MAX_FILES_PER_DIRECTORY",
    50_000,
    "Advisory per-directory file-count warning threshold",
    "Upload",
    "UPLOAD_MAX_FILE_COUNT_IN_DIR",
)
UPLOAD_NORMAL_FILES_TOTAL_SIZE_BYTES: int = _env_int_mb_with_deprecated_units(
    "MODELSCOPE_UPLOAD_NORMAL_FILES_TOTAL_SIZE_MB",
    500,
    "Advisory total normal-file size warning threshold (MB)",
    "Upload",
    deprecated_byte_names=("UPLOAD_NORMAL_FILE_SIZE_TOTAL_LIMIT",),
)

# Deprecated Python aliases. Runtime code must use the explicit names above.
UPLOAD_BLOB_CONNECT_TIMEOUT = UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS
UPLOAD_BLOB_READ_TIMEOUT = UPLOAD_BLOB_READ_TIMEOUT_SECONDS
UPLOAD_BLOB_TIMEOUT = (UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS, UPLOAD_BLOB_READ_TIMEOUT_SECONDS)
UPLOAD_BLOB_MAX_RETRIES = UPLOAD_BLOB_MAX_ATTEMPTS
UPLOAD_BLOB_RETRY_BACKOFF = UPLOAD_BLOB_RETRY_BACKOFF_BASE_SECONDS
UPLOAD_BLOB_RETRY_MAX_WAIT = UPLOAD_BLOB_RETRY_MAX_DELAY_SECONDS
UPLOAD_BLOB_TQDM_DISABLE_THRESHOLD = UPLOAD_BLOB_PROGRESS_THRESHOLD_BYTES
UPLOAD_RETRY_ALLOWED_METHODS = UPLOAD_HTTP_RETRY_ALLOWED_METHODS
UPLOAD_COMMIT_BATCH_SIZE = UPLOAD_COMMIT_BATCH_MAX_OPERATIONS
UPLOAD_VALIDATE_BLOB_BATCH_SIZE = UPLOAD_BLOB_VALIDATION_BATCH_MAX_OBJECTS
UPLOAD_ADAPTIVE_BATCH_SIZE = UPLOAD_ADAPTIVE_BATCHING_ENABLED
UPLOAD_COMMIT_MAX_RETRIES = UPLOAD_COMMIT_MAX_ATTEMPTS
UPLOAD_COMMIT_MAX_TOTAL_WAIT = UPLOAD_COMMIT_RETRY_TOTAL_WAIT_SECONDS
UPLOAD_BATCH_CONSECUTIVE_FAILURE_LIMIT = UPLOAD_COMMIT_MAX_CONSECUTIVE_FAILED_BATCHES
UPLOAD_FAILED_FILE_MAX_RETRIES = UPLOAD_FAILED_FILE_MAX_RETRY_ROUNDS
UPLOAD_REACT_ENABLED = UPLOAD_RECOVERY_ENABLED
UPLOAD_REACT_ROUND2_BASE_DELAY = UPLOAD_RECOVERY_SERIAL_BACKOFF_BASE_SECONDS
UPLOAD_REACT_ROUND3_FILE_DELAY = UPLOAD_RECOVERY_SINGLE_FILE_DELAY_SECONDS
UPLOAD_REACT_BACKOFF_MAX_EXPONENT = UPLOAD_RECOVERY_BACKOFF_MAX_EXPONENT
UPLOAD_REACT_MAX_DELAY = UPLOAD_RECOVERY_MAX_DELAY_SECONDS
DEFAULT_MAX_WORKERS = UPLOAD_MAX_CONCURRENT_WORKERS
UPLOAD_USE_CACHE = UPLOAD_CACHE_ENABLED
UPLOAD_LFS_ENFORCE_THRESHOLD = UPLOAD_LFS_FORCE_THRESHOLD_BYTES
UPLOAD_SIZE_THRESHOLD_TO_ENFORCE_LFS = UPLOAD_LFS_FORCE_THRESHOLD_BYTES
UPLOAD_MAX_FILE_COUNT_IN_DIR = UPLOAD_MAX_FILES_PER_DIRECTORY
UPLOAD_MAX_FILE_SIZE = UPLOAD_MAX_FILE_SIZE_BYTES
UPLOAD_NORMAL_FILE_SIZE_TOTAL_LIMIT = UPLOAD_NORMAL_FILES_TOTAL_SIZE_BYTES
# This setting never affected runtime upload selection; keep only the import.
UPLOAD_LFS_THRESHOLD: int = 5 * 1024 * 1024

# LFS suffix lists (from old SDK — determines upload mode regardless of size)
MODEL_LFS_SUFFIX: list[str] = [
    ".7z",
    ".arrow",
    ".bin",
    ".bz2",
    ".ckpt",
    ".ftz",
    ".gz",
    ".h5",
    ".joblib",
    ".mlmodel",
    ".model",
    ".msgpack",
    ".npy",
    ".npz",
    ".onnx",
    ".ot",
    ".parquet",
    ".pb",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".rar",
    ".safetensors",
    ".tar",
    ".tflite",
    ".tgz",
    ".wasm",
    ".xz",
    ".zip",
    ".zst",
]
DATASET_LFS_SUFFIX: list[str] = [
    ".7z",
    ".aac",
    ".arrow",
    ".audio",
    ".bmp",
    ".bin",
    ".bz2",
    ".flac",
    ".ftz",
    ".gif",
    ".gz",
    ".h5",
    ".jack",
    ".jpeg",
    ".jpg",
    ".png",
    ".jsonl",
    ".joblib",
    ".lz4",
    ".msgpack",
    ".npy",
    ".npz",
    ".ot",
    ".parquet",
    ".pb",
    ".pickle",
    ".pcm",
    ".pkl",
    ".raw",
    ".rar",
    ".sam",
    ".tar",
    ".tgz",
    ".wasm",
    ".wav",
    ".webm",
    ".webp",
    ".zip",
    ".zst",
    ".tiff",
    ".mp3",
    ".mp4",
    ".ogg",
]

# Default ignore patterns for folder upload
DEFAULT_IGNORE_PATTERNS: list[str] = [
    ".git",
    ".git/*",
    "*/.git",
    "**/.git/**",
    ".cache",
    ".cache/*",
    "*/.cache",
    "**/.cache/**",
]


# ---------------------------------------------------------------------------
# Branding
# ---------------------------------------------------------------------------
MODELSCOPE_ASCII = r"""
 _   .-')                _ .-') _     ('-.             .-')                              _ (`-.    ('-.
( '.( OO )_             ( (  OO) )  _(  OO)           ( OO ).                           ( (OO  ) _(  OO)
 ,--.   ,--.).-'),-----. \     .'_ (,------.,--.     (_)---\_)   .-----.  .-'),-----.  _.`     \(,------.
 |   `.'   |( OO'  .-.  ',`'--..._) |  .---'|  |.-') /    _ |   '  .--./ ( OO'  .-.  '(__...--'' |  .---'
 |         |/   |  | |  ||  |  \  ' |  |    |  | OO )\  :` `.   |  |('-. /   |  | |  | |  /  | | |  |
 |  |'.'|  |\_) |  |\|  ||  |   ' |(|  '--. |  |`-' | '..`''.) /_) |OO  )\_) |  |\|  | |  |_.' |(|  '--.
 |  |   |  |  \ |  | |  ||  |   / : |  .--'(|  '---.'.-._)   \ ||  |`-'|   \ |  | |  | |  .___.' |  .--'
 |  |   |  |   `'  '-'  '|  '--'  / |  `---.|      | \       /(_'  '--'\    `'  '-'  ' |  |      |  `---.
 `--'   `--'     `-----' `-------'  `------'`------'  `-----'    `-----'      `-----'  `--'      `------'
"""  # noqa: E501


# ---------------------------------------------------------------------------
# Logging / deprecated (read logic in utils/logger.py, cli/compat.py)
# ---------------------------------------------------------------------------
_env_register("MODELSCOPE_LOG_LEVEL", "INFO", "SDK log level (DEBUG/INFO/WARNING/ERROR)", "Logging")
_env_register(
    "MODELSCOPE_NO_DEPRECATION_WARNINGS",
    "-",
    "Suppress deprecation warnings",
    "Logging",
    deprecated_names=("MODELSCOPE_HUB_NO_DEPRECATION_WARNINGS",),
)


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
DEFAULT_CACHE_DIR_NAME: str = "modelscope"
SESSION_FILE_NAME: str = "session"
CONFIG_DIR_NAME: str = ".modelscope"
CREDENTIALS_DIR_NAME: str = "credentials"
COOKIES_FILE_NAME: str = "cookies"
GIT_TOKEN_FILE_NAME: str = "git_token"
USER_INFO_FILE_NAME: str = "user"


# ---------------------------------------------------------------------------
# Agent plugin loading (``ms agent install``)
#
# These constrain where the plugin that ``ms agent install`` imports may come
# from. The security model they serve is documented in
# :mod:`modelscope_hub.agent._plugin`.
# ---------------------------------------------------------------------------
ENV_AGENT_PLUGIN_REPO: str = "MODELSCOPE_AGENT_PLUGIN_REPO"

#: Owners allowed to provide the agent plugin.
#:
#: A compile-time constant with **no environment override**, on purpose. This list
#: is the trust anchor for a command that executes downloaded code, and an anchor
#: any parent process can rewrite through the environment is not an anchor: a
#: script that can set env vars could point ``ms agent install`` at a repository
#: it controls. Deciding who is trusted is a reviewed code change.
#:
#: ``MODELSCOPE_AGENT_PLUGIN_REPO`` *is* overridable and that is safe: it chooses
#: which repository to fetch, but the owner still has to appear here, so it can
#: pick among already-trusted owners without widening trust.
AGENT_PLUGIN_TRUSTED_OWNERS: frozenset[str] = frozenset({"modelscope", "AI-ModelScope"})

DEFAULT_AGENT_PLUGIN_REPO: str = "modelscope/agent-hub-plugin"
DEFAULT_AGENT_PLUGIN_REVISION: str = "master"

_env_register(
    ENV_AGENT_PLUGIN_REPO,
    DEFAULT_AGENT_PLUGIN_REPO,
    "Model repository id ('owner/name') of the agent plugin used by 'ms agent install'",
    "Core",
)

__all__ = [
    "AGENT_PLUGIN_TRUSTED_OWNERS",
    "API_CONNECT_TIMEOUT",
    "API_CONNECTION_POOL_MAXSIZE",
    "API_MAX_RETRIES",
    "API_TIMEOUT",
    "CATEGORY_ORDER",
    "COMMIT_MAX_ACTIONS_PER_REQUEST",
    "CONFIG_DIR_NAME",
    "DATASET_LFS_SUFFIX",
    "DEFAULT_AGENT_PLUGIN_REPO",
    "DEFAULT_AGENT_PLUGIN_REVISION",
    "DEFAULT_CACHE_DIR_NAME",
    "DEFAULT_CREDENTIALS_PATH",
    "DEFAULT_DATASET_REVISION",
    "DEFAULT_ENDPOINT",
    "DEFAULT_IGNORE_PATTERNS",
    "DEFAULT_INTL_ENDPOINT",
    "DEFAULT_MAX_WORKERS",
    "DEFAULT_MODELSCOPE_DATA_ENDPOINT",
    "DEFAULT_MODELSCOPE_DOMAIN",
    "DEFAULT_MODELSCOPE_GROUP",
    "DEFAULT_MODELSCOPE_INTL_DATA_ENDPOINT",
    "DEFAULT_MODELSCOPE_INTL_DOMAIN",
    "DEFAULT_SKILLS_DIR",
    "DOWNLOAD_CHUNK_SIZE",
    "DOWNLOAD_PARALLEL_THRESHOLD",
    "DOWNLOAD_PARALLELS",
    "DOWNLOAD_PART_SIZE",
    "DOWNLOAD_RETRY_TIMES",
    "DOWNLOAD_TIMEOUT",
    "ENV_AGENT_PLUGIN_REPO",
    "ENV_FILE_LOCK",
    "ENV_CACHE",
    "ENV_INTRA_CLOUD_ACCELERATION",
    "ENV_INTRA_CLOUD_REGION",
    "ENV_INTER_CLOUD_REGIONS",
    "ENV_MODELSCOPE_DOMAIN",
    "ENV_PREFER_AI_SITE",
    "ENV_REGISTRY",
    "EnvVar",
    "FILE_HASH",
    "FILE_HASH_FIELD",
    "get_upload_ignore_file_pattern",
    "LEGACY_API_PREFIX",
    "License",
    "MODEL_ID_SEPARATOR",
    "MODEL_LFS_SUFFIX",
    "OPENAPI_PREFIX",
    "REPO_TREE_PAGE_MAX_ATTEMPTS",
    "REPO_TREE_PAGE_RETRY_MAX_DELAY_SECONDS",
    "REPO_TYPE_DATASET",
    "REPO_TYPE_MODEL",
    "REPO_TYPE_STUDIO",
    "REPO_TYPE_SUPPORT",
    "RepoType",
    "StrEnum",
    "SESSION_FILE_NAME",
    "StudioVisibility",
    "TEMPORARY_FOLDER_NAME",
    "TokenScope",
    "UPLOAD_ADAPTIVE_BATCHING_ENABLED",
    "UPLOAD_ADAPTIVE_BATCH_SIZE",
    "UPLOAD_BATCH_CONSECUTIVE_FAILURE_LIMIT",
    "UPLOAD_BLOB_CONNECT_TIMEOUT",
    "UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS",
    "UPLOAD_BLOB_MAX_ATTEMPTS",
    "UPLOAD_BLOB_MAX_RETRIES",
    "UPLOAD_BLOB_PROGRESS_THRESHOLD_BYTES",
    "UPLOAD_BLOB_READ_TIMEOUT",
    "UPLOAD_BLOB_READ_TIMEOUT_SECONDS",
    "UPLOAD_BLOB_RETRY_BACKOFF",
    "UPLOAD_BLOB_RETRY_BACKOFF_BASE_SECONDS",
    "UPLOAD_BLOB_RETRY_MAX_DELAY_SECONDS",
    "UPLOAD_BLOB_RETRY_MAX_WAIT",
    "UPLOAD_BLOB_TIMEOUT",
    "UPLOAD_BLOB_TQDM_DISABLE_THRESHOLD",
    "UPLOAD_BLOB_VALIDATION_BATCH_MAX_OBJECTS",
    "UPLOAD_CACHE_ENABLED",
    "UPLOAD_CACHE_FILE",
    "UPLOAD_COMMIT_BATCH_MAX_OPERATIONS",
    "UPLOAD_COMMIT_BATCH_SIZE",
    "UPLOAD_COMMIT_MAX_ATTEMPTS",
    "UPLOAD_COMMIT_MAX_CONSECUTIVE_FAILED_BATCHES",
    "UPLOAD_COMMIT_MAX_INLINE_BYTES",
    "UPLOAD_COMMIT_MAX_PER_HOUR",
    "UPLOAD_COMMIT_MAX_RETRIES",
    "UPLOAD_COMMIT_MAX_RETRY_AFTER_SECONDS",
    "UPLOAD_COMMIT_MAX_TOTAL_WAIT",
    "UPLOAD_COMMIT_RETRY_TOTAL_WAIT_SECONDS",
    "UPLOAD_FAILED_FILE_MAX_RETRIES",
    "UPLOAD_FAILED_FILE_MAX_RETRY_ROUNDS",
    "UPLOAD_HTTP_RETRY_ALLOWED_METHODS",
    "UPLOAD_INLINE_METADATA_PATHS",
    "UPLOAD_LEGACY_PROGRESS_FILE",
    "UPLOAD_LFS_ENFORCE_THRESHOLD",
    "UPLOAD_LFS_FORCE_THRESHOLD_BYTES",
    "UPLOAD_LFS_THRESHOLD",
    "UPLOAD_MAX_CONCURRENT_WORKERS",
    "UPLOAD_MAX_FILE_COUNT",
    "UPLOAD_MAX_FILE_COUNT_IN_DIR",
    "UPLOAD_MAX_FILE_SIZE",
    "UPLOAD_MAX_FILE_SIZE_BYTES",
    "UPLOAD_MAX_FILES_PER_DIRECTORY",
    "UPLOAD_NORMAL_FILE_SIZE_TOTAL_LIMIT",
    "UPLOAD_NORMAL_FILES_TOTAL_SIZE_BYTES",
    "UPLOAD_PROGRESS_MIN_INTERVAL_SECONDS",
    "UPLOAD_REACT_BACKOFF_MAX_EXPONENT",
    "UPLOAD_REACT_ENABLED",
    "UPLOAD_REACT_MAX_DELAY",
    "UPLOAD_REACT_ROUND2_BASE_DELAY",
    "UPLOAD_REACT_ROUND3_FILE_DELAY",
    "UPLOAD_RECOVERY_BACKOFF_MAX_EXPONENT",
    "UPLOAD_RECOVERY_ENABLED",
    "UPLOAD_RECOVERY_MAX_DELAY_SECONDS",
    "UPLOAD_RECOVERY_SERIAL_BACKOFF_BASE_SECONDS",
    "UPLOAD_RECOVERY_SINGLE_FILE_DELAY_SECONDS",
    "UPLOAD_RETRY_ALLOWED_METHODS",
    "UPLOAD_SIZE_THRESHOLD_TO_ENFORCE_LFS",
    "UPLOAD_USE_CACHE",
    "UPLOAD_VALIDATE_BLOB_BATCH_SIZE",
    "Visibility",
]
