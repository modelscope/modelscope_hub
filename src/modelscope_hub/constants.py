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
# Helpers for environment-driven overrides (auto-registering)
# ---------------------------------------------------------------------------
_REGISTERED_NAMES: set[str] = set()
_DEPRECATED_LOOKUP: dict[str, tuple[str, ...]] = {}


def _warn_deprecated_env(
    old: str,
    name: str,
    *,
    expects_mb: bool = False,
    stacklevel: int = 3,
) -> None:
    """Warn that a legacy environment variable remains temporarily supported."""
    message = (
        f"Environment variable {old!r} is deprecated and will be removed in a future version. Use {name!r} instead."
    )
    if expects_mb:
        message += f" {name!r} expects a value in MB."
    warnings.warn(message, FutureWarning, stacklevel=stacklevel)


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


def _env_int(
    name: str,
    default: int,
    description: str = "",
    category: str = "",
    *deprecated_names: str,
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
        return default
    return value if value > 0 else default


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
            return default_mb * 1024 * 1024
        return value * 1024 * 1024 if value > 0 else default_mb * 1024 * 1024
    # Fall back to deprecated names (value already in bytes)
    for old in all_deprecated:
        raw = os.environ.get(old)
        if raw is not None and raw.strip():
            _warn_deprecated_env(old, name, expects_mb=True, stacklevel=2)
            try:
                value = int(raw)
            except ValueError:
                return default_mb * 1024 * 1024
            return value if value > 0 else default_mb * 1024 * 1024
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
            return default_bytes
        return value * 1024 * 1024 if value > 0 else default_bytes

    for old in deprecated_mb_names:
        raw = os.environ.get(old)
        if raw is not None and raw.strip():
            _warn_deprecated_env(old, name, stacklevel=2)
            try:
                value = int(raw)
            except ValueError:
                return default_bytes
            return value * 1024 * 1024 if value > 0 else default_bytes

    for old in deprecated_byte_names:
        raw = os.environ.get(old)
        if raw is not None and raw.strip():
            _warn_deprecated_env(old, name, expects_mb=True, stacklevel=2)
            try:
                value = int(raw)
            except ValueError:
                return default_bytes
            return value if value > 0 else default_bytes

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
UPLOAD_LFS_FORCE_THRESHOLD_BYTES: int = _env_int_mb_with_deprecated_units(
    "MODELSCOPE_UPLOAD_LFS_FORCE_THRESHOLD_MB",
    1,
    "File-size threshold that forces LFS mode (MB)",
    "Upload",
    deprecated_byte_names=("UPLOAD_LFS_ENFORCE_THRESHOLD", "UPLOAD_SIZE_THRESHOLD_TO_ENFORCE_LFS"),
)
UPLOAD_MAX_FILE_SIZE_BYTES: int = _env_int_mb_with_deprecated_units(
    "MODELSCOPE_UPLOAD_MAX_FILE_SIZE_MB",
    100 * 1024,
    "Maximum single upload file size (MB, default 100 GB)",
    "Upload",
    deprecated_mb_names=("UPLOAD_MAX_FILE_SIZE_MB",),
    deprecated_byte_names=("UPLOAD_MAX_FILE_SIZE",),
)
UPLOAD_MAX_FILE_COUNT: int = _env_int(
    "MODELSCOPE_UPLOAD_MAX_FILE_COUNT",
    100_000,
    "Maximum total files per upload",
    "Upload",
    "UPLOAD_MAX_FILE_COUNT",
)
UPLOAD_MAX_FILES_PER_DIRECTORY: int = _env_int(
    "MODELSCOPE_UPLOAD_MAX_FILES_PER_DIRECTORY",
    50_000,
    "Maximum files in one uploaded directory",
    "Upload",
    "UPLOAD_MAX_FILE_COUNT_IN_DIR",
)
UPLOAD_NORMAL_FILES_TOTAL_SIZE_BYTES: int = _env_int_mb_with_deprecated_units(
    "MODELSCOPE_UPLOAD_NORMAL_FILES_TOTAL_SIZE_MB",
    500,
    "Maximum total size of normal (non-LFS) files (MB)",
    "Upload",
    deprecated_byte_names=("UPLOAD_NORMAL_FILE_SIZE_TOTAL_LIMIT",),
)

# Deprecated Python aliases. Runtime code must use the explicit names above.
UPLOAD_BLOB_CONNECT_TIMEOUT = UPLOAD_BLOB_CONNECT_TIMEOUT_SECONDS
UPLOAD_BLOB_READ_TIMEOUT = UPLOAD_BLOB_READ_TIMEOUT_SECONDS
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


__all__ = [
    "API_CONNECT_TIMEOUT",
    "API_MAX_RETRIES",
    "API_TIMEOUT",
    "CATEGORY_ORDER",
    "CONFIG_DIR_NAME",
    "DATASET_LFS_SUFFIX",
    "DEFAULT_CACHE_DIR_NAME",
    "DEFAULT_ENDPOINT",
    "DEFAULT_IGNORE_PATTERNS",
    "DEFAULT_INTL_ENDPOINT",
    "DEFAULT_MAX_WORKERS",
    "DOWNLOAD_CHUNK_SIZE",
    "DOWNLOAD_PARALLEL_THRESHOLD",
    "DOWNLOAD_PARALLELS",
    "DOWNLOAD_PART_SIZE",
    "DOWNLOAD_RETRY_TIMES",
    "DOWNLOAD_TIMEOUT",
    "ENV_FILE_LOCK",
    "ENV_CACHE",
    "ENV_INTRA_CLOUD_ACCELERATION",
    "ENV_INTRA_CLOUD_REGION",
    "ENV_INTER_CLOUD_REGIONS",
    "ENV_MODELSCOPE_DOMAIN",
    "ENV_PREFER_AI_SITE",
    "ENV_REGISTRY",
    "EnvVar",
    "FILE_HASH_FIELD",
    "get_upload_ignore_file_pattern",
    "LEGACY_API_PREFIX",
    "License",
    "MODEL_LFS_SUFFIX",
    "OPENAPI_PREFIX",
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
    "UPLOAD_BLOB_TQDM_DISABLE_THRESHOLD",
    "UPLOAD_BLOB_VALIDATION_BATCH_MAX_OBJECTS",
    "UPLOAD_CACHE_ENABLED",
    "UPLOAD_CACHE_FILE",
    "UPLOAD_COMMIT_BATCH_MAX_OPERATIONS",
    "UPLOAD_COMMIT_BATCH_SIZE",
    "UPLOAD_COMMIT_MAX_ATTEMPTS",
    "UPLOAD_COMMIT_MAX_CONSECUTIVE_FAILED_BATCHES",
    "UPLOAD_COMMIT_MAX_RETRIES",
    "UPLOAD_COMMIT_MAX_TOTAL_WAIT",
    "UPLOAD_COMMIT_RETRY_TOTAL_WAIT_SECONDS",
    "UPLOAD_FAILED_FILE_MAX_RETRIES",
    "UPLOAD_FAILED_FILE_MAX_RETRY_ROUNDS",
    "UPLOAD_HTTP_RETRY_ALLOWED_METHODS",
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
    "UPLOAD_USE_CACHE",
    "UPLOAD_VALIDATE_BLOB_BATCH_SIZE",
    "Visibility",
]
