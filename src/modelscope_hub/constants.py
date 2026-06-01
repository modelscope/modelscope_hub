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


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean from the environment, falling back to ``default``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Network / IO tunables
# ---------------------------------------------------------------------------
API_TIMEOUT: int = _env_int("API_TIMEOUT", 60)
"""Default HTTP timeout (seconds) for API calls."""

API_CONNECT_TIMEOUT: int = _env_int("MODELSCOPE_API_CONNECT_TIMEOUT", 10)
"""Separate connect timeout (seconds) for API calls."""

API_MAX_RETRIES: int = _env_int("API_MAX_RETRIES", 5)
"""Maximum retry attempts for transient API failures."""

# ---------------------------------------------------------------------------
# Endpoint switching
# ---------------------------------------------------------------------------
ENV_MODELSCOPE_DOMAIN: str = "MODELSCOPE_DOMAIN"
"""Env var name: override the hub domain (bare domain or full URL)."""

ENV_PREFER_AI_SITE: str = "MODELSCOPE_PREFER_AI_SITE"
"""Env var name: when ``true``, prefer ``modelscope.ai`` over ``modelscope.cn``."""

DEFAULT_INTL_ENDPOINT: str = "https://www.modelscope.ai"
"""International site endpoint."""

# ---------------------------------------------------------------------------
# Download tunables
# ---------------------------------------------------------------------------
DOWNLOAD_CHUNK_SIZE: int = _env_int("DOWNLOAD_CHUNK_SIZE", 1024 * 1024)
"""Streaming chunk size for downloads (bytes). Defaults to 1 MiB."""

DOWNLOAD_PARALLEL_THRESHOLD_MB: int = _env_int(
    "MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB", 500
)
"""Files larger than this threshold (MB) may use parallel download."""

DOWNLOAD_PARALLELS: int = _env_int("MODELSCOPE_DOWNLOAD_PARALLELS", 1)
"""Number of parallel download streams. 1 means sequential (default)."""

DOWNLOAD_RETRY_TIMES: int = _env_int("DOWNLOAD_RETRY_TIMES", 5)
"""Per-file download retry count (with backoff)."""

DOWNLOAD_TIMEOUT: int = _env_int("DOWNLOAD_TIMEOUT", 60)
"""Per-file download timeout (seconds)."""

DOWNLOAD_PART_SIZE: int = _env_int(
    "DOWNLOAD_PART_SIZE", 160 * 1024 * 1024
)
"""Chunk size for parallel range download (bytes). Default 160 MiB."""

TEMPORARY_FOLDER_NAME: str = "._____temp"
"""Temporary folder name used during downloads."""

FILE_HASH_FIELD: str = "Sha256"
"""API response field name for file hash."""

ENV_FILE_LOCK: str = "MODELSCOPE_HUB_FILE_LOCK"
"""Env var name: enable/disable file lock for concurrent downloads (default true)."""

ENV_INTRA_CLOUD_ACCELERATION: str = "INTRA_CLOUD_ACCELERATION"
"""Env var name: enable Alibaba cloud intra-cloud download acceleration (default true)."""

ENV_INTRA_CLOUD_REGION: str = "INTRA_CLOUD_ACCELERATION_REGION"
"""Env var name: override the detected intra-cloud region ID."""

UPLOAD_LFS_THRESHOLD: int = _env_int("UPLOAD_LFS_THRESHOLD", 5 * 1024 * 1024)
"""Files larger than this threshold (bytes) must use the LFS upload path.

Default of 5 MiB aligns with the OpenAPI direct file-upload limit.
"""

# Upload: LFS enforcement threshold (suffix-independent)
UPLOAD_LFS_ENFORCE_THRESHOLD: int = _env_int(
    "UPLOAD_LFS_ENFORCE_THRESHOLD", 1 * 1024 * 1024
)

# Upload: blob retry
UPLOAD_BLOB_MAX_RETRIES: int = _env_int("UPLOAD_BLOB_MAX_RETRIES", 5)
UPLOAD_BLOB_RETRY_BACKOFF: int = _env_int("UPLOAD_BLOB_RETRY_BACKOFF", 2)
UPLOAD_BLOB_RETRY_MAX_WAIT: int = _env_int("UPLOAD_BLOB_RETRY_MAX_WAIT", 60)
UPLOAD_BLOB_TQDM_DISABLE_THRESHOLD: int = _env_int(
    "UPLOAD_BLOB_TQDM_DISABLE_THRESHOLD", 5 * 1024 * 1024
)

# Upload: blob timeout (connect, read) — separate from general API_TIMEOUT
# Per-socket-operation timeout, not total transfer time.
UPLOAD_BLOB_CONNECT_TIMEOUT: int = _env_int("UPLOAD_BLOB_CONNECT_TIMEOUT", 30)
UPLOAD_BLOB_READ_TIMEOUT: int = _env_int("UPLOAD_BLOB_READ_TIMEOUT", 3600)

# Upload: urllib3 retry — PUT excluded (streaming data can't be replayed)
UPLOAD_RETRY_ALLOWED_METHODS: frozenset[str] = frozenset(
    os.environ.get(
        "UPLOAD_RETRY_ALLOWED_METHODS", "GET,HEAD,DELETE,OPTIONS,TRACE"
    ).split(",")
)

# Upload: batching
UPLOAD_COMMIT_BATCH_SIZE: int = _env_int("UPLOAD_COMMIT_BATCH_SIZE", 256)
UPLOAD_ADAPTIVE_BATCH_SIZE: bool = _env_bool("UPLOAD_ADAPTIVE_BATCH_SIZE", True)
UPLOAD_VALIDATE_BLOB_BATCH_SIZE: int = _env_int(
    "UPLOAD_VALIDATE_BLOB_BATCH_SIZE", 64
)

# Upload: commit retry
UPLOAD_COMMIT_MAX_RETRIES: int = _env_int("UPLOAD_COMMIT_MAX_RETRIES", 5)

# Upload: failed file retry & ReAct
UPLOAD_FAILED_FILE_MAX_RETRIES: int = _env_int(
    "UPLOAD_FAILED_FILE_MAX_RETRIES", 3
)
UPLOAD_REACT_ENABLED: bool = _env_bool("UPLOAD_REACT_ENABLED", True)
UPLOAD_REACT_ROUND2_BASE_DELAY: int = _env_int(
    "UPLOAD_REACT_ROUND2_BASE_DELAY", 2
)
UPLOAD_REACT_ROUND3_FILE_DELAY: int = _env_int(
    "UPLOAD_REACT_ROUND3_FILE_DELAY", 5
)
UPLOAD_REACT_BACKOFF_MAX_EXPONENT: int = _env_int(
    "UPLOAD_REACT_BACKOFF_MAX_EXPONENT", 5
)
UPLOAD_REACT_MAX_DELAY: int = _env_int("UPLOAD_REACT_MAX_DELAY", 120)

# Upload: workers
DEFAULT_MAX_WORKERS: int = _env_int(
    "DEFAULT_MAX_WORKERS", min(8, (os.cpu_count() or 4) + 4)
)

# Upload: cache / tracker
UPLOAD_USE_CACHE: bool = _env_bool("UPLOAD_USE_CACHE", True)
UPLOAD_CACHE_FILE: str = ".ms_upload_cache"
UPLOAD_LEGACY_PROGRESS_FILE: str = ".ms_upload_progress"

# Upload: limits
UPLOAD_MAX_FILE_SIZE: int = _env_int(
    "UPLOAD_MAX_FILE_SIZE", 100 * 1024 * 1024 * 1024
)
UPLOAD_MAX_FILE_COUNT: int = _env_int("UPLOAD_MAX_FILE_COUNT", 100_000)
UPLOAD_MAX_FILE_COUNT_IN_DIR: int = _env_int("UPLOAD_MAX_FILE_COUNT_IN_DIR", 50_000)
"""Per-directory file count limit."""

UPLOAD_NORMAL_FILE_SIZE_TOTAL_LIMIT: int = _env_int(
    "UPLOAD_NORMAL_FILE_SIZE_TOTAL_LIMIT", 500 * 1024 * 1024
)
"""Total size limit for non-LFS (normal) files in a single commit (bytes)."""

# LFS suffix lists (from old SDK — determines upload mode regardless of size)
MODEL_LFS_SUFFIX: list[str] = [
    ".7z", ".arrow", ".bin", ".bz2", ".ckpt", ".ftz", ".gz", ".h5",
    ".joblib", ".mlmodel", ".model", ".msgpack", ".npy", ".npz", ".onnx",
    ".ot", ".parquet", ".pb", ".pickle", ".pkl", ".pt", ".pth", ".rar",
    ".safetensors", ".tar", ".tflite", ".tgz", ".wasm", ".xz", ".zip", ".zst",
]
DATASET_LFS_SUFFIX: list[str] = [
    ".7z", ".aac", ".arrow", ".audio", ".bmp", ".bin", ".bz2", ".flac",
    ".ftz", ".gif", ".gz", ".h5", ".jack", ".jpeg", ".jpg", ".png", ".jsonl",
    ".joblib", ".lz4", ".msgpack", ".npy", ".npz", ".ot", ".parquet", ".pb",
    ".pickle", ".pcm", ".pkl", ".raw", ".rar", ".sam", ".tar", ".tgz",
    ".wasm", ".wav", ".webm", ".webp", ".zip", ".zst", ".tiff", ".mp3",
    ".mp4", ".ogg",
]

# Default ignore patterns for folder upload
DEFAULT_IGNORE_PATTERNS: list[str] = [
    ".git", ".git/*", "*/.git", "**/.git/**",
    ".cache", ".cache/*", "*/.cache", "**/.cache/**",
]


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
DEFAULT_CACHE_DIR_NAME: str = "modelscope"
TOKEN_FILE_NAME: str = "token"
CONFIG_DIR_NAME: str = ".modelscope"


__all__ = [
    "API_CONNECT_TIMEOUT",
    "API_MAX_RETRIES",
    "API_TIMEOUT",
    "CONFIG_DIR_NAME",
    "DATASET_LFS_SUFFIX",
    "DEFAULT_CACHE_DIR_NAME",
    "DEFAULT_ENDPOINT",
    "DEFAULT_IGNORE_PATTERNS",
    "DEFAULT_INTL_ENDPOINT",
    "DEFAULT_MAX_WORKERS",
    "DOWNLOAD_CHUNK_SIZE",
    "DOWNLOAD_PARALLEL_THRESHOLD_MB",
    "DOWNLOAD_PARALLELS",
    "DOWNLOAD_PART_SIZE",
    "DOWNLOAD_RETRY_TIMES",
    "DOWNLOAD_TIMEOUT",
    "ENV_FILE_LOCK",
    "ENV_INTRA_CLOUD_ACCELERATION",
    "ENV_INTRA_CLOUD_REGION",
    "ENV_MODELSCOPE_DOMAIN",
    "ENV_PREFER_AI_SITE",
    "FILE_HASH_FIELD",
    "LEGACY_API_PREFIX",
    "License",
    "MODEL_LFS_SUFFIX",
    "OPENAPI_PREFIX",
    "RepoType",
    "StrEnum",
    "TEMPORARY_FOLDER_NAME",
    "TOKEN_FILE_NAME",
    "UPLOAD_ADAPTIVE_BATCH_SIZE",
    "UPLOAD_BLOB_CONNECT_TIMEOUT",
    "UPLOAD_BLOB_MAX_RETRIES",
    "UPLOAD_BLOB_READ_TIMEOUT",
    "UPLOAD_BLOB_RETRY_BACKOFF",
    "UPLOAD_BLOB_RETRY_MAX_WAIT",
    "UPLOAD_BLOB_TQDM_DISABLE_THRESHOLD",
    "UPLOAD_CACHE_FILE",
    "UPLOAD_COMMIT_BATCH_SIZE",
    "UPLOAD_COMMIT_MAX_RETRIES",
    "UPLOAD_FAILED_FILE_MAX_RETRIES",
    "UPLOAD_LEGACY_PROGRESS_FILE",
    "UPLOAD_LFS_ENFORCE_THRESHOLD",
    "UPLOAD_LFS_THRESHOLD",
    "UPLOAD_MAX_FILE_COUNT",
    "UPLOAD_MAX_FILE_COUNT_IN_DIR",
    "UPLOAD_MAX_FILE_SIZE",
    "UPLOAD_NORMAL_FILE_SIZE_TOTAL_LIMIT",
    "UPLOAD_REACT_BACKOFF_MAX_EXPONENT",
    "UPLOAD_REACT_ENABLED",
    "UPLOAD_REACT_MAX_DELAY",
    "UPLOAD_REACT_ROUND2_BASE_DELAY",
    "UPLOAD_REACT_ROUND3_FILE_DELAY",
    "UPLOAD_RETRY_ALLOWED_METHODS",
    "UPLOAD_USE_CACHE",
    "UPLOAD_VALIDATE_BLOB_BATCH_SIZE",
    "Visibility",
]
