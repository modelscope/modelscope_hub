"""Internal file upload implementation.

Strict behavioral parity with the old modelscope SDK upload pipeline:
- LFS detection by suffix list + size threshold
- Upload tracker for resumable uploads (.ms_upload_cache)
- Adaptive batch sizing
- Pipeline mode: ThreadPoolExecutor uploads + ordered batch commits
- Per-file retry with exponential backoff
- Per-commit retry
- ReAct progressive retry fallback (parallel → serial → single-file)
- Upload report
"""

from __future__ import annotations

import base64
import builtins as _builtins
import fnmatch
import hashlib
import io
import json
import os
import posixpath
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, BinaryIO

from tqdm.auto import tqdm

from .constants import (
    API_CONNECTION_POOL_MAXSIZE,
    COMMIT_MAX_ACTIONS_PER_REQUEST,
    DATASET_LFS_SUFFIX,
    DEFAULT_IGNORE_PATTERNS,
    MODEL_LFS_SUFFIX,
    UPLOAD_ADAPTIVE_BATCHING_ENABLED,
    UPLOAD_BLOB_MAX_ATTEMPTS,
    UPLOAD_BLOB_PROGRESS_THRESHOLD_BYTES,
    UPLOAD_BLOB_RETRY_BACKOFF_BASE_SECONDS,
    UPLOAD_BLOB_RETRY_MAX_DELAY_SECONDS,
    UPLOAD_BLOB_VALIDATION_BATCH_MAX_OBJECTS,
    UPLOAD_CACHE_ENABLED,
    UPLOAD_CACHE_FILE,
    UPLOAD_COMMIT_BATCH_MAX_OPERATIONS,
    UPLOAD_COMMIT_MAX_ATTEMPTS,
    UPLOAD_COMMIT_MAX_CONSECUTIVE_FAILED_BATCHES,
    UPLOAD_COMMIT_MAX_INLINE_BYTES,
    UPLOAD_COMMIT_MAX_PER_HOUR,
    UPLOAD_COMMIT_MAX_RETRY_AFTER_SECONDS,
    UPLOAD_COMMIT_RETRY_TOTAL_WAIT_SECONDS,
    UPLOAD_FAILED_FILE_MAX_RETRY_ROUNDS,
    UPLOAD_INLINE_METADATA_PATHS,
    UPLOAD_LEGACY_PROGRESS_FILE,
    UPLOAD_LFS_FORCE_THRESHOLD_BYTES,
    UPLOAD_MAX_CONCURRENT_WORKERS,
    UPLOAD_MAX_FILE_COUNT,
    UPLOAD_MAX_FILE_SIZE_BYTES,
    UPLOAD_MAX_FILES_PER_DIRECTORY,
    UPLOAD_NORMAL_FILES_TOTAL_SIZE_BYTES,
    UPLOAD_PROGRESS_MIN_INTERVAL_SECONDS,
    UPLOAD_RECOVERY_BACKOFF_MAX_EXPONENT,
    UPLOAD_RECOVERY_ENABLED,
    UPLOAD_RECOVERY_MAX_DELAY_SECONDS,
    UPLOAD_RECOVERY_SERIAL_BACKOFF_BASE_SECONDS,
    UPLOAD_RECOVERY_SINGLE_FILE_DELAY_SECONDS,
)
from .errors import (
    FileIntegrityError,
    HubError,
    InvalidParameter,
    NetworkError,
    RateLimitError,
    StorageError,
)
from .utils.file_utils import compute_hash
from .utils.logger import get_logger

if TYPE_CHECKING:
    from ._legacy_api import LegacyClient
    from ._openapi import OpenAPIClient
    from .config import HubConfig

logger = get_logger("upload")


PathOrFileObj = str | Path | bytes | BinaryIO | IO[bytes]

_TRACKER_VERSION = 3


class _DuplicateBlob:
    """Marker: an earlier file in this run uploads this exact content.

    Identical content hashes to one oid, and the batch pre-sign step hands every
    occurrence the same upload URL -- so without this marker each occurrence
    would PUT the same bytes again. The server only reports "already stored"
    once a blob has landed, which cannot help when all the pre-signing happens
    before any upload starts.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return "<duplicate blob, uploaded by an earlier file>"


DUPLICATE_BLOB = _DuplicateBlob()


# ====================================================================
# Helpers
# ====================================================================


class _CountedReadStream:
    """File wrapper that counts bytes read and updates a progress bar."""

    def __init__(self, file_obj: Any, expected_size: int, pbar: Any, chunk_size: int) -> None:
        self._file = file_obj
        self._expected_size = expected_size
        self._pbar = pbar
        self._chunk_size = chunk_size
        self._bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        read_size = self._chunk_size if size < 0 else min(size, self._chunk_size)
        chunk = self._file.read(read_size)
        if chunk:
            n = len(chunk)
            self._bytes_read += n
            self._pbar.update(n)
        return chunk

    @property
    def bytes_read(self) -> int:
        return self._bytes_read

    def verify_complete(self) -> None:
        if self._bytes_read != self._expected_size:
            raise FileIntegrityError(
                f"Upload data incomplete: read {self._bytes_read} bytes, "
                f"expected {self._expected_size} bytes. "
                f"File may have been modified during upload."
            )


def _is_inline_metadata(path: str | Path) -> bool:
    """Return whether *path* must stay inline in the commit rather than go to LFS.

    Checked before the size and suffix rules so that lowering the LFS threshold
    can never turn a repository file the Hub parses server-side into a pointer.
    """
    if not isinstance(path, (str, Path)):
        return False
    return Path(path).name.upper() in UPLOAD_INLINE_METADATA_PATHS


def _is_lfs(path: str | Path, size: int, repo_type: str) -> bool:
    """Determine if a file should use LFS upload mode (suffix + size threshold)."""
    if _is_inline_metadata(path):
        return False
    if size > UPLOAD_LFS_FORCE_THRESHOLD_BYTES:
        return True
    suffix = Path(path).suffix.lower() if isinstance(path, (str, Path)) else ""
    if repo_type == "model":
        return suffix in MODEL_LFS_SUFFIX
    if repo_type == "dataset":
        return suffix in DATASET_LFS_SUFFIX
    return size > UPLOAD_LFS_FORCE_THRESHOLD_BYTES


def _upload_mode(path: str | Path, size: int, repo_type: str) -> str:
    """Return the commit/upload mode for a file."""
    return "lfs" if _is_lfs(path, size, repo_type) else "normal"


def _calculate_adaptive_batch_size(total_files: int, max_operations: int) -> int:
    """Return the operation target, always below the server's hard ceiling.

    File count is the primary constraint because repository publishing is more
    sensitive to action count than to blob volume. Request bytes are enforced
    independently by :func:`_plan_commit_batches`.
    """
    if total_files <= 0:
        return 1
    ceiling = max(1, COMMIT_MAX_ACTIONS_PER_REQUEST)
    cap = max_operations if max_operations > 0 else total_files
    return max(1, min(cap, ceiling, total_files))


_COMMIT_ENVELOPE_ESTIMATED_BYTES = 512
_COMMIT_OPERATION_ESTIMATED_BYTES = 192
_LFS_POINTER_ESTIMATED_BYTES = 128


def _estimate_commit_operation_bytes(path_in_repo: str, size: int, repo_type: str) -> int:
    """Estimate the JSON bytes one create operation contributes to a commit."""
    path_bytes = len(path_in_repo.encode("utf-8"))
    scalar_bytes = len(str(max(0, size)))
    if _is_lfs(path_in_repo, size, repo_type):
        return _COMMIT_OPERATION_ESTIMATED_BYTES + _LFS_POINTER_ESTIMATED_BYTES + path_bytes + scalar_bytes
    encoded_bytes = (max(0, size) + 2) // 3 * 4
    return _COMMIT_OPERATION_ESTIMATED_BYTES + path_bytes + scalar_bytes + encoded_bytes


def _plan_commit_batches(
    files: list[tuple[str, str]],
    repo_type: str,
    *,
    max_operations: int,
    max_inline_bytes: int,
    sizes: dict[str, int] | None = None,
) -> list[int]:
    """Split files by action count first and estimated request bytes second.

    Normal content is counted after base64 expansion. LFS blob bytes are not in
    the commit request, but their pointer/action JSON still consumes a small
    amount of space. A single oversized inline operation is allowed to make
    progress and emits a warning. A tiny final count-only batch is rebalanced
    with its predecessor when both balanced batches still fit all constraints.
    """
    if not files:
        return []
    cap = max_operations if max_operations > 0 else len(files)
    cap = max(1, min(cap, max(1, COMMIT_MAX_ACTIONS_PER_REQUEST)))
    weights: list[int] = []
    for path_in_repo, file_path in files:
        size = sizes.get(file_path, 0) if sizes is not None else _safe_size(file_path)
        is_lfs = _is_lfs(path_in_repo, size, repo_type)
        weight = _estimate_commit_operation_bytes(path_in_repo, size, repo_type)
        weights.append(weight)
        if not is_lfs and max_inline_bytes > 0 and _COMMIT_ENVELOPE_ESTIMATED_BYTES + weight > max_inline_bytes:
            logger.warning(
                "Inline file %s alone exceeds the advisory commit request budget (%d > %d bytes); "
                "sending it in a dedicated commit.",
                path_in_repo,
                _COMMIT_ENVELOPE_ESTIMATED_BYTES + weight,
                max_inline_bytes,
            )

    batch_indexes: list[list[int]] = []
    current: list[int] = []
    request_bytes = _COMMIT_ENVELOPE_ESTIMATED_BYTES
    for index, weight in enumerate(weights):
        exceeds_bytes = max_inline_bytes > 0 and request_bytes + weight > max_inline_bytes
        if current and (len(current) >= cap or exceeds_bytes):
            batch_indexes.append(current)
            current = []
            request_bytes = _COMMIT_ENVELOPE_ESTIMATED_BYTES
        current.append(index)
        request_bytes += weight
    if current:
        batch_indexes.append(current)

    if len(batch_indexes) >= 2 and len(batch_indexes[-1]) * 4 < len(batch_indexes[-2]):
        combined = batch_indexes[-2] + batch_indexes[-1]
        midpoint = (len(combined) + 1) // 2
        candidates = (combined[:midpoint], combined[midpoint:])

        def _fits(candidate: list[int]) -> bool:
            estimated = _COMMIT_ENVELOPE_ESTIMATED_BYTES + sum(weights[i] for i in candidate)
            return len(candidate) <= cap and (max_inline_bytes <= 0 or estimated <= max_inline_bytes)

        if all(_fits(candidate) for candidate in candidates):
            batch_indexes[-2:] = [list(candidates[0]), list(candidates[1])]

    return [len(batch) for batch in batch_indexes]


def _safe_size(file_path: str) -> int:
    try:
        return os.stat(file_path).st_size
    except OSError:
        return 0


def _warn_advisory_upload_limits(
    files: list[tuple[str, str]],
    sizes: dict[str, int],
    repo_type: str,
) -> None:
    """Warn about advisory scale limits without refusing a valid upload."""
    if len(files) > UPLOAD_MAX_FILE_COUNT:
        logger.warning(
            "Upload contains %d files, above the advisory limit of %d; continuing.",
            len(files),
            UPLOAD_MAX_FILE_COUNT,
        )

    directory_counts: dict[str, int] = {}
    oversized: list[tuple[str, int]] = []
    normal_size = 0
    for path_in_repo, file_path in files:
        directory = posixpath.dirname(path_in_repo)
        directory_counts[directory] = directory_counts.get(directory, 0) + 1
        size = sizes.get(file_path, 0)
        if size > UPLOAD_MAX_FILE_SIZE_BYTES:
            oversized.append((path_in_repo, size))
        if not _is_lfs(path_in_repo, size, repo_type):
            normal_size += size

    crowded = [
        (path or "/", count) for path, count in directory_counts.items() if count > UPLOAD_MAX_FILES_PER_DIRECTORY
    ]
    if crowded:
        preview = ", ".join(f"{path} ({count})" for path, count in crowded[:3])
        suffix = f" and {len(crowded) - 3} more" if len(crowded) > 3 else ""
        logger.warning(
            "Upload has directories above the advisory %d-file limit: %s%s; continuing.",
            UPLOAD_MAX_FILES_PER_DIRECTORY,
            preview,
            suffix,
        )
    if oversized:
        preview = ", ".join(f"{path} ({size} bytes)" for path, size in oversized[:3])
        suffix = f" and {len(oversized) - 3} more" if len(oversized) > 3 else ""
        logger.warning(
            "Upload has files above the advisory %d-byte single-file limit: %s%s; continuing.",
            UPLOAD_MAX_FILE_SIZE_BYTES,
            preview,
            suffix,
        )
    if normal_size > UPLOAD_NORMAL_FILES_TOTAL_SIZE_BYTES:
        logger.warning(
            "Total normal (non-LFS) content is %d bytes, above the advisory limit of %d; continuing. "
            "Consider lowering MODELSCOPE_UPLOAD_LFS_FORCE_THRESHOLD.",
            normal_size,
            UPLOAD_NORMAL_FILES_TOTAL_SIZE_BYTES,
        )


def _normalize_path_in_repo(path_in_repo: str | None) -> str:
    """Collapse a repo destination prefix to a clean, root-relative form.

    ``"."``, ``"./"``, ``""`` and ``"/"`` all denote the repository root and
    must yield no prefix. Left literal, a value like ``"."`` becomes a ``"./"``
    prefix on every file and rides into each commit action's ``path``, which the
    Hub rejects wholesale as an invalid commit action (E3021). Separators are
    normalized and ``.``/``..`` segments resolved; a path that escapes the root
    is refused rather than silently rewritten.
    """
    if not path_in_repo:
        return ""
    cleaned = posixpath.normpath(path_in_repo.strip().replace("\\", "/")).strip("/")
    if cleaned in ("", "."):
        return ""
    if cleaned == ".." or cleaned.startswith("../"):
        raise InvalidParameter(f"path_in_repo must stay within the repository root, got {path_in_repo!r}")
    return cleaned


def _compute_file_hash(
    file_path_or_obj: str | Path | bytes | BinaryIO | IO[bytes],
    buffer_size_mb: int = 16,
) -> dict:
    """Compute SHA256 hash and size for a file, bytes, or file-like object.

    Returns dict with 'file_hash', 'file_size', 'file_path_or_obj' keys.
    """
    if isinstance(file_path_or_obj, bytes):
        return {
            "file_path_or_obj": file_path_or_obj,
            "file_hash": hashlib.sha256(file_path_or_obj).hexdigest(),
            "file_size": len(file_path_or_obj),
        }
    if isinstance(file_path_or_obj, (str, Path)):
        path = Path(file_path_or_obj)
        file_size = path.stat().st_size
        file_hash = compute_hash(path, "sha256")
        return {
            "file_path_or_obj": str(file_path_or_obj),
            "file_hash": file_hash,
            "file_size": file_size,
        }
    # BinaryIO / file-like object: read into bytes
    data = file_path_or_obj.read()
    return {
        "file_path_or_obj": data,
        "file_hash": hashlib.sha256(data).hexdigest(),
        "file_size": len(data),
    }


def _matches_patterns(path: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return False
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def _filter_repo_objects(
    items: list[str],
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> list[str]:
    """Filter file paths using fnmatch allow/ignore pattern lists."""
    filtered = []
    for item in items:
        if allow_patterns and not _matches_patterns(item, allow_patterns):
            continue
        if ignore_patterns and _matches_patterns(item, ignore_patterns):
            continue
        filtered.append(item)
    return filtered


class _ErrorCategory:
    TRANSIENT_NETWORK = "transient_network"
    TRANSIENT_SERVER = "transient_server"
    THROTTLED = "throttled"
    AUTH_FAILED = "auth_failed"
    NOT_FOUND = "not_found"
    FILE_INVALID = "file_invalid"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"

    _NON_RETRYABLE = {"auth_failed", "not_found", "file_invalid", "permanent"}

    @classmethod
    def is_retryable(cls, category: str) -> bool:
        return category not in cls._NON_RETRYABLE


_CATEGORY_BY_ERROR_CODE: dict[str, str] = {
    "E1001": _ErrorCategory.TRANSIENT_NETWORK,  # timeout
    "E1002": _ErrorCategory.TRANSIENT_SERVER,  # server error
    "E1003": _ErrorCategory.TRANSIENT_SERVER,  # storage error
    "E1020": _ErrorCategory.TRANSIENT_NETWORK,  # network/connection error
    "E1021": _ErrorCategory.THROTTLED,  # rate limit
    "E1022": _ErrorCategory.FILE_INVALID,  # cache error
    "E2020": _ErrorCategory.TRANSIENT_SERVER,  # file integrity (auto-retry)
    "E3001": _ErrorCategory.AUTH_FAILED,  # authentication
    "E3002": _ErrorCategory.AUTH_FAILED,  # permission
    "E3020": _ErrorCategory.NOT_FOUND,  # not exist
    "E3021": _ErrorCategory.FILE_INVALID,  # invalid parameter
    "E3023": _ErrorCategory.FILE_INVALID,  # not supported
    "E9001": _ErrorCategory.UNKNOWN,  # unknown/fallback
}


_RETRYABLE_COMMIT_BUSINESS_CODES = frozenset({10030000001})
_RETRYABLE_COMMIT_MESSAGE_MARKERS = (
    "branch changed",
    "could not update refs",
    "please retry",
    "try again",
    "commit rejected by repository policy",
)


def _is_retryable_commit_error(error: Exception) -> bool:
    """Return whether a commit failure is transient in commit context."""
    if isinstance(error, HubError) and error.retryable:
        return True
    body = getattr(error, "response_body", None)
    if isinstance(body, dict):
        raw_code = body.get("Code") if body.get("Code") is not None else body.get("code")
        try:
            if raw_code is not None and int(raw_code) in _RETRYABLE_COMMIT_BUSINESS_CODES:
                return True
        except (TypeError, ValueError):
            pass
    message = getattr(error, "message", None) or str(error)
    lowered = str(message).lower()
    status_code = getattr(error, "status_code", None)
    return status_code in (400, 409) and any(marker in lowered for marker in _RETRYABLE_COMMIT_MESSAGE_MARKERS)


def classify_error(error: Exception, *, commit_context: bool = False) -> str:
    """Classify an exception for retry strategy using the SDK error hierarchy."""
    if commit_context and _is_retryable_commit_error(error):
        return _ErrorCategory.TRANSIENT_SERVER
    if isinstance(error, HubError):
        code = getattr(error, "error_code", None)
        if code and code in _CATEGORY_BY_ERROR_CODE:
            return _CATEGORY_BY_ERROR_CODE[code]
        return _ErrorCategory.UNKNOWN if error.retryable else _ErrorCategory.PERMANENT

    if isinstance(error, FileNotFoundError):
        return _ErrorCategory.FILE_INVALID
    if isinstance(error, _builtins.PermissionError):
        return _ErrorCategory.FILE_INVALID
    if isinstance(error, (ConnectionError, TimeoutError)):
        return _ErrorCategory.TRANSIENT_NETWORK
    if isinstance(error, (IOError, OSError)):
        error_str = str(error).lower()
        if "size changed" in error_str or "no such file" in error_str:
            return _ErrorCategory.FILE_INVALID
        if "permission" in error_str or "access denied" in error_str:
            return _ErrorCategory.FILE_INVALID
        return _ErrorCategory.TRANSIENT_NETWORK

    return _ErrorCategory.UNKNOWN


# ====================================================================
# Upload Tracker
# ====================================================================


class FileStatus:
    UPLOADED = "u"
    COMMITTED = "c"
    FAILED = "f"


class UploadTracker:
    """Persistent JSON cache at {folder}/.ms_upload_cache for resumable uploads."""

    def __init__(self, cache_path: str | Path, repo_id: str) -> None:
        self._path = Path(cache_path)
        self._repo_id = repo_id
        self._files: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._dirty = False
        self._load()

    @staticmethod
    def _make_key(rel_path: str, mtime: float, size: int) -> str:
        return f"{rel_path}|{mtime}|{size}"

    def get_hash(self, rel_path: str, mtime: float, size: int) -> dict | None:
        key = self._make_key(rel_path, mtime, size)
        with self._lock:
            entry = self._files.get(key)
        if entry is None or "hash" not in entry:
            return None
        return {
            "file_path_or_obj": rel_path,
            "file_hash": entry["hash"],
            "file_size": entry["size"],
        }

    def put_hash(self, rel_path: str, mtime: float, size: int, hash_info: dict) -> None:
        key = self._make_key(rel_path, mtime, size)
        with self._lock:
            entry = self._files.get(key, {})
            entry["hash"] = hash_info["file_hash"]
            entry["size"] = hash_info["file_size"]
            self._files[key] = entry
            self._dirty = True

    def is_committed(self, rel_path: str, mtime: float, size: int) -> bool:
        key = self._make_key(rel_path, mtime, size)
        with self._lock:
            entry = self._files.get(key)
        return entry is not None and entry.get("status") == FileStatus.COMMITTED

    def get_status(self, rel_path: str, mtime: float, size: int) -> str | None:
        key = self._make_key(rel_path, mtime, size)
        with self._lock:
            entry = self._files.get(key)
        return entry.get("status") if entry else None

    def begin_attempt(self, rel_path: str, mtime: float, size: int) -> None:
        """Clear stale failure metadata when a later run retries a file."""
        key = self._make_key(rel_path, mtime, size)
        with self._lock:
            entry = self._files.get(key)
            if entry is None or entry.get("status") == FileStatus.COMMITTED:
                return
            if entry.get("status") == FileStatus.FAILED:
                entry.pop("status", None)
            entry.pop("error_type", None)
            self._dirty = True

    def mark_uploaded(self, rel_path: str, mtime: float, size: int) -> None:
        key = self._make_key(rel_path, mtime, size)
        with self._lock:
            if key in self._files:
                self._files[key]["status"] = FileStatus.UPLOADED
                self._files[key].pop("error_type", None)
                self._dirty = True

    def mark_committed_batch(self, file_keys: list[tuple[str, float, int]]) -> None:
        with self._lock:
            for rel_path, mtime, size in file_keys:
                key = self._make_key(rel_path, mtime, size)
                if key in self._files:
                    self._files[key]["status"] = FileStatus.COMMITTED
                    self._files[key].pop("error_type", None)
            self._dirty = True

    def mark_failed(self, rel_path: str, mtime: float, size: int, error_type: str = "") -> None:
        key = self._make_key(rel_path, mtime, size)
        with self._lock:
            if key in self._files:
                self._files[key]["status"] = FileStatus.FAILED
                if error_type:
                    self._files[key]["error_type"] = error_type
            else:
                entry: dict[str, Any] = {"status": FileStatus.FAILED}
                if error_type:
                    entry["error_type"] = error_type
                self._files[key] = entry
            self._dirty = True

    def save(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            data = {
                "version": _TRACKER_VERSION,
                "repo_id": self._repo_id,
                "files": {k: dict(v) for k, v in self._files.items()},
            }
            self._dirty = False
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=str(self._path.parent), suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                os.replace(tmp_path, str(self._path))
            except BaseException:
                os.unlink(tmp_path)
                raise
        except Exception as e:
            logger.warning("Failed to save upload tracker: %s", e)

    def clear(self) -> None:
        try:
            self._path.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Failed to delete tracker file: %s", e)
        with self._lock:
            self._files.clear()
            self._dirty = False

    def _load(self) -> None:
        if not self._path.exists():
            self._check_legacy_progress()
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load upload tracker, starting fresh: %s", e)
            return

        version = data.get("version")
        if version is None:
            self._migrate_v1(data)
            return

        stored_repo = data.get("repo_id", "")
        if stored_repo and stored_repo != self._repo_id:
            logger.warning(
                "Tracker repo_id mismatch (cached: %s, current: %s), ignoring stale tracker.",
                stored_repo,
                self._repo_id,
            )
            return

        self._files = data.get("files", {})
        committed_count = sum(1 for e in self._files.values() if e.get("status") == FileStatus.COMMITTED)
        if committed_count > 0:
            logger.info(
                "Upload tracker loaded: %d entries, %d committed.",
                len(self._files),
                committed_count,
            )
        self._check_legacy_progress()

    def _migrate_v1(self, data: dict) -> None:
        migrated = {}
        for key, value in data.items():
            if isinstance(value, dict) and "file_hash" in value:
                migrated[key] = {
                    "hash": value["file_hash"],
                    "size": value.get("file_size", 0),
                }
        self._files = migrated
        self._dirty = True
        if migrated:
            logger.info(
                "Migrated %d entries from legacy hash cache format.",
                len(migrated),
            )

    def _check_legacy_progress(self) -> None:
        legacy_path = self._path.parent / UPLOAD_LEGACY_PROGRESS_FILE
        if legacy_path.exists():
            logger.warning(
                "Legacy upload progress file detected: %s. This file is no longer used. You may delete it safely.",
                legacy_path,
            )


class NullTracker:
    """No-op tracker for when caching is disabled."""

    def get_hash(self, rel_path: str, mtime: float, size: int) -> None:
        return None

    def put_hash(self, rel_path: str, mtime: float, size: int, hash_info: dict) -> None:
        pass

    def is_committed(self, rel_path: str, mtime: float, size: int) -> bool:
        return False

    def get_status(self, rel_path: str, mtime: float, size: int) -> None:
        return None

    def begin_attempt(self, rel_path: str, mtime: float, size: int) -> None:
        pass

    def mark_uploaded(self, rel_path: str, mtime: float, size: int) -> None:
        pass

    def mark_committed_batch(self, file_keys: list) -> None:
        pass

    def mark_failed(self, rel_path: str, mtime: float, size: int, error_type: str = "") -> None:
        pass

    def save(self) -> None:
        pass

    def clear(self) -> None:
        pass


# ====================================================================
# Batch Tracker
# ====================================================================


class BatchTracker:
    """Thread-safe tracker for pre-assigned upload batches.

    Batch sizes are supplied as a plan rather than a single number so that a
    batch can be closed on inlined-content volume as well as on file count.
    """

    def __init__(self, total_files: int, batch_sizes: list[int] | int) -> None:
        if isinstance(batch_sizes, int):
            step = max(1, batch_sizes)
            sizes = [min(step, total_files - start) for start in range(0, total_files, step)]
        else:
            sizes = [size for size in batch_sizes if size > 0]
        assigned = sum(sizes)
        if assigned < total_files:
            # Never drop files: a short plan gets the remainder as a final batch.
            sizes.append(total_files - assigned)
        self._batch_sizes = sizes
        self._num_batches = len(sizes)

        # file index -> batch index, so a completed upload can find its batch
        # without assuming batches are uniform.
        self._owner: list[int] = []
        self._batch_start: list[int] = []
        offset = 0
        for batch_idx, size in enumerate(sizes):
            self._batch_start.append(offset)
            self._owner.extend([batch_idx] * size)
            offset += size

        self._batch_results: list[list[dict]] = [[] for _ in range(self._num_batches)]
        self._batch_failures: list[list[tuple]] = [[] for _ in range(self._num_batches)]
        self._batch_expected: list[int] = list(sizes)
        self._batch_events: list[threading.Event] = [threading.Event() for _ in range(self._num_batches)]
        self._lock = threading.Lock()

    @property
    def num_batches(self) -> int:
        return self._num_batches

    def batch_range(self, batch_idx: int) -> tuple[int, int]:
        """Return the ``[start, end)`` file-index range owned by *batch_idx*."""
        start = self._batch_start[batch_idx]
        return start, start + self._batch_sizes[batch_idx]

    def batch_index(self, file_index: int) -> int:
        return self._owner[file_index]

    def record_success(self, file_index: int, result: dict) -> None:
        idx = self.batch_index(file_index)
        with self._lock:
            self._batch_results[idx].append(result)
            if self._is_batch_complete(idx):
                self._batch_events[idx].set()

    def record_failure(self, file_index: int, item: tuple, error: Exception) -> None:
        idx = self.batch_index(file_index)
        with self._lock:
            self._batch_failures[idx].append((item, error))
            if self._is_batch_complete(idx):
                self._batch_events[idx].set()

    def mark_file_skipped(self, file_index: int) -> None:
        idx = self.batch_index(file_index)
        with self._lock:
            self._batch_expected[idx] -= 1
            if self._is_batch_complete(idx):
                self._batch_events[idx].set()

    def wait_for_batch(self, batch_idx: int) -> tuple[list[dict], list[tuple]]:
        self._batch_events[batch_idx].wait()
        with self._lock:
            return (
                list(self._batch_results[batch_idx]),
                list(self._batch_failures[batch_idx]),
            )

    def _is_batch_complete(self, batch_idx: int) -> bool:
        count = len(self._batch_results[batch_idx]) + len(self._batch_failures[batch_idx])
        return count >= self._batch_expected[batch_idx]


class _CommitRateGovernor:
    """Sliding-window limiter that keeps commits under a per-hour budget.

    Reacting to a throttle costs a wasted round trip, and when the server holds
    the connection open instead of answering, a full read timeout. Pacing ahead
    of the limit avoids both. Disabled when the budget is not positive, so it is
    inert for interactive uploads that never approach the ceiling.
    """

    _WINDOW_SECONDS = 3600.0

    def __init__(self, max_per_hour: int) -> None:
        self._max_per_hour = max_per_hour
        self._timestamps: list[float] = []
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._max_per_hour > 0

    def acquire(self) -> float:
        """Block until a commit slot is free; return the seconds spent waiting."""
        if not self.enabled:
            return 0.0
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - self._WINDOW_SECONDS
                self._timestamps = [ts for ts in self._timestamps if ts > cutoff]
                if len(self._timestamps) < self._max_per_hour:
                    self._timestamps.append(now)
                    return waited
                sleep_for = self._timestamps[0] - cutoff
            logger.info(
                "Commit budget reached (%d/hour), pausing %.0fs before the next commit ...",
                self._max_per_hour,
                sleep_for,
            )
            time.sleep(max(sleep_for, 0.1))
            waited += max(sleep_for, 0.1)


# ====================================================================
# Upload Manager
# ====================================================================


class UploadManager:
    """Internal file upload implementation with production-grade retry and resume."""

    def __init__(
        self,
        legacy_client: LegacyClient,
        config: HubConfig,
        openapi_client: OpenAPIClient | None = None,
        *,
        create_repo_fn: Any = None,
    ) -> None:
        self._client = legacy_client
        self._config = config
        self._openapi = openapi_client
        self._create_repo_fn = create_repo_fn
        # The commit budget is a server-side property of the repository, not of
        # one call, so the governor is shared by every commit this manager makes
        # -- batch commits, recovery rounds, sync deletes and single files alike.
        # Pacing only the happy path would leave recovery free to hammer a
        # server that is already throttling.
        self._commit_governor = _CommitRateGovernor(UPLOAD_COMMIT_MAX_PER_HOUR)

    # ------------------------------------------------------------------
    # Public: upload_file
    # ------------------------------------------------------------------
    def upload_file(
        self,
        repo_id: str,
        repo_type: str,
        path_or_fileobj: PathOrFileObj,
        path_in_repo: str,
        *,
        commit_message: str = "Upload file",
        commit_description: str | None = None,
        revision: str = "master",
        buffer_size_mb: int = 16,
        disable_tqdm: bool = False,
    ) -> dict:
        """Upload a single file to a repository."""
        if path_or_fileobj is None:
            raise InvalidParameter("Path or file object cannot be None!")

        path_in_repo = _normalize_path_in_repo(path_in_repo)

        if isinstance(path_or_fileobj, (str, Path)):
            path_or_fileobj = os.path.abspath(os.path.expanduser(str(path_or_fileobj)))
            path_in_repo = path_in_repo or os.path.basename(path_or_fileobj)
        else:
            if not path_in_repo:
                raise InvalidParameter("Arg `path_in_repo` cannot be empty!")

        hash_info = _compute_file_hash(path_or_fileobj, buffer_size_mb)
        file_hash = hash_info["file_hash"]
        file_size = hash_info["file_size"]
        warning_source = str(path_or_fileobj) if isinstance(path_or_fileobj, (str, Path)) else path_in_repo
        _warn_advisory_upload_limits(
            [(path_in_repo, warning_source)],
            {warning_source: file_size},
            repo_type,
        )
        # If BinaryIO was consumed, _compute_file_hash returns the bytes
        if not isinstance(path_or_fileobj, (str, Path, bytes)):
            path_or_fileobj = hash_info["file_path_or_obj"]

        commit_message = commit_message or f"Upload {path_in_repo} to ModelScope hub"

        if self._create_repo_fn is not None:
            self._create_repo_fn(repo_id, repo_type)

        upload_mode = _upload_mode(path_in_repo, file_size, repo_type)
        if upload_mode == "lfs":
            upload_res = self._upload_blob(
                repo_id=repo_id,
                repo_type=repo_type,
                sha256=file_hash,
                size=file_size,
                data=path_or_fileobj,
                disable_tqdm=disable_tqdm,
                tqdm_desc=f"[Uploading] {path_in_repo}",
                buffer_size_mb=buffer_size_mb,
            )
        else:
            upload_res = {
                "url": None,
                "is_uploaded": True,
                "is_reused": False,
                "is_blob_uploaded": False,
            }

        operation = self._build_operation(
            path_in_repo=path_in_repo,
            path_or_fileobj=path_or_fileobj,
            hash_info=hash_info,
            upload_mode=upload_mode,
            is_uploaded=upload_res["is_uploaded"],
        )

        print(f"Committing file to {repo_id} ...", flush=True)
        # Same commit path as folder uploads: a single file gets the transient
        # retry and the Retry-After handling too. Committing directly meant a
        # throttled or briefly unavailable server failed the call outright.
        return self._commit_with_retry(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=[operation],
            commit_message=commit_message,
            revision=revision,
        )

    # ------------------------------------------------------------------
    # Public: delete_files
    # ------------------------------------------------------------------
    def delete_files(
        self,
        repo_id: str,
        repo_type: str,
        file_paths: list[str],
        *,
        commit_message: str = "Delete files",
        revision: str = "master",
    ) -> dict:
        """Delete repository files through a commit operation.

        The direct repository DELETE endpoints reject API-token authentication.
        Commit ``delete`` actions use the same supported write path as uploads.

        A single commit is capped by the server at
        :data:`COMMIT_MAX_ACTIONS_PER_REQUEST` actions, so a larger request is
        split across sequential commits. Deletions within one commit are atomic;
        across a split they are not, and a failure part-way leaves the earlier
        commits applied -- which is reported rather than hidden, so the caller can
        retry with the remaining paths.
        """
        paths = list(dict.fromkeys(path for path in file_paths if path))
        if not paths:
            raise InvalidParameter("file_paths must contain at least one non-empty path.")

        chunk_size = max(1, COMMIT_MAX_ACTIONS_PER_REQUEST)
        chunks = [paths[i : i + chunk_size] for i in range(0, len(paths), chunk_size)]
        if len(chunks) > 1:
            logger.info(
                "Deleting %d file(s) in %d commit(s) (server caps one commit at %d actions).",
                len(paths),
                len(chunks),
                chunk_size,
            )

        deleted: list[str] = []
        for index, chunk in enumerate(chunks):
            message = commit_message if len(chunks) == 1 else f"{commit_message} ({index + 1}/{len(chunks)})"
            try:
                self._commit_with_retry(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    operations=self._build_delete_operations(chunk),
                    commit_message=message,
                    revision=revision,
                )
            except Exception:
                if deleted:
                    logger.error(
                        "Delete commit %d/%d failed after %d file(s) were already removed.",
                        index + 1,
                        len(chunks),
                        len(deleted),
                    )
                raise
            deleted.extend(chunk)

        return {
            "deleted_files": deleted,
            "failed_files": [],
            "total_files": len(deleted),
        }

    # ------------------------------------------------------------------
    # Public: upload_folder
    # ------------------------------------------------------------------
    def upload_folder(
        self,
        repo_id: str,
        repo_type: str,
        folder_path: str | Path,
        *,
        path_in_repo: str = "",
        commit_message: str | None = None,
        commit_description: str | None = None,
        revision: str = "master",
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        max_workers: int | None = None,
        use_cache: bool | None = None,
        disable_tqdm: bool = False,
        sync_remote_repo: bool = False,
        tracker_path: str | Path | None = None,
        progress_callback: Any = None,
    ) -> dict | list[dict] | None:
        """Upload a folder with resumable support, adaptive batching, and retry."""
        start_time = time.time()

        if not repo_id:
            raise InvalidParameter("The arg `repo_id` cannot be empty!")
        if folder_path is None:
            raise InvalidParameter("The arg `folder_path` cannot be None!")

        if max_workers is None:
            max_workers = UPLOAD_MAX_CONCURRENT_WORKERS
        if use_cache is None:
            use_cache = UPLOAD_CACHE_ENABLED

        # Normalize patterns
        allow_patterns = allow_patterns or None
        if ignore_patterns is None:
            ignore_patterns = []
        elif isinstance(ignore_patterns, str):
            ignore_patterns = [ignore_patterns]
        else:
            ignore_patterns = list(ignore_patterns)
        ignore_patterns += DEFAULT_IGNORE_PATTERNS

        if allow_patterns is not None:
            ignore_patterns = [p for p in ignore_patterns if p not in allow_patterns]

        commit_message = commit_message if commit_message is not None else f"Upload to {repo_id} on ModelScope hub"
        commit_description = commit_description or "Uploading files"

        # Exclude internal cache files from upload
        _internal_files = [UPLOAD_CACHE_FILE, UPLOAD_LEGACY_PROGRESS_FILE]
        _internal_ignore = [p for f in _internal_files for p in (f, f"*/{f}")]
        ignore_patterns = ignore_patterns + _internal_ignore

        # Collect files
        logger.info("Preparing files to upload ...")
        file_sizes: dict[str, int] = {}
        sorted_files = self._prepare_upload_folder(
            folder_path=folder_path,
            path_in_repo=path_in_repo,
            repo_type=repo_type,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            sizes_out=file_sizes,
        )

        # For sync mode: collect ALL local files (unfiltered) to avoid
        # treating pattern-excluded files as remote orphans.
        if sync_remote_repo:
            all_local_files_in_repo = self._prepare_upload_folder(
                folder_path=folder_path,
                path_in_repo=path_in_repo,
                repo_type=repo_type,
                allow_patterns=None,
                ignore_patterns=None,
                warn_limits=False,
            )
            all_local_paths_in_repo = {p for p, _ in all_local_files_in_repo}
        else:
            all_local_paths_in_repo = set()

        if not sorted_files:
            raise InvalidParameter(f"No files to upload in the folder: {folder_path} !")

        logger.info("Checking %d files to upload ...", len(sorted_files))

        if self._create_repo_fn is not None:
            self._create_repo_fn(repo_id, repo_type)

        # Sort for deterministic assignment, then remove committed files before
        # planning. Otherwise a resume with four pending files in four old batch
        # ranges needlessly creates four one-file commits.
        sorted_files = sorted(sorted_files, key=lambda x: x[0])

        # The cache normally lives in the uploaded folder, but a caller that
        # stages files into a throwaway tree can keep it outside.
        folder_path_resolved = Path(folder_path).resolve()
        if use_cache:
            cache_path = Path(tracker_path).expanduser() if tracker_path else folder_path_resolved / UPLOAD_CACHE_FILE
            tracker: UploadTracker | NullTracker = UploadTracker(cache_path, repo_id=repo_id)
        else:
            tracker = NullTracker()

        pending_files: list[tuple[str, str]] = []
        skipped_count = 0
        for file_path_in_repo, file_path in sorted_files:
            try:
                st = os.stat(file_path)
                if tracker.is_committed(file_path_in_repo, st.st_mtime, st.st_size):
                    skipped_count += 1
                    continue
                tracker.begin_attempt(file_path_in_repo, st.st_mtime, st.st_size)
            except OSError as e:
                logger.warning(
                    "Cannot stat file %s, will re-upload: %s",
                    file_path_in_repo,
                    e,
                )
            pending_files.append((file_path_in_repo, file_path))
        tracker.save()

        max_operations = (
            _calculate_adaptive_batch_size(len(pending_files), UPLOAD_COMMIT_BATCH_MAX_OPERATIONS)
            if UPLOAD_ADAPTIVE_BATCHING_ENABLED
            else (UPLOAD_COMMIT_BATCH_MAX_OPERATIONS if UPLOAD_COMMIT_BATCH_MAX_OPERATIONS > 0 else len(pending_files))
        )
        batch_plan = _plan_commit_batches(
            pending_files,
            repo_type,
            max_operations=max_operations,
            max_inline_bytes=UPLOAD_COMMIT_MAX_INLINE_BYTES,
            sizes=file_sizes,
        )
        batch_tracker = BatchTracker(len(pending_files), batch_plan)
        files_to_upload = list(enumerate(pending_files))

        plan_details: list[str] = []
        offset = 0
        for count in batch_plan:
            batch_files = pending_files[offset : offset + count]
            estimated = _COMMIT_ENVELOPE_ESTIMATED_BYTES + sum(
                _estimate_commit_operation_bytes(path, file_sizes.get(local, 0), repo_type)
                for path, local in batch_files
            )
            lfs_count = sum(1 for path, local in batch_files if _is_lfs(path, file_sizes.get(local, 0), repo_type))
            plan_details.append(f"{count} ops/{estimated} bytes/{lfs_count} LFS")
            offset += count
        logger.info(
            "Commit plan: %d batch(es), %d pending, %d skipped (target %d ops, request budget %d bytes): %s",
            len(batch_plan),
            len(pending_files),
            skipped_count,
            max_operations,
            UPLOAD_COMMIT_MAX_INLINE_BYTES,
            "; ".join(plan_details[:8]) + (f"; ... {len(plan_details) - 8} more" if len(plan_details) > 8 else ""),
        )

        # Batch pre-validation for every LFS candidate.
        #
        # Without a cached hash the per-file upload path would ask the git-lfs
        # batch endpoint for its own pre-signed URL, one round trip per file.
        # Hashing up front lets all candidates be pre-signed in groups instead,
        # which is the difference between one request per file and one per group
        # once small files are routed to LFS.
        pre_validated_map: dict[str, str | None] = {}
        lfs_hash_info_map: dict[int, dict] = {}

        for file_idx, (file_path_in_repo, file_path) in files_to_upload:
            size = file_sizes.get(file_path, 0)
            if _upload_mode(file_path_in_repo, size, repo_type) != "lfs":
                continue
            try:
                st = os.stat(file_path)
            except OSError:
                continue
            cached = tracker.get_hash(file_path_in_repo, st.st_mtime, st.st_size)
            if cached is None:
                continue
            if _upload_mode(file_path_in_repo, cached["file_size"], repo_type) == "lfs":
                lfs_hash_info_map[file_idx] = cached

        uncached_lfs = [
            (file_idx, file_info)
            for file_idx, file_info in files_to_upload
            if file_idx not in lfs_hash_info_map
            and _upload_mode(file_info[0], file_sizes.get(file_info[1], 0), repo_type) == "lfs"
        ]
        if uncached_lfs:
            lfs_hash_info_map.update(self._hash_files_parallel(uncached_lfs, tracker, max_workers))

        if lfs_hash_info_map:
            objects = [{"oid": info["file_hash"], "size": info["file_size"]} for info in lfs_hash_info_map.values()]
            # Identical content shares one oid, so the request set is keyed by
            # digest rather than by file.
            unique_objects = list({obj["oid"]: obj for obj in objects}.values())
            validated = self._validate_blobs_batch(
                repo_id=repo_id,
                repo_type=repo_type,
                objects=unique_objects,
                max_workers=max_workers,
            )
            pre_validated_map = validated
            reused = sum(1 for v in validated.values() if v is None)
            unresolved = len(unique_objects) - len(validated)
            logger.info(
                "Pre-validated %d/%d distinct blob(s) for %d file(s) in %d request(s): "
                "%d already stored, %d to upload%s.",
                len(validated),
                len(unique_objects),
                len(objects),
                -(-len(unique_objects) // max(1, UPLOAD_BLOB_VALIDATION_BATCH_MAX_OBJECTS)),
                reused,
                len(validated) - reused,
                f", {unresolved} unresolved (will negotiate per file)" if unresolved else "",
            )

        # Elect one owner per distinct blob.
        #
        # `files_to_upload` is in ascending file index, and a batch owns a
        # contiguous ascending index range, so the first file holding a given oid
        # always lands in a batch no later than any of its duplicates. Batches are
        # committed in order and each waits for its own files, so by the time a
        # duplicate's batch commits, its owner has already finished -- which is
        # what lets the duplicates skip the transfer with no locking and no risk
        # of a worker pool deadlocking on itself.
        blob_owner: dict[str, int] = {}
        blob_ready: dict[str, bool] = {}
        blob_state_lock = threading.Lock()
        deduped_files = 0
        deduped_bytes = 0
        for file_idx, _file_info in files_to_upload:
            info = lfs_hash_info_map.get(file_idx)
            if info is None:
                continue
            oid = info["file_hash"]
            if pre_validated_map.get(oid, "") is None:
                # Already stored server-side; nobody needs to transfer it.
                blob_ready[oid] = True
                continue
            if oid not in blob_owner:
                blob_owner[oid] = file_idx
            else:
                deduped_files += 1
                deduped_bytes += info["file_size"]
        if deduped_files:
            logger.info(
                "Deduplicated %d file(s) sharing content with an earlier file: %d byte(s) that "
                "would otherwise be uploaded twice.",
                deduped_files,
                deduped_bytes,
            )

        if skipped_count > 0:
            logger.info("%d file(s) already committed, skipping.", skipped_count)

        logger.info(
            "Scan complete: %d total, %d committed (skip), %d to process.",
            len(sorted_files),
            skipped_count,
            len(files_to_upload),
        )

        logger.info(
            "Uploading %d file(s) in %d batch(es) (pipeline mode).",
            len(files_to_upload),
            batch_tracker.num_batches,
        )

        # Pipeline: upload workers
        def _upload_worker(file_idx: int, file_info: tuple, pre_validated: Any = None) -> None:
            path_in_repo_w, file_path_w = file_info
            owned_oid: str | None = None
            info = lfs_hash_info_map.get(file_idx)
            if info is not None and blob_owner.get(info["file_hash"]) == file_idx:
                owned_oid = info["file_hash"]
            try:
                logger.debug("Uploading: %s ...", path_in_repo_w)
                result = self._upload_single_file(
                    path_in_repo_w,
                    file_path_w,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    tracker=tracker,
                    pre_validated=pre_validated,
                    hash_info=lfs_hash_info_map.get(file_idx),
                    disable_tqdm=disable_tqdm,
                )
                logger.debug("Uploaded: %s", path_in_repo_w)
                # Publish the blob outcome before the batch is marked complete:
                # the consumer reads it as soon as the batch event fires.
                if owned_oid is not None:
                    with blob_state_lock:
                        blob_ready[owned_oid] = True
                batch_tracker.record_success(file_idx, result)
                _report_wire(result)
            except Exception as e:
                logger.error("Upload failed: %s - %s", path_in_repo_w, e)
                if owned_oid is not None:
                    with blob_state_lock:
                        blob_ready[owned_oid] = False
                batch_tracker.record_failure(file_idx, file_info, e)

        # Pipeline: consume batches in order
        commit_infos: list[dict] = []
        all_results: list[dict] = []
        retry_failed_files: list[tuple] = []
        terminal_failures: list[tuple] = []
        retry_commit_batches: list[tuple[list[dict], Exception]] = []
        num_batches = batch_tracker.num_batches
        committed_files = 0
        committed_bytes = 0
        total_bytes = sum(file_sizes.values())
        # Wire-level counters, advanced as each file's transfer finishes rather
        # than when its commit lands.
        wire_lock = threading.Lock()
        wire_state = {"bytes": 0, "files": 0, "reported_bytes": 0, "last_emit": 0.0}

        def _emit(payload: dict) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(payload)
            except Exception as cb_error:  # noqa: BLE001 - a reporter must not fail the upload
                logger.warning("Progress callback raised %s, continuing upload.", cb_error)

        def _report(
            event: str,
            batch_idx: int,
            files: int,
            num_bytes: int,
            inline_bytes: int = 0,
            error: str | None = None,
        ) -> None:
            """Emit a batch-level progress event.

            A folder upload is otherwise silent between batches, so a long run is
            indistinguishable from a hung one. Byte counts are included because a
            consumer that only learns file counts cannot compute a throughput rate
            or an ETA, which is most of what progress is for. ``inline_bytes`` is
            the part of the batch that travels inside this commit rather than
            having already gone to object storage, so a consumer can attribute
            wire traffic to the right moment without double counting.
            """
            with wire_lock:
                wire_bytes, wire_files = wire_state["bytes"], wire_state["files"]
            _emit(
                {
                    "event": event,
                    "repo_id": repo_id,
                    "batch_index": batch_idx,
                    "num_batches": num_batches,
                    "batch_files": files,
                    "batch_bytes": num_bytes,
                    "batch_inline_bytes": inline_bytes,
                    "committed_files": committed_files,
                    "committed_bytes": committed_bytes,
                    "uploaded_bytes": wire_bytes,
                    "uploaded_files": wire_files,
                    "total_files": len(sorted_files),
                    "total_bytes": total_bytes,
                    "skipped_files": skipped_count,
                    "elapsed": time.time() - start_time,
                    "error": error,
                }
            )

        def _report_wire(result: dict) -> None:
            """Account one finished file transfer, emitting at a throttled rate.

            Commits land in lumps tens of seconds apart, so a consumer fed only by
            commit events sees a rate that alternates between a spike and zero and
            cannot tell a slow batch from a hung one. Blob uploads finish
            continuously, which is the signal a rate should be built from. Only
            bytes that really went to object storage count: a deduplicated blob
            transfers nothing.
            """
            if progress_callback is None:
                return
            moved = result["file_size_on_disk"] if result.get("is_blob_uploaded") else 0
            now = time.monotonic()
            with wire_lock:
                wire_state["bytes"] += moved
                wire_state["files"] += 1
                due = now - wire_state["last_emit"] >= UPLOAD_PROGRESS_MIN_INTERVAL_SECONDS
                if not due:
                    return
                wire_state["last_emit"] = now
                delta = wire_state["bytes"] - wire_state["reported_bytes"]
                wire_state["reported_bytes"] = wire_state["bytes"]
                snapshot = (wire_state["bytes"], wire_state["files"])
            _emit(
                {
                    "event": "upload_progress",
                    "repo_id": repo_id,
                    "uploaded_bytes": snapshot[0],
                    "uploaded_bytes_delta": delta,
                    "uploaded_files": snapshot[1],
                    "committed_files": committed_files,
                    "committed_bytes": committed_bytes,
                    "total_files": len(sorted_files),
                    "total_bytes": total_bytes,
                    "skipped_files": skipped_count,
                    "elapsed": time.time() - start_time,
                    "error": None,
                }
            )

        def _flush_wire() -> None:
            """Emit whatever wire bytes the throttle has not reported yet."""
            if progress_callback is None:
                return
            with wire_lock:
                delta = wire_state["bytes"] - wire_state["reported_bytes"]
                if delta <= 0:
                    return
                wire_state["reported_bytes"] = wire_state["bytes"]
                snapshot = (wire_state["bytes"], wire_state["files"])
            _emit(
                {
                    "event": "upload_progress",
                    "repo_id": repo_id,
                    "uploaded_bytes": snapshot[0],
                    "uploaded_bytes_delta": delta,
                    "uploaded_files": snapshot[1],
                    "committed_files": committed_files,
                    "committed_bytes": committed_bytes,
                    "total_files": len(sorted_files),
                    "total_bytes": total_bytes,
                    "skipped_files": skipped_count,
                    "elapsed": time.time() - start_time,
                    "error": None,
                }
            )

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for file_idx, file_info in files_to_upload:
                    pv: str | bool | _DuplicateBlob | None = None
                    if file_idx in lfs_hash_info_map:
                        cached_hash = lfs_hash_info_map[file_idx]["file_hash"]
                        # "Answered, and the server did not ask for an upload"
                        # means the blob already exists and is reused. "Never
                        # answered" means the group failed, and the file has to
                        # negotiate its own URL -- treating that as reuse would
                        # skip the transfer and commit a pointer to a blob that
                        # was never stored.
                        if cached_hash in pre_validated_map:
                            url = pre_validated_map[cached_hash]
                            if url is None:
                                pv = True
                            elif blob_owner.get(cached_hash) == file_idx:
                                pv = url
                            else:
                                pv = DUPLICATE_BLOB
                    executor.submit(_upload_worker, file_idx, file_info, pv)

                consecutive_failures = 0
                for batch_idx in tqdm(
                    range(num_batches),
                    desc="[Committing batches]",
                    total=num_batches,
                    disable=disable_tqdm,
                ):
                    results, failures = batch_tracker.wait_for_batch(batch_idx)
                    # Every file of this batch has finished its transfer by now,
                    # so publish the bytes the throttle may still be holding
                    # before the commit event reports the batch as done.
                    _flush_wire()

                    if failures:
                        retry_failed_files.extend(failures)
                        for item, err in failures:
                            logger.error("  Failed: %s - %s", item[0], err)

                    # A file that skipped its transfer because a duplicate owned
                    # it must not be committed if that owner's upload failed:
                    # the commit would reference a blob that was never stored.
                    # Its owner is in this batch or an earlier one, both already
                    # resolved, so the outcome is known here.
                    orphaned: list[dict] = []
                    if blob_owner:
                        with blob_state_lock:
                            ready_snapshot = dict(blob_ready)
                        committable = []
                        for item_r in results:
                            oid_r = item_r["file_hash_info"]["file_hash"]
                            if (
                                item_r.get("upload_mode") == "lfs"
                                and oid_r in blob_owner
                                and not ready_snapshot.get(oid_r, False)
                            ):
                                orphaned.append(item_r)
                                continue
                            committable.append(item_r)
                        if orphaned:
                            logger.warning(
                                "Batch %d/%d: %d file(s) deferred, the upload of the content they "
                                "share failed; they will be retried on their own.",
                                batch_idx + 1,
                                num_batches,
                                len(orphaned),
                            )
                            retry_failed_files.extend(
                                (
                                    (item_r["file_path_in_repo"], item_r["file_path"]),
                                    StorageError("shared blob upload failed"),
                                )
                                for item_r in orphaned
                            )
                            results = committable

                    self._track_uploaded_batch(tracker, results)

                    operations = self._build_batch_operations(results, repo_type)
                    if not operations:
                        logger.error(
                            "Batch %d/%d: all files failed, skipping commit.",
                            batch_idx + 1,
                            num_batches,
                        )
                        _report("batch_failed", batch_idx, len(failures), 0, error="all files failed to upload")
                        continue

                    batch_commit_message = f"{commit_message} (batch {batch_idx + 1}/{num_batches})"
                    try:
                        commit_info = self._commit_with_retry(
                            repo_id=repo_id,
                            repo_type=repo_type,
                            operations=operations,
                            commit_message=batch_commit_message,
                            revision=revision,
                        )
                        commit_infos.append(commit_info)
                        all_results.extend(results)
                        logger.info(
                            "Batch %d/%d: committed %d file(s).",
                            batch_idx + 1,
                            num_batches,
                            len(results),
                        )
                        self._track_committed_batch(tracker, results)
                        consecutive_failures = 0
                        batch_bytes = sum(r["file_size_on_disk"] for r in results)
                        batch_inline_bytes = sum(
                            r["file_size_on_disk"] for r in results if r.get("upload_mode") != "lfs"
                        )
                        committed_files += len(results)
                        committed_bytes += batch_bytes
                        _report("batch_committed", batch_idx, len(results), batch_bytes, batch_inline_bytes)
                    except Exception as e:
                        logger.error(
                            "Batch %d/%d commit failed: %s",
                            batch_idx + 1,
                            num_batches,
                            e,
                        )
                        _report(
                            "batch_failed",
                            batch_idx,
                            len(results),
                            sum(r["file_size_on_disk"] for r in results),
                            error=str(e),
                        )
                        category = classify_error(e, commit_context=True)
                        if not _ErrorCategory.is_retryable(category):
                            for r in results:
                                tracker.mark_failed(
                                    r["file_path_in_repo"],
                                    r["file_mtime"],
                                    r["file_size_on_disk"],
                                    error_type="commit_" + category,
                                )
                                terminal_failures.append(((r["file_path_in_repo"], r["file_path"]), e))
                            logger.error(
                                "Batch %d/%d: terminal failure for this run (%s); %d file(s) will not be retried "
                                "automatically. Rerun upload to retry them.",
                                batch_idx + 1,
                                num_batches,
                                category,
                                len(results),
                            )
                        else:
                            retry_commit_batches.append((list(results), e))
                            logger.warning(
                                "Batch %d/%d: retained %d uploaded file(s) for commit-only recovery "
                                "(error_category=%s).",
                                batch_idx + 1,
                                num_batches,
                                len(results),
                                category,
                            )
                        consecutive_failures += 1

                        if consecutive_failures >= UPLOAD_COMMIT_MAX_CONSECUTIVE_FAILED_BATCHES:
                            raise RuntimeError(
                                f"Upload aborted: {consecutive_failures} consecutive batch commits failed. "
                                f"Last error: {e}"
                            )
        finally:
            tracker.save()

        # ReAct progressive retry fallback
        #
        # Recovery has to report progress too. It is exactly the moment an
        # operator is watching, and a run whose recovered volume never reaches
        # the metrics under-reports by however much it rescued: an 8 GiB run that
        # lost one 512-file batch to a rejected commit finished with 40000 files
        # on the Hub but 91 MB missing from done_bytes.
        def _report_recovery(results: list[dict], label: str) -> None:
            nonlocal committed_files, committed_bytes
            recovered_bytes = sum(r["file_size_on_disk"] for r in results)
            inline_bytes = sum(r["file_size_on_disk"] for r in results if r.get("upload_mode") != "lfs")
            committed_files += len(results)
            committed_bytes += recovered_bytes
            _flush_wire()
            _emit(
                {
                    "event": "recovery_committed",
                    "repo_id": repo_id,
                    "stage": label,
                    "batch_index": -1,
                    "num_batches": num_batches,
                    "batch_files": len(results),
                    "batch_bytes": recovered_bytes,
                    "batch_inline_bytes": inline_bytes,
                    "committed_files": committed_files,
                    "committed_bytes": committed_bytes,
                    "total_files": len(sorted_files),
                    "total_bytes": total_bytes,
                    "skipped_files": skipped_count,
                    "elapsed": time.time() - start_time,
                    "error": None,
                }
            )

        if retry_commit_batches:
            commit_failures, recovered_commits, recovered_results = self._retry_failed_commits(
                failed_batches=retry_commit_batches,
                tracker=tracker,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
                revision=revision,
                on_committed=_report_recovery,
            )
            terminal_failures.extend(commit_failures)
            commit_infos.extend(recovered_commits)
            all_results.extend(recovered_results)

        if retry_failed_files and UPLOAD_RECOVERY_ENABLED:
            retry_failed_files, react_commits, react_results = self._retry_failed_files_react(
                failed_files=retry_failed_files,
                tracker=tracker,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
                revision=revision,
                max_workers=max_workers,
                disable_tqdm=disable_tqdm,
                on_uploaded=_report_wire,
                on_committed=_report_recovery,
            )
            commit_infos.extend(react_commits)
            all_results.extend(react_results)
        elif retry_failed_files:
            retry_failed_files = self._retry_failed_simple(
                failed_files=retry_failed_files,
                tracker=tracker,
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=commit_message,
                commit_description=commit_description,
                revision=revision,
                commit_infos=commit_infos,
                all_results=all_results,
                disable_tqdm=disable_tqdm,
                on_uploaded=_report_wire,
                on_committed=_report_recovery,
            )

        tracker.save()
        all_failures = terminal_failures + retry_failed_files

        # Sync: delete remote orphan files only after every upload commit landed.
        deleted_count = 0
        if sync_remote_repo and not all_failures:
            prefix = path_in_repo.strip("/") if path_in_repo else ""
            orphans = self._compute_remote_orphans(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                local_paths_in_repo=all_local_paths_in_repo,
                path_in_repo_prefix=prefix,
            )
            if orphans:
                delete_ops = self._build_delete_operations(orphans)
                delete_commit_message = f"{commit_message} (sync: delete {len(orphans)} orphan file(s))"
                try:
                    delete_commit = self._commit_with_retry(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        operations=delete_ops,
                        commit_message=delete_commit_message,
                        revision=revision,
                    )
                    commit_infos.append(delete_commit)
                    deleted_count = len(orphans)
                    logger.info(
                        "Sync: deleted %d orphan file(s) from remote.",
                        deleted_count,
                    )
                except Exception as e:
                    logger.error("Sync delete commit failed: %s", e)

        # Upload report
        elapsed = time.time() - start_time
        total_files = len(sorted_files)
        failed_count = len(all_failures)
        lfs_reused_count = sum(1 for r in all_results if r.get("upload_mode") == "lfs" and r.get("is_reused"))
        lfs_uploaded_count = sum(1 for r in all_results if r.get("is_blob_uploaded"))
        normal_count = sum(1 for r in all_results if r.get("upload_mode") == "normal")
        committed_count = len(all_results)

        print("=" * 60)
        print("Upload Report")
        print("-" * 60)
        print(f"  Total files       : {total_files}")
        print(f"  Skipped (cached)  : {skipped_count}")
        print(f"  Normal committed  : {normal_count}")
        print(f"  LFS existed       : {lfs_reused_count}")
        print(f"  LFS uploaded PUT  : {lfs_uploaded_count}")
        print(f"  Failed            : {failed_count}")
        print(f"  Committed         : {committed_count}")
        print(f"  Deleted (sync)    : {deleted_count}")
        print(f"  Elapsed           : {elapsed:.1f}s")
        print("=" * 60)

        if all_failures:
            for (path_in_repo_f, _), err in all_failures:
                logger.error("  - %s: %s: %s", path_in_repo_f, type(err).__name__, err)
            succeeded = total_files - failed_count
            raise StorageError(
                f"{failed_count} file(s) failed to upload. "
                f"Please manually try again. Successfully uploaded "
                f"{succeeded} file(s) will be automatically skipped "
                f"during the retry."
            )

        if not commit_infos:
            if skipped_count == len(sorted_files):
                logger.info("All files were already committed.")
                return None
            return None

        return commit_infos[0] if len(commit_infos) == 1 else commit_infos

    # ------------------------------------------------------------------
    # Internal: remote orphan detection and deletion
    # ------------------------------------------------------------------
    def _compute_remote_orphans(
        self,
        repo_id: str,
        repo_type: str,
        revision: str,
        local_paths_in_repo: set[str],
        path_in_repo_prefix: str,
    ) -> list[str]:
        """Compute remote files that are not present locally (orphans).

        Only files under ``path_in_repo_prefix`` are considered.
        Returns a list of remote file paths to delete.
        """
        logger.info("Sync: fetching remote file list for orphan detection ...")
        raw_items = self._client.list_repo_files(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            recursive=True,
        )

        remote_paths: list[str] = []
        for item in raw_items:
            item_type = item.get("Type") or item.get("type") or "blob"
            if item_type == "tree":
                continue
            path = item.get("Path") or item.get("path") or item.get("Name") or ""
            if not path:
                continue
            remote_paths.append(path)

        # Filter by prefix scope
        if path_in_repo_prefix:
            scope_prefix = path_in_repo_prefix + "/"
            remote_paths = [p for p in remote_paths if p.startswith(scope_prefix)]

        orphans = [p for p in remote_paths if p not in local_paths_in_repo]
        logger.info(
            "Sync: %d remote file(s) in scope, %d orphan(s) detected.",
            len(remote_paths),
            len(orphans),
        )
        return orphans

    @staticmethod
    def _build_delete_operations(orphan_paths: list[str]) -> list[dict]:
        """Build delete operations for orphan remote files."""
        return [
            {
                "action": "delete",
                "path": p,
                "type": "normal",
                "size": 0,
                "sha256": "",
                "content": "",
                "encoding": "",
            }
            for p in orphan_paths
        ]

    # ------------------------------------------------------------------
    # Internal: single file upload with retry
    # ------------------------------------------------------------------
    def _upload_single_file(
        self,
        file_path_in_repo: str,
        file_path: str,
        *,
        repo_id: str,
        repo_type: str,
        tracker: UploadTracker | NullTracker | None = None,
        pre_validated: Any = None,
        hash_info: dict | None = None,
        disable_tqdm: bool = False,
    ) -> dict:
        if tracker is None:
            tracker = NullTracker()
        hash_info_d = None
        file_stat = None
        is_real_path = isinstance(file_path, (str, os.PathLike))

        if hash_info is not None:
            # Already computed during batch pre-validation; re-reading the file
            # to hash it again would double the disk cost of every LFS file.
            hash_info_d = dict(hash_info)
            hash_info_d["file_path_or_obj"] = file_path

        if hash_info_d is None and is_real_path:
            try:
                file_stat = os.stat(file_path)
                cached = tracker.get_hash(file_path_in_repo, file_stat.st_mtime, file_stat.st_size)
                if cached is not None:
                    hash_info_d = cached
                    hash_info_d["file_path_or_obj"] = file_path
            except OSError:
                file_stat = None

        if hash_info_d is None:
            hash_info_d = _compute_file_hash(file_path_or_obj=file_path)
            if is_real_path:
                try:
                    if file_stat is None:
                        file_stat = os.stat(file_path)
                    tracker.put_hash(
                        file_path_in_repo,
                        file_stat.st_mtime,
                        file_stat.st_size,
                        hash_info_d,
                    )
                except OSError:
                    pass

        if file_stat is None and is_real_path:
            try:
                file_stat = os.stat(file_path)
            except OSError:
                pass

        file_size: int = hash_info_d["file_size"]
        file_hash: str = hash_info_d["file_hash"]

        upload_mode = _upload_mode(file_path_in_repo, file_size, repo_type)

        if upload_mode == "lfs":
            # Retry loop for transient blob upload failures
            last_error = None
            for attempt in range(UPLOAD_BLOB_MAX_ATTEMPTS):
                try:
                    if isinstance(file_path, (str, os.PathLike)):
                        current_size = os.path.getsize(str(file_path))
                        if current_size != file_size:
                            raise InvalidParameter(
                                f"File size changed since hash computation: "
                                f"was {file_size}, now {current_size}. "
                                f"File may have been modified: {file_path_in_repo}"
                            )
                    upload_res = self._upload_blob(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        sha256=file_hash,
                        size=file_size,
                        data=file_path,
                        disable_tqdm=(disable_tqdm or file_size <= UPLOAD_BLOB_PROGRESS_THRESHOLD_BYTES),
                        tqdm_desc=f"[Uploading {file_path_in_repo}]",
                        pre_validated=pre_validated,
                    )
                    break
                except (HubError, ConnectionError, TimeoutError) as e:
                    if isinstance(e, HubError) and not e.retryable:
                        raise
                    last_error = e
                    if attempt < UPLOAD_BLOB_MAX_ATTEMPTS - 1:
                        wait = min(
                            UPLOAD_BLOB_RETRY_BACKOFF_BASE_SECONDS**attempt,
                            UPLOAD_BLOB_RETRY_MAX_DELAY_SECONDS,
                        )
                        logger.warning(
                            "Blob upload attempt %d/%d failed for %s: %s, retrying in %ds ...",
                            attempt + 1,
                            UPLOAD_BLOB_MAX_ATTEMPTS,
                            file_path_in_repo,
                            e,
                            wait,
                        )
                        time.sleep(wait)
            else:
                raise StorageError(
                    "Blob upload failed after "
                    f"{UPLOAD_BLOB_MAX_ATTEMPTS} attempts for "
                    f"{file_path_in_repo}: {last_error}"
                ) from last_error
        else:
            if isinstance(file_path, (str, os.PathLike)):
                current_size = os.path.getsize(str(file_path))
                if current_size != file_size:
                    raise InvalidParameter(
                        f"File size changed since hash computation: "
                        f"was {file_size}, now {current_size}. "
                        f"File may have been modified: {file_path_in_repo}"
                    )
            upload_res = {
                "url": None,
                "is_uploaded": True,
                "is_reused": False,
                "is_blob_uploaded": False,
            }

        return {
            "file_path_in_repo": file_path_in_repo,
            "file_path": file_path,
            "file_mtime": file_stat.st_mtime if file_stat else 0,
            "file_size_on_disk": (file_stat.st_size if file_stat else hash_info_d.get("file_size", 0)),
            "is_uploaded": upload_res["is_uploaded"],
            "is_reused": upload_res.get("is_reused", False),
            "is_blob_uploaded": upload_res.get("is_blob_uploaded", False),
            "upload_mode": upload_mode,
            "file_hash_info": hash_info_d,
        }

    # ------------------------------------------------------------------
    # Internal: blob upload
    # ------------------------------------------------------------------
    def _upload_blob(
        self,
        *,
        repo_id: str,
        repo_type: str,
        sha256: str,
        size: int,
        data: PathOrFileObj,
        disable_tqdm: bool = False,
        tqdm_desc: str = "[Uploading]",
        buffer_size_mb: int = 16,
        pre_validated: Any = None,
    ) -> dict:
        res_d: dict = {
            "url": None,
            "is_uploaded": False,
            "is_reused": False,
            "is_blob_uploaded": False,
        }

        if pre_validated is True:
            logger.info("Blob %s already exists globally, reuse.", sha256[:8])
            res_d["is_uploaded"] = True
            res_d["is_reused"] = True
            return res_d

        if pre_validated is DUPLICATE_BLOB:
            # An earlier file in this run owns the transfer for this content.
            # Batch ordering guarantees it has finished before any commit that
            # references this file, so nothing has to be waited on here.
            logger.debug("Blob %s is uploaded by an earlier duplicate, skipping transfer.", sha256[:8])
            res_d["is_uploaded"] = True
            res_d["is_reused"] = True
            return res_d

        if isinstance(pre_validated, str):
            upload_url: str = pre_validated
        else:
            validated = self._client.validate_blobs(
                repo_id=repo_id,
                repo_type=repo_type,
                objects=[{"oid": sha256, "size": size}],
            )
            maybe_url = validated.get(sha256)
            if maybe_url is None:
                logger.info("Blob %s already exists globally, reuse.", sha256[:8])
                res_d["is_uploaded"] = True
                res_d["is_reused"] = True
                return res_d
            upload_url = maybe_url

        chunk_size = buffer_size_mb * 1024 * 1024

        with tqdm(
            total=size,
            unit="B",
            unit_scale=True,
            desc=tqdm_desc,
            disable=disable_tqdm,
        ) as pbar:
            if isinstance(data, (str, Path)):
                with open(data, "rb") as f:
                    stream = _CountedReadStream(f, size, pbar, chunk_size)
                    self._client.upload_blob(upload_url=upload_url, data=stream, size=size)
                stream.verify_complete()
            elif isinstance(data, bytes):
                stream = _CountedReadStream(io.BytesIO(data), size, pbar, chunk_size)
                self._client.upload_blob(upload_url=upload_url, data=stream, size=size)
                stream.verify_complete()
            else:
                stream = _CountedReadStream(data, size, pbar, chunk_size)
                self._client.upload_blob(upload_url=upload_url, data=stream, size=size)
                stream.verify_complete()

        res_d["url"] = upload_url
        res_d["is_uploaded"] = True
        res_d["is_blob_uploaded"] = True
        return res_d

    # ------------------------------------------------------------------
    # Internal: batch blob validation
    # ------------------------------------------------------------------
    def _hash_files_parallel(
        self,
        files: list[tuple[int, tuple[str, str]]],
        tracker: UploadTracker | NullTracker,
        max_workers: int,
    ) -> dict[int, dict]:
        """Hash *files* concurrently and record the results in *tracker*.

        Hashing ahead of the upload pipeline is what makes group pre-signing
        possible: the git-lfs batch endpoint is keyed by ``sha256``, so without
        the digests up front each file has to negotiate its own upload URL.
        """
        hashed: dict[int, dict] = {}

        def _hash_one(entry: tuple[int, tuple[str, str]]) -> tuple[int, dict] | None:
            file_idx, (path_in_repo, file_path) = entry
            try:
                st = os.stat(file_path)
                info = _compute_file_hash(file_path_or_obj=file_path)
            except OSError as error:
                # Leave it to the upload worker, which reports per-file failures.
                logger.debug("Cannot pre-hash %s: %s", path_in_repo, error)
                return None
            tracker.put_hash(path_in_repo, st.st_mtime, st.st_size, info)
            return file_idx, info

        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            for outcome in executor.map(_hash_one, files):
                if outcome is not None:
                    hashed[outcome[0]] = outcome[1]

        logger.info("Hashed %d LFS candidate(s) for batch pre-validation.", len(hashed))
        return hashed

    def _validate_blobs_batch(
        self,
        repo_id: str,
        repo_type: str,
        objects: list[dict],
        max_workers: int = 1,
    ) -> dict[str, str | None]:
        """Pre-sign every object, in parallel groups, tolerating group failures.

        The groups are independent requests, so running them serially made the
        pre-sign phase scale linearly with the file count -- measurably so: 150
        groups took 25s of pure round-trip latency before a single byte moved.

        The batch endpoint answers only about objects that *need* uploading; an
        object it does not mention already exists server-side. That is normalised
        here into an explicit ``oid -> None`` entry, so the returned map covers
        every object of every group that answered.

        A group that fails is not fatal: its oids are simply absent from the map
        and the per-file upload path negotiates their URL itself. Keeping
        "answered: already exists" and "never answered" distinguishable is what
        makes that safe -- conflating them would skip the transfer and commit a
        pointer to a blob that was never stored.
        """
        batch_size = max(1, UPLOAD_BLOB_VALIDATION_BATCH_MAX_OBJECTS)
        chunks = [objects[i : i + batch_size] for i in range(0, len(objects), batch_size)]
        if not chunks:
            return {}

        def validate(chunk: list[dict]) -> dict[str, str | None]:
            try:
                validated = self._client.validate_blobs(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    objects=chunk,
                )
            except Exception as error:  # noqa: BLE001 - degrades to per-file negotiation
                logger.warning(
                    "Blob pre-validation failed for %d object(s) (%s); those files will negotiate "
                    "their upload URL individually.",
                    len(chunk),
                    error,
                )
                return {}
            answered: dict[str, str | None] = {obj["oid"]: None for obj in chunk}
            answered.update(validated)
            return answered

        result: dict[str, str | None] = {}
        if len(chunks) == 1:
            return validate(chunks[0])
        # Bounded by the HTTP pool: more in-flight requests than pooled
        # connections just trades latency for TLS handshakes.
        workers = max(1, min(max_workers, API_CONNECTION_POOL_MAXSIZE, len(chunks)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for validated in executor.map(validate, chunks):
                result.update(validated)
        return result

    # ------------------------------------------------------------------
    # Internal: commit with retry
    # ------------------------------------------------------------------
    def _commit_with_retry(
        self,
        *,
        repo_id: str,
        repo_type: str,
        operations: list[dict],
        commit_message: str,
        revision: str = "master",
        max_attempts: int = UPLOAD_COMMIT_MAX_ATTEMPTS,
    ) -> dict:
        last_error: Exception | None = None
        start_time = time.monotonic()
        throttled_wait = 0.0
        for attempt in range(max_attempts):
            self._commit_governor.acquire()
            try:
                return self._client.create_commit(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    operations=operations,
                    commit_message=commit_message,
                    revision=revision,
                )
            except HubError as e:
                if not _is_retryable_commit_error(e):
                    raise
                last_error = e
            except (ConnectionError, TimeoutError) as e:
                last_error = e
            except Exception as e:
                last_error = e

            # A throttled commit carries the wait the server wants; honoring it
            # beats guessing, and it is budgeted apart from the transient-error
            # allowance because its duration is known and can legitimately
            # exceed it.
            retry_after = getattr(last_error, "retry_after", None) if isinstance(last_error, RateLimitError) else None
            if retry_after is not None:
                wait = float(retry_after)
                if wait > UPLOAD_COMMIT_MAX_RETRY_AFTER_SECONDS:
                    logger.error(
                        "Commit throttled with Retry-After=%.0fs, above the %ds ceiling; aborting retries.",
                        wait,
                        UPLOAD_COMMIT_MAX_RETRY_AFTER_SECONDS,
                    )
                    break
                throttled_wait += wait
                logger.warning(
                    "Commit attempt %d/%d throttled, honoring Retry-After=%.0fs ...",
                    attempt + 1,
                    max_attempts,
                    wait,
                )
                time.sleep(wait)
                continue

            wait = min(2**attempt, 60)
            elapsed = time.monotonic() - start_time - throttled_wait
            if elapsed + wait > UPLOAD_COMMIT_RETRY_TOTAL_WAIT_SECONDS:
                logger.error(
                    "Commit total wait time would exceed %ds (already %.1fs elapsed), aborting retries.",
                    UPLOAD_COMMIT_RETRY_TOTAL_WAIT_SECONDS,
                    elapsed,
                )
                break
            logger.warning(
                "Commit attempt %d/%d failed: %s, retrying in %ds ...",
                attempt + 1,
                max_attempts,
                last_error,
                wait,
            )
            time.sleep(wait)

        if isinstance(last_error, HubError):
            raise last_error
        raise NetworkError(f"Commit failed after {max_attempts} attempts: {last_error}") from last_error

    # ------------------------------------------------------------------
    # Internal: build operations
    # ------------------------------------------------------------------
    def _build_operation(
        self,
        path_in_repo: str,
        path_or_fileobj: PathOrFileObj,
        hash_info: dict,
        upload_mode: str,
        is_uploaded: bool,
    ) -> dict:
        if upload_mode == "lfs":
            return {
                "action": "create",
                "path": path_in_repo,
                "type": "lfs",
                "size": hash_info["file_size"],
                "sha256": hash_info["file_hash"],
                "content": "",
                "encoding": "",
            }
        else:
            if isinstance(path_or_fileobj, bytes):
                content_bytes = path_or_fileobj
            elif isinstance(path_or_fileobj, (str, Path)):
                content_bytes = Path(path_or_fileobj).read_bytes()
            else:
                pos = path_or_fileobj.tell()
                content_bytes = path_or_fileobj.read()
                path_or_fileobj.seek(pos)
            return {
                "action": "create",
                "path": path_in_repo,
                "type": "normal",
                "size": hash_info["file_size"],
                "sha256": "",
                "content": base64.b64encode(content_bytes).decode(),
                "encoding": "base64",
            }

    def _build_batch_operations(self, results: list[dict], repo_type: str) -> list[dict]:
        operations = []
        for item_d in results:
            file_path = item_d["file_path"]
            hash_info = item_d["file_hash_info"]
            upload_mode = item_d.get("upload_mode") or _upload_mode(
                item_d["file_path_in_repo"],
                hash_info["file_size"],
                repo_type,
            )
            op = self._build_operation(
                path_in_repo=item_d["file_path_in_repo"],
                path_or_fileobj=file_path,
                hash_info=hash_info,
                upload_mode=upload_mode,
                is_uploaded=item_d["is_uploaded"],
            )
            operations.append(op)
        return operations

    # ------------------------------------------------------------------
    # Internal: tracker helpers
    # ------------------------------------------------------------------
    def _track_uploaded_batch(
        self,
        tracker: UploadTracker | NullTracker,
        results: list[dict],
    ) -> None:
        for r in results:
            tracker.mark_uploaded(r["file_path_in_repo"], r["file_mtime"], r["file_size_on_disk"])
        tracker.save()

    def _track_committed_batch(
        self,
        tracker: UploadTracker | NullTracker,
        results: list[dict],
    ) -> None:
        tracker.mark_committed_batch(
            [(r["file_path_in_repo"], r["file_mtime"], r["file_size_on_disk"]) for r in results]
        )
        tracker.save()

    def _plan_result_batches(self, results: list[dict], repo_type: str) -> list[list[dict]]:
        """Apply the normal commit planner to already-uploaded results."""
        if not results:
            return []
        files = [(r["file_path_in_repo"], r["file_path"]) for r in results]
        sizes = {r["file_path"]: r["file_size_on_disk"] for r in results}
        max_operations = _calculate_adaptive_batch_size(len(files), UPLOAD_COMMIT_BATCH_MAX_OPERATIONS)
        counts = _plan_commit_batches(
            files,
            repo_type,
            max_operations=max_operations,
            max_inline_bytes=UPLOAD_COMMIT_MAX_INLINE_BYTES,
            sizes=sizes,
        )
        batches: list[list[dict]] = []
        offset = 0
        for count in counts:
            batches.append(results[offset : offset + count])
            offset += count
        return batches

    @staticmethod
    def _should_split_commit_failure(error: Exception) -> bool:
        body = getattr(error, "response_body", None)
        raw_code = body.get("Code") if isinstance(body, dict) else None
        if raw_code is None and isinstance(body, dict):
            raw_code = body.get("code")
        try:
            policy_code = raw_code is not None and int(raw_code) in _RETRYABLE_COMMIT_BUSINESS_CODES
        except (TypeError, ValueError):
            policy_code = False
        message = str(getattr(error, "message", None) or error).lower()
        return "repository policy" in message or (policy_code and getattr(error, "status_code", None) == 400)

    def _retry_failed_commits(
        self,
        *,
        failed_batches: list[tuple[list[dict], Exception]],
        tracker: UploadTracker | NullTracker,
        repo_id: str,
        repo_type: str,
        commit_message: str,
        revision: str,
        on_committed: Any = None,
    ) -> tuple[list[tuple], list[dict], list[dict]]:
        """Retry commits without re-uploading blobs, splitting policy rejects."""
        failures: list[tuple] = []
        commit_infos: list[dict] = []
        committed_results: list[dict] = []

        def commit_batch(
            batch: list[dict],
            label: str,
            prior_error: Exception,
            *,
            split_before_attempt: bool = False,
        ) -> None:
            if split_before_attempt and self._should_split_commit_failure(prior_error) and len(batch) > 1:
                midpoint = (len(batch) + 1) // 2
                logger.warning(
                    "Commit-only recovery is splitting a policy-rejected batch of %d into %d and %d files.",
                    len(batch),
                    midpoint,
                    len(batch) - midpoint,
                )
                commit_batch(batch[:midpoint], label + "a", prior_error)
                commit_batch(batch[midpoint:], label + "b", prior_error)
                return

            try:
                commit_info = self._commit_with_retry(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    operations=self._build_batch_operations(batch, repo_type),
                    commit_message=f"{commit_message} (commit recovery {label})",
                    revision=revision,
                )
            except Exception as error:
                if self._should_split_commit_failure(error) and len(batch) > 1:
                    midpoint = (len(batch) + 1) // 2
                    commit_batch(batch[:midpoint], label + "a", error)
                    commit_batch(batch[midpoint:], label + "b", error)
                    return
                category = classify_error(error, commit_context=True)
                for result in batch:
                    tracker.mark_failed(
                        result["file_path_in_repo"],
                        result["file_mtime"],
                        result["file_size_on_disk"],
                        error_type="commit_" + category,
                    )
                    failures.append(((result["file_path_in_repo"], result["file_path"]), error))
                logger.error("Commit-only recovery %s failed for %d file(s): %s", label, len(batch), error)
                return

            commit_infos.append(commit_info)
            committed_results.extend(batch)
            self._track_committed_batch(tracker, batch)
            if on_committed is not None:
                on_committed(batch, f"commit recovery {label}")
            logger.info("Commit-only recovery %s committed %d file(s).", label, len(batch))

        for batch_index, (results, error) in enumerate(failed_batches, start=1):
            for part_index, batch in enumerate(self._plan_result_batches(results, repo_type), start=1):
                commit_batch(
                    batch,
                    f"{batch_index}.{part_index}",
                    error,
                    split_before_attempt=True,
                )
        return failures, commit_infos, committed_results

    # ------------------------------------------------------------------
    # Internal: file collection
    # ------------------------------------------------------------------
    def _prepare_upload_folder(
        self,
        folder_path: str | Path,
        path_in_repo: str,
        repo_type: str = "model",
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        sizes_out: dict[str, int] | None = None,
        warn_limits: bool = True,
    ) -> list[tuple[str, str]]:
        folder = Path(folder_path).expanduser().resolve()
        if not folder.is_dir():
            raise InvalidParameter(f"Provided path: '{folder}' is not a directory")

        all_files = sorted(path for path in folder.glob("**/*") if path.is_file())
        relpath_to_abspath = {path.relative_to(folder).as_posix(): str(path) for path in all_files}
        filtered_keys = _filter_repo_objects(
            list(relpath_to_abspath.keys()),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

        # ``path_in_repo`` is a destination prefix, and "." / "./" / "" / "/"
        # all mean the repository root.
        norm_prefix = _normalize_path_in_repo(path_in_repo)
        prefix = f"{norm_prefix}/" if norm_prefix else ""
        prepared = [(prefix + relpath, relpath_to_abspath[relpath]) for relpath in filtered_keys]

        selected_sizes: dict[str, int] = {}
        for _path_in_repo, file_path in prepared:
            selected_sizes[file_path] = Path(file_path).stat().st_size
        if sizes_out is not None:
            sizes_out.update(selected_sizes)
        if warn_limits:
            _warn_advisory_upload_limits(prepared, selected_sizes, repo_type)

        logger.info("Prepared %d files for upload.", len(prepared))
        return prepared

    # ------------------------------------------------------------------
    # Internal: ReAct progressive retry
    # ------------------------------------------------------------------
    def _retry_failed_files_react(
        self,
        failed_files: list[tuple],
        tracker: UploadTracker | NullTracker,
        repo_id: str,
        repo_type: str,
        commit_message: str,
        revision: str,
        max_workers: int,
        disable_tqdm: bool = False,
        on_uploaded: Any = None,
        on_committed: Any = None,
    ) -> tuple[list[tuple], list[dict], list[dict]]:
        commit_infos: list[dict] = []
        all_successes: list[dict] = []
        retry_counts: dict[str, int] = {}
        permanent_failures: list[tuple] = []
        retryable = list(failed_files)

        remaining = []
        for item_err in retryable:
            (path_in_repo_r, file_path_r), err = item_err
            category = classify_error(err)
            if _ErrorCategory.is_retryable(category):
                remaining.append(item_err)
            else:
                permanent_failures.append(item_err)
                try:
                    st = os.stat(file_path_r) if isinstance(file_path_r, (str, os.PathLike)) else None
                except OSError:
                    st = None
                tracker.mark_failed(
                    path_in_repo_r,
                    st.st_mtime if st else 0,
                    st.st_size if st else 0,
                    error_type=category,
                )
                logger.error(
                    "[ReAct] Permanent failure: %s (%s: %s)",
                    path_in_repo_r,
                    category,
                    err,
                )
        retryable = remaining

        round_configs: list[dict[str, Any]] = [
            {
                "name": "Round 1 (parallel)",
                "parallel": True,
                "workers": max(1, max_workers // 2),
                "delay": 0,
            },
            {
                "name": "Round 2 (serial+backoff)",
                "parallel": False,
                "workers": 1,
                "delay": UPLOAD_RECOVERY_SERIAL_BACKOFF_BASE_SECONDS,
            },
            {
                "name": "Round 3 (single-file)",
                "parallel": False,
                "workers": 1,
                "delay": UPLOAD_RECOVERY_SINGLE_FILE_DELAY_SECONDS,
            },
        ]

        for round_idx, cfg in enumerate(round_configs):
            if not retryable:
                break

            round_name = cfg["name"]
            logger.info(
                "[ReAct] %s: retrying %d file(s) ...",
                round_name,
                len(retryable),
            )

            round_successes: list[dict] = []
            round_failures: list[tuple] = []

            if cfg["parallel"] and len(retryable) > 1:
                with ThreadPoolExecutor(max_workers=cfg["workers"]) as executor:
                    future_map: dict = {}
                    for (path_in_repo_r, file_path_r), _err in retryable:
                        future = executor.submit(
                            self._upload_single_file,
                            path_in_repo_r,
                            file_path_r,
                            repo_id=repo_id,
                            repo_type=repo_type,
                            tracker=tracker,
                            disable_tqdm=disable_tqdm,
                        )
                        future_map[future] = (path_in_repo_r, file_path_r)
                    for future in as_completed(future_map):
                        path_in_repo_r, file_path_r = future_map[future]
                        try:
                            result = future.result()
                            round_successes.append(result)
                            if on_uploaded is not None:
                                on_uploaded(result)
                        except Exception as e:
                            round_failures.append(((path_in_repo_r, file_path_r), e))
            else:
                for i, ((path_in_repo_r, file_path_r), _err) in enumerate(retryable):
                    if cfg["delay"] > 0 and i > 0:
                        delay = (
                            cfg["delay"] * (2 ** min(i, UPLOAD_RECOVERY_BACKOFF_MAX_EXPONENT))
                            if round_idx == 1
                            else cfg["delay"]
                        )
                        delay = min(delay, UPLOAD_RECOVERY_MAX_DELAY_SECONDS)
                        logger.info(
                            "[ReAct] Waiting %ds before retrying %s ...",
                            delay,
                            path_in_repo_r,
                        )
                        time.sleep(delay)
                    try:
                        result = self._upload_single_file(
                            path_in_repo_r,
                            file_path_r,
                            repo_id=repo_id,
                            repo_type=repo_type,
                            tracker=tracker,
                            disable_tqdm=disable_tqdm,
                        )
                        round_successes.append(result)
                        if on_uploaded is not None:
                            on_uploaded(result)
                    except Exception as e:
                        logger.error(
                            "[ReAct] %s: failed %s - %s",
                            round_name,
                            path_in_repo_r,
                            e,
                        )
                        round_failures.append(((path_in_repo_r, file_path_r), e))

            self._track_uploaded_batch(tracker, round_successes)
            for batch in self._plan_result_batches(round_successes, repo_type):
                operations = self._build_batch_operations(batch, repo_type)
                if not operations:
                    continue
                try:
                    commit_info = self._commit_with_retry(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        operations=operations,
                        commit_message=f"{commit_message} ({round_name})",
                        revision=revision,
                    )
                    commit_infos.append(commit_info)
                    all_successes.extend(batch)
                    self._track_committed_batch(tracker, batch)
                    if on_committed is not None:
                        on_committed(batch, round_name)
                    logger.info(
                        "[ReAct] %s: committed %d file(s).",
                        round_name,
                        len(batch),
                    )
                except Exception as e:
                    logger.error("[ReAct] %s commit failed: %s", round_name, e)
                    commit_failures, recovered_commits, recovered_results = self._retry_failed_commits(
                        failed_batches=[(batch, e)],
                        tracker=tracker,
                        repo_id=repo_id,
                        repo_type=repo_type,
                        commit_message=commit_message,
                        revision=revision,
                        on_committed=on_committed,
                    )
                    permanent_failures.extend(commit_failures)
                    commit_infos.extend(recovered_commits)
                    all_successes.extend(recovered_results)

            new_retryable = []
            for item_err in round_failures:
                (path_in_repo_r, file_path_r), err = item_err
                retry_counts[path_in_repo_r] = retry_counts.get(path_in_repo_r, 0) + 1
                if retry_counts[path_in_repo_r] >= 3:
                    permanent_failures.append(item_err)
                    try:
                        st = os.stat(file_path_r) if isinstance(file_path_r, (str, os.PathLike)) else None
                    except OSError:
                        st = None
                    tracker.mark_failed(
                        path_in_repo_r,
                        st.st_mtime if st else 0,
                        st.st_size if st else 0,
                        error_type="max_retries_exceeded",
                    )
                    logger.error("[ReAct] Max retries exceeded for %s", path_in_repo_r)
                    continue
                category = classify_error(err)
                if _ErrorCategory.is_retryable(category):
                    new_retryable.append(item_err)
                else:
                    permanent_failures.append(item_err)
                    try:
                        st = os.stat(file_path_r) if isinstance(file_path_r, (str, os.PathLike)) else None
                    except OSError:
                        st = None
                    tracker.mark_failed(
                        path_in_repo_r,
                        st.st_mtime if st else 0,
                        st.st_size if st else 0,
                        error_type=category,
                    )
                    logger.error(
                        "[ReAct] Permanent failure: %s (%s)",
                        path_in_repo_r,
                        category,
                    )

            progress = len(retryable) - len(new_retryable)
            if progress > 0:
                logger.info(
                    "[ReAct] %s: made progress — %d file(s) resolved, %d remaining.",
                    round_name,
                    progress,
                    len(new_retryable),
                )
            elif new_retryable:
                logger.warning(
                    "[ReAct] %s: no progress, escalating to next round.",
                    round_name,
                )

            retryable = new_retryable

        all_failures = permanent_failures + retryable
        if retryable:
            logger.error(
                "[ReAct] %d file(s) still failing after all retry rounds.",
                len(retryable),
            )

        return all_failures, commit_infos, all_successes

    # ------------------------------------------------------------------
    # Internal: simple retry (when ReAct is disabled)
    # ------------------------------------------------------------------
    def _retry_failed_simple(
        self,
        failed_files: list[tuple],
        tracker: UploadTracker | NullTracker,
        repo_id: str,
        repo_type: str,
        commit_message: str,
        commit_description: str | None,
        revision: str,
        commit_infos: list[dict],
        all_results: list[dict],
        disable_tqdm: bool = False,
        on_uploaded: Any = None,
        on_committed: Any = None,
    ) -> list[tuple]:
        total_failed_files = list(failed_files)
        terminal_failures: list[tuple] = []
        for retry_round in range(UPLOAD_FAILED_FILE_MAX_RETRY_ROUNDS):
            if not total_failed_files:
                break
            logger.info(
                "Retry round %d/%d: re-uploading %d failed file(s) ...",
                retry_round + 1,
                UPLOAD_FAILED_FILE_MAX_RETRY_ROUNDS,
                len(total_failed_files),
            )
            retry_failures: list[tuple] = []
            retry_successes: list[dict] = []
            for (path_in_repo_r, file_path_r), _err in total_failed_files:
                try:
                    result = self._upload_single_file(
                        path_in_repo_r,
                        file_path_r,
                        repo_id=repo_id,
                        repo_type=repo_type,
                        tracker=tracker,
                        disable_tqdm=disable_tqdm,
                    )
                    retry_successes.append(result)
                    if on_uploaded is not None:
                        on_uploaded(result)
                except Exception as e:
                    logger.error("  Retry failed: %s - %s", path_in_repo_r, e)
                    retry_failures.append(((path_in_repo_r, file_path_r), e))
            if retry_successes:
                self._track_uploaded_batch(tracker, retry_successes)
                for batch in self._plan_result_batches(retry_successes, repo_type):
                    operations = self._build_batch_operations(batch, repo_type)
                    if not operations:
                        continue
                    try:
                        commit_info = self._commit_with_retry(
                            repo_id=repo_id,
                            repo_type=repo_type,
                            operations=operations,
                            commit_message=(f"{commit_message} (retry round {retry_round + 1})"),
                            revision=revision,
                        )
                        commit_infos.append(commit_info)
                        all_results.extend(batch)
                        self._track_committed_batch(tracker, batch)
                        if on_committed is not None:
                            on_committed(batch, f"retry round {retry_round + 1}")
                        logger.info(
                            "  Retry round %d: committed %d file(s).",
                            retry_round + 1,
                            len(batch),
                        )
                    except Exception as e:
                        logger.error(
                            "  Retry round %d commit failed: %s",
                            retry_round + 1,
                            e,
                        )
                        commit_failures, recovered_commits, recovered_results = self._retry_failed_commits(
                            failed_batches=[(batch, e)],
                            tracker=tracker,
                            repo_id=repo_id,
                            repo_type=repo_type,
                            commit_message=commit_message,
                            revision=revision,
                            on_committed=on_committed,
                        )
                        terminal_failures.extend(commit_failures)
                        commit_infos.extend(recovered_commits)
                        all_results.extend(recovered_results)
            total_failed_files = retry_failures
        return terminal_failures + total_failed_files
