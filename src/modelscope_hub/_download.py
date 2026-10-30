"""Internal file download implementation.

Supports single-file and whole-repo (snapshot) downloads with:
- HTTP Range-based resume
- SHA256 integrity verification
- tqdm progress display
- Parallel downloads via ThreadPoolExecutor
- Parallel range download (split large files into parts)
- Per-file download retry with backoff
- File lock for multiprocess safety
- User-agent / snapshot headers
- Intra-cloud acceleration
- Local-files-only (offline) mode
- Custom progress callbacks
- Local snapshot cache directory management
"""

from __future__ import annotations

import contextlib
import copy
import errno
import fnmatch
import hashlib
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import requests
from tqdm.auto import tqdm
from urllib3.exceptions import MaxRetryError
from urllib3.util.retry import Retry

from ._download_control import (
    _DownloadCancelledError,
    _DownloadControl,
    _DownloadDeadlineError,
    _network_context,
    _ReadHTTPAdapter,
    _RecoveryBudget,
)
from .constants import (
    DOWNLOAD_CHUNK_SIZE,
    DOWNLOAD_FILE_LOCK_TIMEOUT_SECONDS,
    DOWNLOAD_LOW_SPEED_WINDOW_SECONDS,
    DOWNLOAD_MIN_BYTES_PER_SECOND,
    DOWNLOAD_PARALLEL_THRESHOLD,
    DOWNLOAD_PARALLELS,
    DOWNLOAD_PART_SIZE,
    DOWNLOAD_RETRY_MAX_DELAY_SECONDS,
    DOWNLOAD_RETRY_TIMES,
    DOWNLOAD_TIMEOUT,
    DOWNLOAD_TOTAL_TIMEOUT,
    ENV_FILE_LOCK,
    ENV_INTER_CLOUD_REGIONS,
    ENV_INTRA_CLOUD_ACCELERATION,
    ENV_INTRA_CLOUD_REGION,
    REPO_TREE_LIST_TIMEOUT_SECONDS,
)
from .errors import (
    APIError,
    AuthenticationError,
    CacheNotFound,
    FileDownloadError,
    FileIntegrityError,
    FileSizeMismatchError,
    HubError,
    NetworkError,
    NotExistError,
    PermissionDeniedError,
    RequestTimeoutError,
)
from .utils.file_utils import compute_hash, ensure_dir
from .utils.logger import get_logger
from .utils.patterns import extract_common_prefix

if TYPE_CHECKING:
    from ._legacy_api import LegacyClient
    from .config import HubConfig
    from .types import DownloadReport, DownloadRetryPolicy

logger = get_logger("download")

DOWNLOAD_HASH_RETRY_TIMES = 3
# Attempts for a file whose delivered size contradicts the repository listing.
# The server answered consistently, so one more try over the default route is
# enough to rule out a stale replica or CDN node before reporting the file.
DOWNLOAD_SIZE_RETRY_TIMES = 2
# Seconds before the first integrity retry, doubled for each further one, so a
# transient CDN or replica inconsistency has time to settle.
DOWNLOAD_INTEGRITY_RETRY_BACKOFF = 1.0

# Response headers identifying the request and the node that served it. They
# are quoted in integrity errors so a broken object can be traced server-side.
_DIAGNOSTIC_HEADERS = ("x-oss-request-id", "x-request-id", "eagleid", "x-cache", "x-linked-etag")
_CONTENT_RANGE_RE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Progress callback system
# ---------------------------------------------------------------------------
class ProgressCallback:
    """Base class for download progress callbacks.

    Subclass and override :meth:`update` / :meth:`end` to track download
    progress. Instances are created per-file by
    ``callback_cls(filename, file_size)``.
    """

    def __init__(self, filename: str, file_size: int):
        self.filename = filename
        self.file_size = file_size

    def update(self, size: int) -> None:
        pass

    def end(self) -> None:
        pass


class TqdmCallback(ProgressCallback):
    """Progress callback backed by tqdm."""

    def __init__(self, filename: str, file_size: int):
        super().__init__(filename, file_size)
        self.progress = tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=file_size if file_size > 0 else 1,
            initial=0,
            desc=f"Downloading [{self.filename}]",
            leave=True,
        )

    def update(self, size: int) -> None:
        self.progress.update(size)

    def end(self) -> None:
        self.progress.close()


# ---------------------------------------------------------------------------
# File lock
# ---------------------------------------------------------------------------
_STALE_LOCK_SECONDS = 2 * 3600


@contextlib.contextmanager
def _optional_file_lock(
    lock_path: Path | None,
    *,
    enabled: bool = True,
    control: _DownloadControl | None = None,
):
    """Acquire a file lock when *enabled*, otherwise no-op."""
    if not enabled or lock_path is None:
        yield
        return

    from filelock import FileLock, SoftFileLock, Timeout

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    default_interval = 0.2 if control is not None else 60
    lock = FileLock(str(lock_path), timeout=default_interval)
    is_soft = False
    waited = 0.0
    started = time.monotonic()
    next_notice = 60.0

    while True:
        if control is not None:
            control.check()
            lock_limit = float(DOWNLOAD_FILE_LOCK_TIMEOUT_SECONDS)
            if control.recovery is not None:
                lock_limit = min(lock_limit, control.recovery.policy.idle_timeout)
            if time.monotonic() - started >= lock_limit:
                raise _DownloadDeadlineError(f"Download file lock timed out after {lock_limit}s: {lock_path}")
        try:
            lock.acquire(timeout=default_interval)
        except Timeout:
            waited += default_interval
            if is_soft and waited >= _STALE_LOCK_SECONDS:
                try:
                    age = time.time() - lock_path.stat().st_mtime
                except OSError:
                    age = 0
                if age >= _STALE_LOCK_SECONDS:
                    logger.warning(
                        "Removing possibly stale SoftFileLock (age=%.0fs): %s",
                        age,
                        lock_path,
                    )
                    try:
                        lock_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    waited = 0
                    continue
            if waited >= next_notice:
                logger.info("Still waiting to acquire lock on %s", lock_path)
                next_notice = waited + 60
        except NotImplementedError as exc:
            if "use SoftFileLock instead" in str(exc):
                logger.warning(
                    "Filesystem does not support flock, falling back to SoftFileLock for %s",
                    lock_path,
                )
                lock = SoftFileLock(str(lock_path), timeout=default_interval)
                is_soft = True
                continue
            raise
        except OSError as exc:
            if exc.errno in (errno.ESTALE, errno.ENOENT, getattr(errno, "EREMOTEIO", -1)):
                logger.warning(
                    "OSError (errno=%d) on %s, falling back to SoftFileLock.",
                    exc.errno,
                    lock_path,
                )
                lock = SoftFileLock(str(lock_path), timeout=default_interval)
                is_soft = True
                continue
            raise
        else:
            break

    try:
        yield lock
    finally:
        try:
            lock.release()
        except OSError:
            try:
                lock_path.unlink()
            except OSError:
                pass


def _file_lock_enabled() -> bool:
    """Check whether file locking is enabled via environment."""
    from .constants import _env_bool

    return _env_bool(ENV_FILE_LOCK, True)


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------
def _matches_patterns(path: str, patterns: list[str] | None) -> bool:
    """Check if path matches any of the glob patterns.

    Also accepts legacy regex-style patterns (e.g. ``.*\\.bin``) by
    converting ``.*`` to ``*`` before matching.
    """
    if not patterns:
        return False
    for pat in patterns:
        if fnmatch.fnmatch(path, pat):
            return True
        if ".*" in pat:
            glob_pat = pat.replace(".*", "*")
            if fnmatch.fnmatch(path, glob_pat):
                return True
    return False


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------
class _IncompleteBodyError(Exception):
    """The body ended before the announced length arrived; safe to resume."""


class _RangeNotHonouredError(Exception):
    """A ranged request was answered with something other than that range."""


def _listing_roots(patterns: list[str] | None) -> list[str | None]:
    """Use only literal directory prefixes; keep local glob filtering as the final selection check."""
    if not patterns or any(".*" in p or "\\" in p or p.startswith("/") for p in patterns):
        return [None]
    roots = [extract_common_prefix([pattern]) for pattern in patterns]
    if any(root is None or ".." in root.split("/") for root in roots):
        return [None]
    result: list[str | None] = []
    for root in sorted(set(roots), key=lambda value: (len(value or ""), value or "")):
        if root is not None and not any(root == parent or root.startswith(f"{parent}/") for parent in result):
            result.append(root)
    return result


def _parse_content_range(value: str | None) -> tuple[int, int, int | None] | None:
    """Parse ``Content-Range: bytes start-end/total``; *total* may be ``*``."""
    match = _CONTENT_RANGE_RE.fullmatch(value.strip()) if value else None
    if match is None:
        return None
    total = None if match.group(3) == "*" else int(match.group(3))
    start, end = int(match.group(1)), int(match.group(2))
    if end < start or (total is not None and end >= total):
        return None
    return start, end, total


def _unsatisfied_range_total(value: str | None) -> int | None:
    """Return the object size from a 416 ``Content-Range: bytes */total`` header."""
    total = (value or "").rpartition("/")[2].strip()
    return int(total) if total.isdigit() else None


def _checkable_content_length(resp: requests.Response) -> int | None:
    """Return the announced body length when it can be compared with the bytes read.

    Under a ``Content-Encoding`` the header counts encoded bytes while
    ``iter_content`` yields decoded ones, so the two cannot be compared.
    """
    raw = resp.headers.get("Content-Length")
    if raw is None or resp.headers.get("Content-Encoding", "identity").strip().lower() not in ("", "identity"):
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _describe_response(resp: requests.Response, request_id: str) -> str:
    """Summarise a download response for diagnostics without leaking the signed URL."""
    headers = resp.headers
    parts = [f"HTTP {resp.status_code}", f"Content-Length={headers.get('Content-Length', 'absent')}"]
    if headers.get("Content-Range"):
        parts.append(f"Content-Range={headers['Content-Range']}")
    parts.append(f"host={urlparse(resp.url or '').hostname or 'unknown'}")
    parts.append(f"X-Request-ID={request_id}")
    parts.extend(f"{name}={headers[name]}" for name in _DIAGNOSTIC_HEADERS if headers.get(name))
    return ", ".join(parts)


def _incomplete_path(target: Path) -> Path:
    """Return the partial-download path used while *target* is being fetched."""
    return target.with_suffix(target.suffix + ".incomplete")


def _wait_download_retry(retry: Retry, exc: Exception, path: str, control: _DownloadControl) -> Retry:
    control.check()
    if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EROFS, errno.EACCES):
        raise exc
    if isinstance(exc, HubError) and not exc.retryable:
        raise exc
    try:
        retry = retry.increment("GET", path, error=exc)
    except MaxRetryError:
        if isinstance(exc, HubError):
            raise exc
        raise NetworkError(
            f"Download of {path} failed after {DOWNLOAD_RETRY_TIMES} retries: {type(exc).__name__}"
        ) from exc
    requested = getattr(exc, "retry_after", None)
    if control.recovery is not None:
        delay = max(0, requested) if requested is not None else min(retry.get_backoff_time(), 2)
        try:
            control.recovery.retry(delay)
        except Exception as budget_error:
            raise budget_error from exc
        logger.warning("Download recovery for %s (%s), retrying in %ss", path, type(exc).__name__, delay)
        control.sleep(delay)
        return retry
    delay = min(
        max(0, requested) if requested is not None else retry.get_backoff_time(), DOWNLOAD_RETRY_MAX_DELAY_SECONDS
    )
    logger.warning("Download failed for %s (%s), retrying in %ss", path, type(exc).__name__, delay)
    control.sleep(delay)
    return retry


def _finish_download(
    tmp: Path,
    target: Path,
    expected_hash: str | None,
    control: _DownloadControl,
    actual_hash: str | None = None,
) -> None:
    if expected_hash:
        if actual_hash is None:
            digest = hashlib.sha256()
            with tmp.open("rb") as fh:
                while block := fh.read(8 * DOWNLOAD_CHUNK_SIZE):
                    control.check()
                    digest.update(block)
            actual_hash = digest.hexdigest()
        if actual_hash != expected_hash:
            raise FileIntegrityError(
                f"Hash mismatch for {target.name}: expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
            )
    control.check()
    tmp.replace(target)


# ---------------------------------------------------------------------------
# Parallel range download
# ---------------------------------------------------------------------------
def _download_part_with_retry(params: tuple) -> None:
    """Download one byte range into its part file, resuming after transport errors.

    Every response must be exactly the requested range: a server that ignores
    ``Range`` raises :class:`_RangeNotHonouredError` so the caller can fall back to a
    single stream instead of splicing whole-file bodies into the parts.
    """
    (
        file_path,
        progress_callbacks,
        start,
        end,
        url,
        file_name,
        headers,
        cookies,
        file_size,
        request,
        control,
        part_stop,
    ) = params
    expected = end - start + 1
    get_headers = {} if headers is None else copy.deepcopy(headers)
    get_headers["X-Request-ID"] = uuid.uuid4().hex
    retry = Retry(
        total=DOWNLOAD_RETRY_TIMES,
        backoff_factor=1,
        allowed_methods=["GET"],
    )
    part_file_name = f"{file_path}_{start}_{end}"
    reported = 0
    while True:
        try:
            control.check()
            if part_stop.is_set():
                raise _DownloadCancelledError("Range part group cancelled")
            partial_length = os.path.getsize(part_file_name) if os.path.exists(part_file_name) else 0
            if partial_length > expected:
                # Left over from a different split of the file; it cannot be resumed.
                os.remove(part_file_name)
                partial_length = 0
            if partial_length > reported:
                for cb in progress_callbacks:
                    cb.update(partial_length - reported)
                reported = partial_length
            download_start = start + partial_length
            if download_start > end:
                break
            get_headers["Range"] = f"bytes={download_start}-{end}"
            try:
                r = request(get_headers)
            except APIError as exc:
                if exc.status_code == 416:
                    response = getattr(exc, "response", None)
                    total = (
                        _unsatisfied_range_total(response.headers.get("Content-Range"))
                        if response is not None
                        else None
                    )
                    raise FileSizeMismatchError(
                        f"Server reports {total if total is not None else 'fewer'} bytes for {file_name} "
                        f"but the repository lists {file_size} bytes (HTTP 416)"
                    ) from exc
                raise
            with control.response(r):
                if part_stop.is_set():
                    raise _DownloadCancelledError("Range part group cancelled")
                content_range = _parse_content_range(r.headers.get("Content-Range"))
                if content_range is not None and content_range[2] is not None and content_range[2] != file_size:
                    raise FileSizeMismatchError(
                        f"Server reports {content_range[2]} bytes for {file_name} but the repository lists "
                        f"{file_size} bytes ({_describe_response(r, get_headers['X-Request-ID'])})"
                    )
                if content_range is None or content_range[0] != download_start or content_range[1] != end:
                    raise _RangeNotHonouredError(
                        f"asked for bytes={download_start}-{end}, got "
                        f"{_describe_response(r, get_headers['X-Request-ID'])}"
                    )
                with open(part_file_name, "ab") as f:
                    for chunk in control.chunks(r, DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            if f.tell() + len(chunk) > expected:
                                raise FileSizeMismatchError(f"Server exceeded range {start}-{end} for {file_name}")
                            f.write(chunk)
                            reported += len(chunk)
                            for cb in progress_callbacks:
                                cb.update(len(chunk))
            written = os.path.getsize(part_file_name)
            if written != expected:
                if written > expected:
                    os.remove(part_file_name)
                raise _IncompleteBodyError(f"part {start}-{end} holds {written} of {expected} bytes")
            break
        except (
            _RangeNotHonouredError,
            FileIntegrityError,
            _DownloadCancelledError,
            _DownloadDeadlineError,
            NotExistError,
            PermissionDeniedError,
            AuthenticationError,
        ):
            raise
        except Exception as exc:
            retry = _wait_download_retry(retry, exc, f"{file_name}:{start}-{end}", control)


def _parallel_download(
    url: str,
    target: Path,
    file_size: int,
    headers: dict[str, str] | None,
    cookies: dict | None,
    progress_callbacks: list[ProgressCallback] | None = None,
    *,
    request: Callable[[dict[str, str]], requests.Response],
    control: _DownloadControl,
    expected_sha256: str | None = None,
) -> str:
    """Split a large file into parts and download in parallel.

    Returns the SHA256 hex digest of the merged file. When the server ignores
    ranged requests, or serves an object of a different size, the parts are
    removed and the error is re-raised for the caller to handle.
    """
    callbacks = progress_callbacks or []
    part_size = DOWNLOAD_PART_SIZE
    part_stop = threading.Event()
    tasks: list[tuple] = []
    file_path = str(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    num_full_parts = file_size // part_size
    for idx in range(num_full_parts):
        start = idx * part_size
        end = (idx + 1) * part_size - 1
        tasks.append(
            (
                file_path,
                callbacks,
                start,
                end,
                url,
                target.name,
                headers,
                cookies,
                file_size,
                request,
                control,
                part_stop,
            )
        )
    remainder_start = num_full_parts * part_size
    if remainder_start < file_size:
        tasks.append(
            (
                file_path,
                callbacks,
                remainder_start,
                file_size - 1,
                url,
                target.name,
                headers,
                cookies,
                file_size,
                request,
                control,
                part_stop,
            )
        )
    part_names = [f"{file_path}_{task[2]}_{task[3]}" for task in tasks]

    parallels = min(DOWNLOAD_PARALLELS, 16)
    executor = ThreadPoolExecutor(max_workers=parallels, thread_name_prefix="download")
    futures = [executor.submit(_download_part_with_retry, task) for task in tasks]
    try:
        for future in as_completed(futures):
            future.result()
    except BaseException as exc:
        part_stop.set()
        control.abort_responses()
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        if isinstance(exc, (_RangeNotHonouredError, FileIntegrityError)):
            # Parts of a whole-file body or of the wrong object cannot be resumed.
            for name in part_names:
                with contextlib.suppress(OSError):
                    os.remove(name)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        for cb in callbacks:
            cb.end()

    hash_sha256 = hashlib.sha256()
    merged = 0
    tmp_target = target.with_suffix(target.suffix + ".parallel_tmp")
    with open(tmp_target, "wb") as output_file:
        for part_file_name in part_names:
            with open(part_file_name, "rb") as part_file:
                while True:
                    chunk = part_file.read(16 * DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    control.check()
                    output_file.write(chunk)
                    hash_sha256.update(chunk)
                    merged += len(chunk)
            os.remove(part_file_name)
    if merged != file_size:
        tmp_target.unlink(missing_ok=True)
        raise FileSizeMismatchError(f"Merged {merged} of {file_size} bytes for {target.name} from ranged parts")
    actual = hash_sha256.hexdigest()
    if expected_sha256 and actual != expected_sha256:
        tmp_target.unlink(missing_ok=True)
        raise FileIntegrityError(
            f"Hash mismatch for {target.name}: expected {expected_sha256[:16]}..., got {actual[:16]}..."
        )
    control.check()
    tmp_target.replace(target)
    return actual


# ---------------------------------------------------------------------------
# Download manager
# ---------------------------------------------------------------------------
class DownloadManager:
    """Internal file download implementation.

    Dependencies are injected via constructor to keep this class testable.
    """

    def __init__(
        self,
        legacy_client: LegacyClient,
        config: HubConfig,
        *,
        user_agent: dict | str | None = None,
        total_timeout: float | None = None,
    ) -> None:
        from .utils import build_user_agent

        self._client = legacy_client
        self._config = config
        self._user_agent = build_user_agent(config=config, extra=user_agent)
        # None defers to MODELSCOPE_DOWNLOAD_TOTAL_TIMEOUT; 0 or less means no limit.
        self._total_timeout = total_timeout
        self._cached_region: str | None = None
        self._region_lock = threading.Lock()
        # Cache only confirmed OSS-internal resolutions. A repository can contain
        # Git metadata and LFS objects behind different download routes, so cache
        # entries are scoped to repo/type/revision plus the redirect endpoint.
        self._inter_region_cache: dict[tuple[str, str, str, str], tuple[str | None, str]] = {}
        # Guard the cache and the per-route in-flight probes. The probe itself is
        # deliberately performed outside this lock so distinct storage routes can
        # resolve in parallel.
        self._inter_region_lock = threading.Lock()
        self._inter_region_inflight: dict[tuple[str, str, str, str], threading.Event] = {}

    # ------------------------------------------------------------------
    # User-agent & headers
    # ------------------------------------------------------------------
    def _build_user_agent(self, user_agent: dict | str | None = None) -> str:
        from .utils import append_user_agent

        return append_user_agent(self._user_agent, user_agent)

    def _detect_region(self) -> str:
        """Probe the region only once per DownloadManager instance."""
        with self._region_lock:
            if self._cached_region is None:
                self._cached_region = ""
                self._cached_region = self._detect_region_uncached()
            return self._cached_region

    def _detect_region_uncached(self) -> str:
        endpoint = self._client.endpoint
        internal_url = f"{endpoint}/api/v1/repos/internalAccelerationInfo"

        def _get(url: str, timeout: float):
            try:
                with _DownloadControl(total_timeout=timeout) as probe:
                    with self._client.read_context(probe):
                        resp = self._client._read_session.get(url, stream=True, timeout=timeout)
                        probe.track(resp)
                        with probe.response(resp):
                            resp.raise_for_status()
                            resp._content = b"".join(probe.chunks(resp, 8192))
                            resp._content_consumed = True
                        return resp
            except (requests.exceptions.RequestException, _DownloadDeadlineError):
                return None

        region_id = ""
        resp = _get(internal_url, 0.2)
        if resp is not None:
            data = resp.json()
            query_addr = ""
            if "Data" in data:
                query_addr = data["Data"].get("InternalRegionQueryAddress", "")
            if query_addr:
                domain_resp = _get(query_addr, 0.2)
                if domain_resp is not None:
                    region_id = domain_resp.text.strip()

        self._cached_region = region_id
        return region_id

    # OSS internal endpoint hostname pattern:
    #   <prefix>.oss<region>-internal.aliyuncs.com
    # e.g. modelhub-cn-hangzhou.oss-cn-hangzhou-internal.aliyuncs.com
    _OSS_INTERNAL_RE = re.compile(r".*\.oss.*-internal\.aliyuncs\.com$")

    @staticmethod
    def _is_oss_internal_url(url: str) -> bool:
        """Check if a URL points to an OSS internal (intranet) endpoint.

        Matches hostnames like:
            modelhub-cn-hangzhou.oss-cn-hangzhou-internal.aliyuncs.com
        """
        try:
            host = urlparse(url).hostname or ""
        except Exception:
            return False
        return bool(DownloadManager._OSS_INTERNAL_RE.match(host))

    @staticmethod
    def _probe_redirect_url(
        url: str,
        headers: dict[str, str],
        cookies: dict | None = None,
        timeout: float = 5.0,
        control: _DownloadControl | None = None,
    ) -> str:
        """Send a HEAD request without following redirects to get the 302 Location."""
        if control is not None:
            control.check()
            timeout = min(timeout, max(0.001, control.deadline - time.monotonic()))
        try:
            with (
                _DownloadControl(
                    total_timeout=timeout,
                    cancel_event=control.cancel_event if control is not None else None,
                ) as probe,
                _network_context(probe),
                requests.Session() as session,
            ):
                session.mount("https://", _ReadHTTPAdapter(max_retries=0))
                session.mount("http://", _ReadHTTPAdapter(max_retries=0))
                with session.head(url, headers=headers, cookies=cookies, allow_redirects=False, timeout=timeout) as r:
                    if r.status_code in (301, 302, 303, 307, 308):
                        return r.headers.get("Location", "")
        except (requests.exceptions.RequestException, _DownloadDeadlineError):
            pass
        return ""

    @staticmethod
    def _storage_route_key(redirect_url: str) -> str | None:
        """Return a stable identity for the storage endpoint behind a redirect.

        Presigned query parameters and object paths are intentionally excluded:
        they differ per file and request, whereas the endpoint hostname is the
        stable route boundary that must isolate inter-region cache entries.
        """
        try:
            parsed = urlparse(redirect_url)
            host = parsed.hostname
            if not parsed.scheme or not host:
                return None
            port = f":{parsed.port}" if parsed.port is not None else ""
            return f"{parsed.scheme.lower()}://{host.lower()}{port}"
        except (TypeError, ValueError):
            return None

    def _get_inter_cloud_regions(self) -> list[str]:
        """Read the inter-cloud peer region list from the environment."""
        from .constants import _env

        raw = _env(ENV_INTER_CLOUD_REGIONS, "INTER_CLOUD_ACCELERATION_REGIONS") or ""
        return [r.strip().lower() for r in raw.split(",") if r.strip()]

    def _resolve_inter_region_headers(
        self,
        url: str,
        headers: dict[str, str],
        cookies: dict | None = None,
        peer_regions: list[str] | None = None,
        initial_redirect_url: str | None = None,
        control: _DownloadControl | None = None,
    ) -> tuple[dict[str, str], str]:
        """Probe peer regions and return headers with the best region for OSS internal download.

        Steps:
        1. Probe with current (local) region — if already OSS internal, return as-is.
        2. Try each peer region in order (skip duplicates of local) — first OSS internal hit wins.
        3. If all miss, fall back to original headers (default path).

        Returns:
            (headers, source) where source is "local", "peer", or "default".
        """
        if peer_regions is None:
            peer_regions = self._get_inter_cloud_regions()
        if not peer_regions:
            return headers, "default"

        current_region = headers.get("x-aliyun-region-id", "").lower()

        # Step 1: Probe with current (local) region. Callers that already
        # established the storage route pass the initial redirect to avoid an
        # identical second HEAD request.
        redirect_url = (
            self._probe_redirect_url(url, headers, cookies, control=control)
            if initial_redirect_url is None
            else initial_redirect_url
        )
        if self._is_oss_internal_url(redirect_url):
            logger.debug("Inter-region: local region already yields OSS internal URL, skipping.")
            return headers, "local"

        # Step 2: Try each peer region in order
        for peer_region in peer_regions:
            if peer_region == current_region:
                continue
            probe_headers = {**headers, "x-aliyun-region-id": peer_region}
            redirect_url = self._probe_redirect_url(url, probe_headers, cookies, control=control)
            if self._is_oss_internal_url(redirect_url):
                logger.debug(
                    'Inter-region acceleration: using peer region "%s" (OSS internal).',
                    peer_region,
                )
                return probe_headers, "peer"

        # Step 3: All peers missed
        logger.debug("Inter-region: no peer region yields OSS internal URL, using default path.")
        return headers, "default"

    def _build_download_headers(
        self,
        user_agent: dict | str | None = None,
        *,
        accelerate: bool = True,
    ) -> dict[str, str]:
        from .constants import _env, _env_bool

        headers: dict[str, str] = {
            "user-agent": self._build_user_agent(user_agent),
            "snapshot-identifier": uuid.uuid4().hex,
            "Accept-Encoding": "identity",
        }
        if accelerate and _env_bool(ENV_INTRA_CLOUD_ACCELERATION, True):
            region = (_env(ENV_INTRA_CLOUD_REGION, "INTRA_CLOUD_ACCELERATION_REGION") or "").strip()
            if not region:
                try:
                    region = self._detect_region()
                except Exception:
                    region = ""
            if region:
                headers["x-aliyun-region-id"] = region
        return headers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def download_file(
        self,
        repo_id: str,
        repo_type: str,
        file_path: str,
        revision: str | None = None,
        cache_dir: Path | None = None,
        local_dir: Path | None = None,
        force: bool = False,
        expected_sha256: str | None = None,
        local_files_only: bool = False,
        file_size: int | None = None,
        user_agent: dict | str | None = None,
        progress_callbacks: list[type[ProgressCallback]] | None = None,
        _cancel_event: threading.Event | None = None,
        _recovery: _RecoveryBudget | None = None,
    ) -> Path:
        """Download one file with cache-first lookup and a shared budget for locks, probes, and retries."""
        with _DownloadControl(
            total_timeout=DOWNLOAD_TOTAL_TIMEOUT if self._total_timeout is None else self._total_timeout,
            cancel_event=_cancel_event,
            min_rate=DOWNLOAD_MIN_BYTES_PER_SECOND,
            speed_window=_recovery.policy.low_speed_window
            if _recovery is not None
            else DOWNLOAD_LOW_SPEED_WINDOW_SECONDS,
            header_timeout=DOWNLOAD_TIMEOUT,
            recovery=_recovery,
            budget_label="Per-file download",
            budget_env="MODELSCOPE_DOWNLOAD_TOTAL_TIMEOUT",
        ) as control:
            with self._client.read_context(control):
                return self._download_file(
                    repo_id,
                    repo_type,
                    file_path,
                    revision,
                    cache_dir,
                    local_dir,
                    force,
                    expected_sha256,
                    local_files_only,
                    file_size,
                    user_agent,
                    progress_callbacks,
                    _control=control,
                )

    def _download_file(
        self,
        repo_id: str,
        repo_type: str,
        file_path: str,
        revision: str | None = None,
        cache_dir: Path | None = None,
        local_dir: Path | None = None,
        force: bool = False,
        expected_sha256: str | None = None,
        local_files_only: bool = False,
        file_size: int | None = None,
        user_agent: dict | str | None = None,
        progress_callbacks: list[type[ProgressCallback]] | None = None,
        *,
        _control: _DownloadControl,
    ) -> Path:
        """Download a single file from a repository.

        When *local_dir* is provided the file is placed directly under that
        directory (preserving relative path structure).  Otherwise the
        standard cache layout is used::

            {cache_dir}/{type}s/{owner}--{name}/snapshots/{revision}/{file_path}

        Parameters
        ----------
        repo_id:
            Repository identifier (``owner/name``).
        repo_type:
            One of the :class:`~.constants.RepoType` values.
        file_path:
            Path within the repository.
        revision:
            Branch, tag, or commit hash.
        cache_dir:
            Override for the default cache directory.
        local_dir:
            When set, download directly into this directory instead of cache.
        force:
            Re-download even if file exists in cache.
        expected_sha256:
            When provided, verify downloaded file hash and use it for
            cache hit validation.
        local_files_only:
            When ``True``, return the cached path without network access.
            Raises ``ValueError`` if the file is not cached.
        file_size:
            Known file size, e.g. from the repository listing. Received bytes
            and cached files are validated against it, and it enables parallel
            range download for large files.
        user_agent:
            Custom user-agent info appended to the default UA string.
        progress_callbacks:
            List of :class:`ProgressCallback` *classes* (not instances).
            Each class is instantiated per-file with ``(filename, file_size)``.

        Returns
        -------
        Path
            Absolute path to the downloaded (or cached) file on disk.

        Raises
        ------
        FileSizeMismatchError
            When the server keeps delivering a size other than *file_size*.
        FileIntegrityError
            When the content keeps failing the *expected_sha256* check. A
            rejected download never remains at the target path.
        """
        if local_files_only:
            return self._resolve_cached_file(
                repo_id,
                repo_type,
                file_path,
                revision=revision,
                cache_dir=cache_dir,
                local_dir=local_dir,
            )

        effective_revision = revision or "master"
        if local_dir is not None:
            target = Path(local_dir) / file_path
        else:
            # Fallback: reuse old SDK (<=1.37) cache if it exists
            legacy = self._find_legacy_repo_dir(repo_id, repo_type, cache_dir)
            if legacy is not None:
                target = legacy / file_path
            else:
                root = self._repo_cache_dir(repo_id, repo_type, cache_dir)
                target = root / "snapshots" / effective_revision / file_path

        if not force and self._cache_hit(target, expected_sha256, file_size, control=_control):
            return target

        use_lock = _file_lock_enabled()
        lock_path = self._lock_path(repo_id, repo_type, cache_dir=cache_dir, file_path=file_path) if use_lock else None
        with _optional_file_lock(lock_path, enabled=use_lock, control=_control):
            # Re-check after acquiring lock — another process may have finished.
            if not force and self._cache_hit(target, expected_sha256, file_size, control=_control):
                return target

            ensure_dir(target.parent)
            if not force:
                # Whatever is at target failed validation. Drop it now so a failed
                # download cannot leave it looking finished.
                target.unlink(missing_ok=True)

            # A hash mismatch gets the full retry budget. A size mismatch means
            # the server consistently served the wrong object, so it only gets
            # one more attempt, over the default route.
            failures = {"hash": 0, "size": 0}
            limits = {
                "hash": min(DOWNLOAD_HASH_RETRY_TIMES, 2)
                if _control.recovery is not None
                else DOWNLOAD_HASH_RETRY_TIMES,
                "size": DOWNLOAD_SIZE_RETRY_TIMES,
            }
            prefer_default_route = False
            while True:
                _control.check()
                try:
                    self._download_with_resume(
                        repo_id,
                        repo_type,
                        file_path,
                        effective_revision,
                        target,
                        file_size=file_size,
                        user_agent=user_agent,
                        progress_callbacks=progress_callbacks,
                        prefer_default_route=prefer_default_route,
                        control=_control,
                        expected_sha256=expected_sha256,
                    )
                    break
                except FileIntegrityError as exc:
                    kind = "size" if isinstance(exc, FileSizeMismatchError) else "hash"
                    # A size mismatch is raised before the rename, so the target is
                    # still the previous copy (only possible with force).
                    self._discard_download(target, keep_target=True)
                    target.with_suffix(target.suffix + ".parallel_tmp").unlink(missing_ok=True)
                    failures[kind] += 1
                    if failures[kind] >= limits[kind]:
                        raise
                    if kind == "size":
                        prefer_default_route = True
                    delay = DOWNLOAD_INTEGRITY_RETRY_BACKOFF * 2 ** (sum(failures.values()) - 1)
                    logger.warning(
                        "%s validation failed for %s, retrying in %.0fs (%d/%d): %s",
                        kind.capitalize(),
                        file_path,
                        delay,
                        failures[kind],
                        limits[kind],
                        exc.message,
                    )
                    if _control.recovery is not None:
                        try:
                            _control.recovery.retry(delay)
                        except Exception as budget_error:
                            raise budget_error from exc
                    _control.sleep(delay)

        return target

    def download_repo_report(
        self,
        repo_id: str,
        repo_type: str,
        *,
        revision: str | None = None,
        cache_dir: Path | None = None,
        local_dir: Path | None = None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        max_workers: int = 4,
        file_list: Iterable[dict] | None = None,
        tracker_path: Path | None = None,
        retry_failed: bool = False,
        retry_policy: DownloadRetryPolicy | None = None,
        user_agent: dict | str | None = None,
        progress_callbacks: list[type[ProgressCallback]] | None = None,
    ) -> DownloadReport:
        """Continue after isolated file failures and return a durable completion report."""
        from ._resilient_download import download_repo_report

        return download_repo_report(
            self,
            repo_id,
            repo_type,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=max_workers,
            file_list=file_list,
            tracker_path=tracker_path,
            retry_failed=retry_failed,
            retry_policy=retry_policy,
            user_agent=user_agent,
            progress_callbacks=progress_callbacks,
        )

    def download_repo(
        self,
        repo_id: str,
        repo_type: str,
        revision: str | None = None,
        cache_dir: Path | None = None,
        local_dir: Path | None = None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        max_workers: int = 4,
        local_files_only: bool = False,
        user_agent: dict | str | None = None,
        progress_callbacks: list[type[ProgressCallback]] | None = None,
        file_list: list[dict] | None = None,
    ) -> Path:
        """Download an entire repository (snapshot download).

        Parameters
        ----------
        repo_id:
            Repository identifier (``owner/name``).
        repo_type:
            One of the :class:`~.constants.RepoType` values.
        revision:
            Branch, tag, or commit hash.
        cache_dir:
            Override for the default cache directory.
        local_dir:
            When set, download directly into this directory instead of cache.
        allow_patterns:
            Only files matching these globs will be downloaded.
        ignore_patterns:
            Files matching these globs will be skipped.
        max_workers:
            Number of parallel download threads.
        local_files_only:
            When ``True``, return the cached snapshot path without network.
        user_agent:
            Custom user-agent info for download headers.
        progress_callbacks:
            List of :class:`ProgressCallback` *classes* (not instances).

        Returns
        -------
        Path
            Absolute path to the snapshot/local directory.

        Raises
        ------
        FileDownloadError
            After every file was attempted, when any of them failed. Completed
            files stay in place; ``failed_files`` maps each failed path to its
            error.
        """
        if local_files_only:
            return self._resolve_cached_snapshot(
                repo_id,
                repo_type,
                revision=revision,
                cache_dir=cache_dir,
                local_dir=local_dir,
            )

        effective_revision = revision or "master"
        if local_dir is not None:
            output_dir = ensure_dir(Path(local_dir))
        else:
            # Fallback: reuse old SDK (<=1.37) cache if it exists
            legacy = self._find_legacy_repo_dir(repo_id, repo_type, cache_dir)
            if legacy is not None:
                logger.info(
                    "Found legacy cache at %s, reusing.",
                    legacy,
                )
                output_dir = legacy
                local_dir = legacy
            else:
                root = self._repo_cache_dir(repo_id, repo_type, cache_dir)
                output_dir = ensure_dir(root / "snapshots" / effective_revision)

        if repo_type in ("skill", "skills"):
            return self._download_archive(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=effective_revision,
                output_dir=output_dir,
            )

        cancel_event = threading.Event()
        if file_list is not None:
            files = file_list
        else:
            files = []
            with _DownloadControl(
                total_timeout=REPO_TREE_LIST_TIMEOUT_SECONDS,
                cancel_event=cancel_event,
                budget_label="Repository listing",
                budget_env="MODELSCOPE_REPO_TREE_LIST_TIMEOUT_SECONDS",
            ) as listing_control:
                with self._client.read_context(listing_control):
                    for listing_root in _listing_roots(allow_patterns):
                        listing_control.check()
                        try:
                            if repo_type in ("dataset", "datasets"):
                                files.extend(
                                    self._client.list_dataset_files_paginated(
                                        repo_id=repo_id,
                                        revision=effective_revision,
                                        root_path=listing_root or "/",
                                    )
                                )
                            else:
                                files.extend(
                                    self._client.list_repo_files(
                                        repo_id=repo_id,
                                        repo_type=repo_type,
                                        revision=effective_revision,
                                        recursive=True,
                                        root=listing_root,
                                    )
                                )
                        except NotExistError:
                            # A missing literal directory has no matches; a missing repository must still raise.
                            if listing_root is None:
                                raise
                            self._client.get_repo_info(repo_id, repo_type)

        download_items: list[tuple[str, str | None, int | None]] = []
        for f in files:
            path = f.get("Path") or f.get("path") or f.get("Name") or ""
            ftype = f.get("Type") or f.get("type") or "blob"
            if ftype == "tree":
                continue
            if not path:
                continue
            if allow_patterns and not _matches_patterns(path, allow_patterns):
                continue
            if ignore_patterns and _matches_patterns(path, ignore_patterns):
                continue
            sha256 = f.get("Sha256") or f.get("sha256") or None
            raw_size = f.get("Size", f.get("size"))
            size = int(raw_size) if raw_size is not None else None
            download_items.append((path, sha256, size))

        if not download_items:
            logger.info("No files to download for %s@%s", repo_id, effective_revision)
            return output_dir

        logger.info("Downloading %d files from %s@%s", len(download_items), repo_id, effective_revision)

        failures: dict[str, BaseException] = {}
        executor = ThreadPoolExecutor(max_workers=max_workers)
        items = iter(download_items)
        futures = {}

        def submit_next() -> None:
            item = next(items, None)
            if item is None:
                return
            fp, sha256, size = item
            future = executor.submit(
                self.download_file,
                repo_id=repo_id,
                repo_type=repo_type,
                file_path=fp,
                revision=effective_revision,
                cache_dir=cache_dir,
                local_dir=local_dir,
                expected_sha256=sha256,
                file_size=size,
                user_agent=user_agent,
                progress_callbacks=progress_callbacks,
                _cancel_event=cancel_event,
            )
            futures[future] = fp

        try:
            # Bound pending tasks instead of creating millions of Futures; stop scheduling after cancellation.
            for _ in range(max_workers * 2):
                submit_next()
            with tqdm(total=len(download_items), desc="Downloading", unit="file") as pbar:
                while futures:
                    completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in completed:
                        fp = futures.pop(future)
                        try:
                            future.result()
                        except Exception as exc:
                            failures[fp] = exc
                            logger.error("Failed to download %s: %s", fp, exc)
                        finally:
                            pbar.update(1)
                        submit_next()
        except BaseException:
            cancel_event.set()
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        if failures:
            # Returning normally would let callers treat a partial snapshot as
            # complete. Every file was attempted first, so a re-run only fetches
            # the failed ones.
            raise FileDownloadError(
                f"{len(failures)} of {len(download_items)} file(s) failed to download from "
                f"{repo_id}@{effective_revision}",
                failed_files=failures,
                total=len(download_items),
                output_dir=output_dir,
            )

        return output_dir

    # ------------------------------------------------------------------
    # Internal: archive-based download (skills)
    # ------------------------------------------------------------------
    def _download_archive(
        self,
        repo_id: str,
        repo_type: str,
        revision: str,
        output_dir: Path,
    ) -> Path:
        """Download a repo via its zip archive endpoint and extract.

        Skill repos do not support per-file ``/repo?FilePath=...`` download.
        The old SDK uses ``/archive/zip/{revision}`` for these.
        """
        import shutil
        import tempfile
        import zipfile

        tmp_path: Path | None = None
        try:
            with (
                _DownloadControl(
                    total_timeout=DOWNLOAD_TOTAL_TIMEOUT if self._total_timeout is None else self._total_timeout,
                    min_rate=DOWNLOAD_MIN_BYTES_PER_SECOND,
                    speed_window=DOWNLOAD_LOW_SPEED_WINDOW_SECONDS,
                    header_timeout=DOWNLOAD_TIMEOUT,
                    budget_label="Archive download",
                    budget_env="MODELSCOPE_DOWNLOAD_TOTAL_TIMEOUT",
                ) as control,
                self._client.read_context(control),
            ):
                retry = Retry(total=DOWNLOAD_RETRY_TIMES, backoff_factor=1, allowed_methods=["GET"])
                while True:
                    try:
                        resp = self._client.download_archive(
                            repo_id=repo_id,
                            repo_type=repo_type,
                            revision=revision,
                        )
                        with control.response(resp), tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                            tmp_path = Path(tmp.name)
                            for chunk in control.chunks(resp, DOWNLOAD_CHUNK_SIZE):
                                tmp.write(chunk)
                        announced = _checkable_content_length(resp)
                        if announced is not None and tmp_path.stat().st_size != announced:
                            raise _IncompleteBodyError(
                                "Archive body length does not match the announced Content-Length"
                            )
                        break
                    except (_DownloadCancelledError, _DownloadDeadlineError):
                        raise
                    except Exception as exc:
                        if tmp_path is not None:
                            tmp_path.unlink(missing_ok=True)
                        retry = _wait_download_retry(retry, exc, repo_id, control)

            with zipfile.ZipFile(tmp_path, "r") as zf:
                zf.extractall(output_dir)

            # Flatten if zip has a single top-level directory
            entries = [e for e in output_dir.iterdir()]
            if len(entries) == 1 and entries[0].is_dir():
                nested = entries[0]
                for item in nested.iterdir():
                    shutil.move(str(item), str(output_dir / item.name))
                nested.rmdir()
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        logger.info("Extracted archive for %s to %s", repo_id, output_dir)
        return output_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _cache_hit(
        target: Path,
        expected_sha256: str | None,
        expected_size: int | None = None,
        *,
        control: _DownloadControl | None = None,
    ) -> bool:
        """Return True if *target* exists and passes the optional size and hash checks."""
        if not target.exists():
            return False
        if expected_size is not None and target.stat().st_size != expected_size:
            logger.debug("Cache stale (size mismatch): %s", target)
            return False
        if expected_sha256:
            if control is None:
                actual = compute_hash(target, "sha256")
            else:
                digest = hashlib.sha256()
                with target.open("rb") as cached:
                    while block := cached.read(8 * DOWNLOAD_CHUNK_SIZE):
                        control.check()
                        digest.update(block)
                actual = digest.hexdigest()
            if actual != expected_sha256:
                logger.debug("Cache stale (hash mismatch): %s", target)
                return False
            logger.debug("Cache hit (hash verified): %s", target)
        else:
            logger.debug("Cache hit: %s", target)
        return True

    @staticmethod
    def _discard_download(target: Path, *, keep_target: bool = False) -> None:
        """Remove a rejected download so neither it nor its partial file looks reusable."""
        _incomplete_path(target).unlink(missing_ok=True)
        if not keep_target:
            target.unlink(missing_ok=True)

    @staticmethod
    def _dir_has_entries(path: Path) -> bool:
        """Return True when a directory exists and contains at least one entry."""
        if not path.is_dir():
            return False
        try:
            next(path.iterdir())
            return True
        except (StopIteration, OSError):
            return False

    @classmethod
    def _cached_snapshot_dirs(cls, snapshots_dir: Path) -> list[Path]:
        """Return non-empty cached snapshot directories sorted by revision name."""
        if not snapshots_dir.is_dir():
            return []
        try:
            children = sorted(snapshots_dir.iterdir(), key=lambda p: p.name)
        except OSError:
            return []
        return [path for path in children if cls._dir_has_entries(path)]

    @classmethod
    def _describe_cached_revisions(cls, snapshots_dir: Path) -> str:
        """Describe the revisions physically present in a snapshots directory."""
        if not snapshots_dir.is_dir():
            return f"No snapshots directory found at: {snapshots_dir}"
        try:
            children = sorted((path for path in snapshots_dir.iterdir() if path.is_dir()), key=lambda p: p.name)
        except OSError as exc:
            return f"Unable to inspect cached revisions under {snapshots_dir}: {exc}"
        if not children:
            return f"No cached revisions found under: {snapshots_dir}"
        shown = [f"{path.name}{' (empty)' if not cls._dir_has_entries(path) else ''}" for path in children[:10]]
        if len(children) > 10:
            shown.append(f"... {len(children) - 10} more")
        return f"Cached revisions under {snapshots_dir}: {', '.join(shown)}"

    def _resolve_cached_snapshot(
        self,
        repo_id: str,
        repo_type: str,
        *,
        revision: str | None,
        cache_dir: Path | None,
        local_dir: Path | None,
    ) -> Path:
        """Resolve a cached snapshot without touching the network or creating directories."""
        if local_dir is not None:
            output_dir = Path(local_dir)
            if self._dir_has_entries(output_dir):
                return output_dir
            raise CacheNotFound(
                "Cannot find cached snapshot in the requested local directory and outgoing traffic has been disabled. "
                f"Local directory: {output_dir}",
                cache_dir=str(output_dir),
            )

        legacy = self._find_legacy_repo_dir(repo_id, repo_type, cache_dir)
        if legacy is not None:
            logger.info("Found legacy cache at %s, reusing.", legacy)
            return legacy

        root = self._repo_cache_dir_path(repo_id, repo_type, cache_dir)
        snapshots = root / "snapshots"
        if revision:
            output_dir = snapshots / revision
            if self._dir_has_entries(output_dir):
                return output_dir
            raise CacheNotFound(
                f"Cannot find cached snapshot for {repo_type} repo '{repo_id}' at revision '{revision}' and outgoing "
                f"traffic has been disabled. {self._describe_cached_revisions(snapshots)}",
                cache_dir=str(output_dir),
            )

        for default_name in ("master", "main"):
            default_snapshot = snapshots / default_name
            if self._dir_has_entries(default_snapshot):
                return default_snapshot

        candidates = self._cached_snapshot_dirs(snapshots)
        if len(candidates) == 1:
            logger.warning(
                "Using cached revision %s because no revision was provided and outgoing traffic is disabled.",
                candidates[0].name,
            )
            return candidates[0]
        if candidates:
            revisions = ", ".join(path.name for path in candidates[:10])
            if len(candidates) > 10:
                revisions += f", ... {len(candidates) - 10} more"
            raise CacheNotFound(
                f"Cannot determine which cached snapshot to use for {repo_type} repo '{repo_id}' because no revision "
                f"was provided and outgoing traffic has been disabled. Available cached revisions: {revisions}. "
                "Pass 'revision' to select one.",
                cache_dir=str(snapshots),
            )
        raise CacheNotFound(
            f"Cannot find cached snapshot for {repo_type} repo '{repo_id}' and outgoing traffic has been disabled. "
            f"{self._describe_cached_revisions(snapshots)}",
            cache_dir=str(snapshots),
        )

    def _resolve_cached_file(
        self,
        repo_id: str,
        repo_type: str,
        file_path: str,
        *,
        revision: str | None,
        cache_dir: Path | None,
        local_dir: Path | None,
    ) -> Path:
        """Resolve a cached file without touching the network or creating directories."""
        if local_dir is not None:
            target = Path(local_dir) / file_path
            if target.exists():
                return target
            raise CacheNotFound(
                f"Cannot find cached file '{file_path}' in the requested local directory and outgoing traffic has "
                f"been disabled. Local directory: {Path(local_dir)}",
                cache_dir=str(target.parent),
            )

        legacy = self._find_legacy_repo_dir(repo_id, repo_type, cache_dir)
        if legacy is not None:
            target = legacy / file_path
            if target.exists():
                return target

        root = self._repo_cache_dir_path(repo_id, repo_type, cache_dir)
        snapshots = root / "snapshots"
        if revision:
            target = snapshots / revision / file_path
            if target.exists():
                return target
            raise CacheNotFound(
                f"Cannot find cached file '{file_path}' for {repo_type} repo '{repo_id}' at revision '{revision}' "
                f"and outgoing traffic has been disabled. {self._describe_cached_revisions(snapshots)}",
                cache_dir=str(target.parent),
            )

        for default_name in ("master", "main"):
            target = snapshots / default_name / file_path
            if target.exists():
                return target

        matches = []
        for snapshot in self._cached_snapshot_dirs(snapshots):
            candidate = snapshot / file_path
            if candidate.exists():
                matches.append(candidate)
        if len(matches) == 1:
            logger.warning(
                "Using cached file from revision %s because no revision was provided and outgoing traffic is disabled.",
                matches[0].relative_to(snapshots).parts[0],
            )
            return matches[0]
        if matches:
            revisions = ", ".join(path.relative_to(snapshots).parts[0] for path in matches[:10])
            if len(matches) > 10:
                revisions += f", ... {len(matches) - 10} more"
            raise CacheNotFound(
                f"Cannot determine which cached file to use for {repo_type} repo '{repo_id}' because no revision was "
                f"provided and outgoing traffic has been disabled. File '{file_path}' exists in cached revisions: "
                f"{revisions}. Pass 'revision' to select one.",
                cache_dir=str(snapshots),
            )
        raise CacheNotFound(
            f"Cannot find cached file '{file_path}' for {repo_type} repo '{repo_id}' and outgoing traffic has been "
            f"disabled. {self._describe_cached_revisions(snapshots)}",
            cache_dir=str(snapshots),
        )

    def _repo_cache_dir_path(
        self,
        repo_id: str,
        repo_type: str,
        cache_dir: Path | None = None,
    ) -> Path:
        """Compute the repo cache directory path without creating it."""
        base = cache_dir or self._config.cache_dir
        segment = f"{repo_type}s" if not repo_type.endswith("s") else repo_type
        safe_id = repo_id.replace("/", "--")
        return base / segment / safe_id

    def _repo_cache_dir(
        self,
        repo_id: str,
        repo_type: str,
        cache_dir: Path | None = None,
    ) -> Path:
        """Compute the cache directory for a given repo."""
        return ensure_dir(self._repo_cache_dir_path(repo_id, repo_type, cache_dir))

    def _find_legacy_repo_dir(
        self,
        repo_id: str,
        repo_type: str,
        cache_dir: Path | None = None,
    ) -> Path | None:
        """Check for old SDK (<=1.37) cache layout and return it if non-empty.

        Old SDKs encoded dots in the repo name as ``___`` and, when no
        ``MODELSCOPE_CACHE`` was set, stored everything under a ``hub/``
        sub-directory of the cache root. Both historical layouts are probed,
        in priority order::

            {base}/{type}s/{owner}/{name___}        # MODELSCOPE_CACHE explicitly set
            {base}/hub/{type}s/{owner}/{name___}    # default cache (~/.cache/modelscope/hub)

        e.g.  ~/.cache/modelscope/hub/models/Qwen/Qwen3___5-0___8B/

        Returns the first existing, non-empty candidate, or ``None`` when the
        cache is clean (so the caller falls back to the new layout).
        """
        base = cache_dir or self._config.cache_dir
        segment = f"{repo_type}s" if not repo_type.endswith("s") else repo_type
        parts = repo_id.split("/", 1)
        if len(parts) != 2:
            return None
        owner, name = parts
        safe_name = name.replace(".", "___")
        candidates = (
            base / segment / owner / safe_name,
            base / "hub" / segment / owner / safe_name,
        )
        for legacy_path in candidates:
            if not legacy_path.is_dir():
                continue
            try:
                if any(legacy_path.iterdir()):
                    return legacy_path
            except OSError:
                continue
        return None

    def _lock_path(
        self,
        repo_id: str,
        repo_type: str,
        cache_dir: Path | None = None,
        file_path: str | None = None,
    ) -> Path:
        """Compute a fixed-length lock file path for a repo or file.

        File systems commonly cap a single path component at 255 bytes. Older
        versions embedded the full ``file_path`` in the lock basename, so a
        perfectly valid remote filename could make the local lock filename too
        long before the download even started. Keep the lock *key* semantics
        (repo type + repo id + optional file path), but store only a stable
        SHA-256 digest in the basename.
        """
        base = cache_dir or self._config.cache_dir
        scope = "file" if file_path is not None else "repo"
        key = "\0".join(
            (
                "modelscope-hub-download-lock-v2",
                scope,
                str(repo_type),
                repo_id,
                file_path or "",
            )
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return base / ".lock" / f"{scope}_{digest}.lock"

    def _download_with_resume(
        self,
        repo_id: str,
        repo_type: str,
        file_path: str,
        revision: str,
        target: Path,
        *,
        file_size: int | None = None,
        user_agent: dict | str | None = None,
        progress_callbacks: list[type[ProgressCallback]] | None = None,
        prefer_default_route: bool = False,
        control: _DownloadControl,
        expected_sha256: str | None = None,
    ) -> Path:
        """Download a file with HTTP Range resume support and retry.

        Before the partial file is moved into place, the bytes received are
        checked against the announced ``Content-Length`` and, when *file_size*
        is known, against the repository listing. A short body is resumed; a
        body whose size contradicts the listing raises
        :class:`~.errors.FileSizeMismatchError` and leaves no partial file.
        *prefer_default_route* skips intra-cloud and peer-region routing so a
        retry can avoid a replica that served bad content.
        """
        use_parallel = file_size is not None and file_size > DOWNLOAD_PARALLEL_THRESHOLD and DOWNLOAD_PARALLELS > 1

        download_headers = self._build_download_headers(user_agent, accelerate=not prefer_default_route)

        # Inter-region acceleration: first discover the file's storage route,
        # then cache only a confirmed internal resolution for that exact route.
        # Git metadata may return 200 without a redirect while LFS objects use
        # OSS; a metadata miss must therefore never suppress an LFS probe.
        # Progress bar prefix: "⚡ " = local OSS, "⇄ " = peer OSS, "  " = CDN, "" = not configured
        source_prefix = ""
        peer_regions = [] if prefer_default_route else self._get_inter_cloud_regions()
        if peer_regions:
            source = "default"
            cached: tuple[str | None, str] | None = None
            probe_event: threading.Event | None = None
            try:
                probe_url = self._client.get_download_url(repo_id, repo_type, file_path, revision)
                cookies = {"m_session_id": self._client.token} if self._client.token else None
                initial_redirect_url = self._probe_redirect_url(probe_url, download_headers, cookies, control=control)
                route_key = self._storage_route_key(initial_redirect_url)

                if route_key is not None:
                    cache_key = (repo_id, repo_type, revision, route_key)
                    while cached is None:
                        is_probe_owner = False
                        with self._inter_region_lock:
                            cached = self._inter_region_cache.get(cache_key)
                            if cached is not None:
                                break
                            probe_event = self._inter_region_inflight.get(cache_key)
                            if probe_event is None:
                                probe_event = threading.Event()
                                self._inter_region_inflight[cache_key] = probe_event
                                is_probe_owner = True

                        if not is_probe_owner:
                            assert probe_event is not None
                            while not probe_event.wait(0.1):
                                control.check()
                            continue

                        try:
                            download_headers, source = self._resolve_inter_region_headers(
                                probe_url,
                                download_headers,
                                cookies,
                                peer_regions=peer_regions,
                                initial_redirect_url=initial_redirect_url,
                                control=control,
                            )
                            resolved_region = download_headers.get("x-aliyun-region-id")
                            if source != "default":
                                with self._inter_region_lock:
                                    self._inter_region_cache[cache_key] = (resolved_region, source)
                                cached = (resolved_region, source)
                        finally:
                            with self._inter_region_lock:
                                self._inter_region_inflight.pop(cache_key, None)
                                assert probe_event is not None
                                probe_event.set()

                        # A non-internal result is intentionally not cached. The
                        # current file falls back now; a waiting file retries as
                        # the next single-flight owner instead of inheriting it.
                        if cached is None:
                            break

                    if cached is not None:
                        resolved_region, source = cached
                        if resolved_region is not None:
                            download_headers["x-aliyun-region-id"] = resolved_region
            except (_DownloadCancelledError, _DownloadDeadlineError):
                raise
            except Exception as exc:
                logger.warning(
                    "Failed to resolve inter-region acceleration: %s. Falling back to default.",
                    exc,
                )
                source = "default"

            source_prefix = {
                "local": "\u26a1 ",
                "peer": "\u21c4 ",
                "default": "  ",
            }[source]

        def request(headers: dict[str, str]) -> requests.Response:
            control.check()
            with self._client.read_context(control):
                return self._client.download_stream(
                    repo_id,
                    repo_type,
                    file_path,
                    revision,
                    headers=headers,
                )

        if use_parallel:
            assert file_size is not None  # guaranteed by use_parallel above
            url = self._client.get_download_url(
                repo_id,
                repo_type,
                file_path,
                revision,
            )
            cookies = None
            if self._client.token:
                cookies = {"m_session_id": self._client.token}

            cb_instances = []
            if progress_callbacks:
                cb_instances = [cls(file_path, file_size) for cls in progress_callbacks]

            try:
                _parallel_download(
                    url=url,
                    target=target,
                    file_size=file_size,
                    headers=download_headers,
                    cookies=cookies,
                    progress_callbacks=cb_instances,
                    request=request,
                    control=control,
                    expected_sha256=expected_sha256,
                )
                return target
            except _RangeNotHonouredError as exc:
                logger.warning("Ranged requests are not honoured for %s (%s); using a single stream", file_path, exc)

        # Single-stream download with retry
        cb_classes = list(progress_callbacks or [])
        cb_instances = [cls(file_path, file_size or 0) for cls in cb_classes]
        reported = 0
        # The listing reports 0 both for empty files and for unknown sizes.
        expected_size = file_size

        retry = Retry(
            total=DOWNLOAD_RETRY_TIMES,
            backoff_factor=1,
            allowed_methods=["GET"],
        )
        tmp_path = _incomplete_path(target)

        while True:
            try:
                control.check()
                existing_size = tmp_path.stat().st_size if tmp_path.exists() else 0
                if expected_size is not None and existing_size > expected_size:
                    # Longer than the file itself, so it cannot be a prefix of it.
                    tmp_path.unlink()
                    existing_size = 0
                if existing_size > reported:
                    for cb in cb_instances:
                        cb.update(existing_size - reported)
                    reported = existing_size
                if tmp_path.exists() and expected_size is not None and existing_size == expected_size:
                    # An earlier run received every byte but stopped before the
                    # rename. Resuming would only earn a 416 from the CDN; the
                    # caller's hash check decides whether the bytes are right.
                    _finish_download(tmp_path, target, expected_sha256, control)
                    for cb in cb_instances:
                        cb.end()
                    return target

                request_id = uuid.uuid4().hex
                extra_headers: dict[str, str] = copy.deepcopy(download_headers)
                extra_headers["X-Request-ID"] = request_id
                if existing_size > 0:
                    extra_headers["Range"] = f"bytes={existing_size}-"
                    logger.debug("Resuming download from byte %d", existing_size)

                try:
                    resp = request(extra_headers)
                except requests.Timeout as exc:
                    raise RequestTimeoutError(f"Download timed out for {file_path}: {exc}") from exc
                except requests.ConnectionError as exc:
                    raise NetworkError(f"Download connection failed for {file_path}: {exc}") from exc
                except requests.RequestException as exc:
                    raise NetworkError(f"Download failed for {file_path}: {exc}") from exc
                except APIError as exc:
                    if exc.status_code == 416 and existing_size > 0:
                        # The object does not extend past the partial file, so the
                        # partial file is stale. Start over.
                        tmp_path.unlink(missing_ok=True)
                        raise _IncompleteBodyError(
                            f"server refused to resume from byte {existing_size} (HTTP 416)"
                        ) from exc
                    raise

                with control.response(resp):
                    # The LFS CDN returns the object SHA-256, enabling validation without caller metadata.
                    linked_hash = resp.headers.get("X-Linked-ETag", "").strip('"')
                    if expected_sha256 is None and re.fullmatch(r"[0-9a-fA-F]{64}", linked_hash):
                        expected_sha256 = linked_hash.lower()
                    content_length = _checkable_content_length(resp)
                    content_range = _parse_content_range(resp.headers.get("Content-Range"))
                    if existing_size > 0 and (resp.status_code == 206 or content_range is not None):
                        if content_range is None or content_range[0] != existing_size:
                            tmp_path.unlink(missing_ok=True)
                            raise _IncompleteBodyError(
                                f"resume from byte {existing_size} was answered with "
                                f"{_describe_response(resp, request_id)}; restarting"
                            )
                        mode, offset = "ab", existing_size
                    else:
                        # Anything but a matching 206 carries the whole file: files
                        # served without a redirect ignore Range entirely.
                        mode, offset = "wb", 0
                        if reported:
                            for cb in cb_instances:
                                cb.update(-reported)
                            reported = 0

                    if resp.status_code == 206 and (content_range is None or content_range[0] != offset):
                        tmp_path.unlink(missing_ok=True)
                        raise _IncompleteBodyError(
                            "Server returned a partial response that does not match the requested start offset"
                        )
                    announced: int | None = None
                    if content_range is not None and content_range[2] is not None:
                        announced = content_range[2]
                    elif content_length is not None:
                        announced = offset + content_length
                    if expected_size is not None and announced is not None and announced != expected_size:
                        tmp_path.unlink(missing_ok=True)
                        raise FileSizeMismatchError(
                            f"Server announced {announced} bytes for {file_path} but the repository lists "
                            f"{expected_size} bytes ({_describe_response(resp, request_id)}); the remote object "
                            "is empty, truncated or out of date"
                        )

                    received = 0
                    digest = hashlib.sha256() if expected_sha256 else None
                    if digest is not None and offset:
                        with tmp_path.open("rb") as prefix:
                            while block := prefix.read(8 * DOWNLOAD_CHUNK_SIZE):
                                control.check()
                                digest.update(block)
                    with tqdm(
                        total=announced,
                        initial=offset,
                        unit="B",
                        unit_scale=True,
                        desc=f"{source_prefix}{Path(file_path).name}",
                        leave=False,
                    ) as pbar:
                        with open(tmp_path, mode) as fh:
                            for chunk in control.chunks(resp, DOWNLOAD_CHUNK_SIZE):
                                if chunk:
                                    if expected_size is not None and offset + received + len(chunk) > expected_size:
                                        raise FileSizeMismatchError(
                                            f"Server exceeded {expected_size} bytes for {file_path} "
                                            f"({_describe_response(resp, request_id)})"
                                        )
                                    fh.write(chunk)
                                    if digest is not None:
                                        digest.update(chunk)
                                    received += len(chunk)
                                    reported += len(chunk)
                                    pbar.update(len(chunk))
                                    for cb in cb_instances:
                                        cb.update(len(chunk))

                    if content_length is not None and received < content_length:
                        # urllib3 2.x raises on a short read, but 1.x ends the body
                        # silently; keep the partial file and resume from it.
                        raise _IncompleteBodyError(
                            f"connection closed after {received} of {content_length} bytes "
                            f"({_describe_response(resp, request_id)})"
                        )
                    if expected_size is not None and offset + received != expected_size:
                        tmp_path.unlink(missing_ok=True)
                        raise FileSizeMismatchError(
                            f"Server returned {offset + received} of {expected_size} bytes for {file_path} "
                            f"({_describe_response(resp, request_id)}); the remote object is empty, truncated "
                            "or out of date"
                        )

                _finish_download(
                    tmp_path,
                    target,
                    expected_sha256,
                    control,
                    actual_hash=digest.hexdigest() if digest is not None else None,
                )
                for cb in cb_instances:
                    cb.end()
                logger.debug("Downloaded: %s", target)
                return target

            except (
                NotExistError,
                PermissionDeniedError,
                AuthenticationError,
                FileIntegrityError,
                _DownloadCancelledError,
                _DownloadDeadlineError,
            ):
                for cb in cb_instances:
                    cb.end()
                raise
            except Exception as exc:
                if not prefer_default_route:
                    # Fall back to the public route for this file only, without changing other files' cached routes.
                    download_headers.pop("x-aliyun-region-id", None)
                try:
                    retry = _wait_download_retry(retry, exc, file_path, control)
                except BaseException:
                    for cb in cb_instances:
                        cb.end()
                    raise
            except BaseException:
                for cb in cb_instances:
                    cb.end()
                raise

    def verify_file(self, file_path: Path, expected_sha256: str) -> bool:
        """Verify a downloaded file's SHA256 hash.

        Raises :class:`~.errors.FileIntegrityError` on mismatch.
        """
        actual = compute_hash(file_path, "sha256")
        if actual != expected_sha256:
            raise FileIntegrityError(
                f"Hash mismatch for {file_path.name}: expected {expected_sha256[:16]}..., got {actual[:16]}..."
            )
        return True
