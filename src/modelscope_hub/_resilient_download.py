"""Streaming report-mode downloads with durable file failure isolation."""

from __future__ import annotations

import errno
import hashlib
import json
import queue
import threading
import uuid
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from filelock import FileLock, Timeout

from ._download import _listing_roots, _matches_patterns
from ._download_control import _DownloadCancelledError, _DownloadControl, _DownloadDeferredError, _RecoveryBudget
from ._download_state import _DownloadState, safe_error
from .constants import DOWNLOAD_TOTAL_TIMEOUT, REPO_TREE_LIST_TIMEOUT_SECONDS
from .errors import AuthenticationError, FileIntegrityError, InvalidParameter, NotSupportedError
from .types import DownloadFileResult, DownloadReport, DownloadRetryPolicy
from .utils.logger import get_logger

if TYPE_CHECKING:
    from ._download import DownloadManager, ProgressCallback

logger = get_logger("download")


def _validate_entry(entry: dict, output: Path) -> tuple[str, int | None, str | None]:
    path = entry.get("Path") or entry.get("path") or entry.get("Name")
    if not isinstance(path, str) or not path or "\\" in path or "\0" in path:
        raise InvalidParameter("File entry has no safe repository-relative path")
    parts = PurePosixPath(path)
    if not parts.parts or parts.is_absolute() or ".." in parts.parts or path.endswith("/"):
        raise InvalidParameter(f"Unsafe download path: {path}")
    if not (output / path).resolve().is_relative_to(output.resolve()):
        raise InvalidParameter(f"Download path escapes output directory: {path}")
    raw_size = entry.get("Size", entry.get("size"))
    size = int(raw_size) if raw_size is not None else None
    if size is not None and size < 0:
        raise InvalidParameter(f"Negative file size: {path}")
    sha256 = entry.get("Sha256") or entry.get("sha256")
    if sha256 is not None:
        if not isinstance(sha256, str) or len(sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in sha256):
            raise InvalidParameter(f"Invalid SHA-256 metadata: {path}")
        sha256 = sha256.lower()
    if size is None and sha256 is None:
        raise InvalidParameter(f"Missing size and SHA-256 metadata; refusing an unverifiable download: {path}")
    return path, size, sha256


def download_repo_report(
    manager: DownloadManager,
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
    """Ordinary file failures return PARTIAL after all known work has been processed.

    A tracker is scoped to endpoint, repo, revision and output, not include
    patterns, so repeated ten-file batches share quarantine and retry budgets.
    retry_failed is an explicit operator-authorized reset, never automatic.
    """
    if not repo_id or max_workers < 1:
        raise InvalidParameter("repo_id and a positive max_workers are required")
    if repo_type not in ("model", "dataset"):
        raise NotSupportedError("Report-mode downloads support model and dataset repositories")
    revision = revision or "master"
    revision_path = PurePosixPath(revision)
    if revision_path.is_absolute() or ".." in revision_path.parts:
        raise InvalidParameter("revision must not escape the snapshot cache directory")
    output = local_dir or manager._find_legacy_repo_dir(repo_id, repo_type, cache_dir)
    if output is None:
        output = manager._repo_cache_dir_path(repo_id, repo_type, cache_dir) / "snapshots" / revision
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "endpoint": manager._client.endpoint,
        "repo": repo_id,
        "type": repo_type,
        "revision": revision,
        "output": str(output.resolve()),
        "schema": "1",
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    tracker = tracker_path or (cache_dir or manager._config.cache_dir) / ".download_state" / f".ms_download_{key}.json"
    tracker = tracker.absolute()
    if tracker.resolve().is_relative_to(output.resolve()):
        raise InvalidParameter("Keep the tracker outside the download directory to avoid uploading state files")
    tracker.parent.mkdir(parents=True, exist_ok=True)
    task_lock = FileLock(str(tracker) + ".lock")
    try:
        task_lock.acquire(timeout=1)
    except Timeout as exc:
        raise InvalidParameter(
            "Another downloader is using this tracker; use separate outputs for parallel tasks"
        ) from exc
    try:
        state = _DownloadState(tracker, identity)
        try:
            return _run(
                manager,
                state,
                output,
                repo_id,
                repo_type,
                revision,
                allow_patterns,
                ignore_patterns,
                max_workers,
                file_list,
                retry_failed,
                retry_policy or DownloadRetryPolicy(),
                user_agent,
                progress_callbacks,
                cache_dir,
            )
        finally:
            state.close()
    finally:
        task_lock.release()


def _run(
    manager: DownloadManager,
    state: _DownloadState,
    output: Path,
    repo_id: str,
    repo_type: str,
    revision: str,
    include: list[str] | None,
    exclude: list[str] | None,
    workers: int,
    file_list: Iterable[dict] | None,
    retry_failed: bool,
    policy: DownloadRetryPolicy,
    user_agent: dict | str | None,
    callbacks: list[type[ProgressCallback]] | None,
    cache_dir: Path | None,
) -> DownloadReport:
    run = uuid.uuid4().hex
    report = DownloadReport(repo_id=repo_id, revision=revision, output_dir=str(output), tracker_path=str(state.path))
    stop = threading.Event()
    incoming: queue.Queue[tuple[dict | None, str | None]] = queue.Queue(maxsize=workers * 4)
    producer_done = threading.Event()
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="download-report")
    futures: dict[Future[None], str] = {}

    def put(entry: dict | None, error: str | None = None) -> None:
        while not stop.is_set():
            try:
                incoming.put((entry, error), timeout=0.1)
                return
            except queue.Full:
                pass

    def produce() -> None:
        try:
            if file_list is not None:
                if retry_failed:
                    for root in _listing_roots(include):
                        state.failed_pages(root or "/", reset=True)
                for entry in file_list:
                    if stop.is_set():
                        break
                    put(entry)
            else:
                with _DownloadControl(total_timeout=REPO_TREE_LIST_TIMEOUT_SECONDS, cancel_event=stop) as control:
                    with manager._client.read_context(control):
                        for root in _listing_roots(include):
                            if repo_type == "dataset":
                                for page, entries, error in manager._client.iter_dataset_file_pages(
                                    repo_id,
                                    revision,
                                    root or "/",
                                    isolated_pages=state.failed_pages(root or "/", reset=retry_failed),
                                ):
                                    if error is not None:
                                        state.fail_page(root or "/", page, error)
                                        put(None, f"Root={root or '/'} page={page}: {safe_error(error)}")
                                    for entry in entries:
                                        put(entry)
                                    control.check()
                            else:
                                for entry in manager._client.list_repo_files(repo_id, repo_type, revision, root=root):
                                    put(entry)
                                if getattr(manager._client._read_state, "listing_incomplete", False):
                                    put(None, f"Root={root or '/'}: model file listing is truncated")
        except _DownloadCancelledError:
            pass
        except Exception as exc:
            if isinstance(exc, AuthenticationError):
                report.global_error = safe_error(exc)
            put(None, f"Enumeration stopped: {safe_error(exc)}")
        finally:
            producer_done.set()

    def transfer(result: DownloadFileResult) -> None:
        try:
            # State hints never override a file that passes current metadata checks.
            _validate_entry({"Path": result.path, "Size": result.size, "Sha256": result.sha256}, output)
            if result.status in ("FAILED", "DEFERRED") and not retry_failed:
                target = output / result.path
                with _DownloadControl(total_timeout=DOWNLOAD_TOTAL_TIMEOUT, cancel_event=stop) as control:
                    valid = target.is_file() and manager._cache_hit(
                        target,
                        result.sha256,
                        result.size,
                        control=control,
                    )
                if valid:
                    state.finish(result.path, "VERIFIED")
                    return
                with state.lock:
                    report.skipped_failures += 1
                    skipped = report.skipped_failures
                if skipped <= 10 or skipped % 1000 == 0:
                    logger.warning(
                        "Skipping isolated file %s (%d skipped); use retry_failed=True after investigation",
                        result.path,
                        skipped,
                    )
                state.finish(result.path, result.status)
                return
            previous = state.begin(result.path, reset=retry_failed and result.status in ("FAILED", "DEFERRED"))
            recovery = _RecoveryBudget(
                policy,
                previous.retries,
                previous.recovery_seconds,
                lambda count, seconds: state.budget(result.path, count, seconds),
            )
            manager.download_file(
                repo_id,
                repo_type,
                result.path,
                revision=revision,
                local_dir=output,
                cache_dir=cache_dir,
                file_size=result.size,
                expected_sha256=result.sha256,
                user_agent=user_agent,
                progress_callbacks=callbacks,
                _cancel_event=stop,
                _recovery=recovery,
            )
        except _DownloadCancelledError:
            state.finish(result.path, result.status if result.status in ("FAILED", "DEFERRED") else "PENDING")
            raise
        except Exception as exc:
            origin = exc.__cause__ if isinstance(exc, _DownloadDeferredError) and exc.__cause__ else exc
            terminal = isinstance(origin, (FileIntegrityError, InvalidParameter)) or not getattr(
                origin, "retryable", False
            )
            state.finish(result.path, "FAILED" if terminal else "DEFERRED", exc)
            if (
                isinstance(exc, AuthenticationError)
                or isinstance(exc, OSError)
                and exc.errno in (errno.ENOSPC, errno.EROFS)
            ):
                raise
            logger.error("Isolated download failure %s; continuing remaining files: %s", result.path, safe_error(exc))
        else:
            state.finish(result.path, "VERIFIED")

    producer = threading.Thread(target=produce, name="download-enumeration", daemon=True)
    producer.start()
    interrupted = False
    entry_count = 0
    try:
        while futures or not producer_done.is_set() or not incoming.empty():
            while len(futures) < workers * 2:
                try:
                    entry, listing_error = incoming.get(timeout=0.05 if not futures else 0)
                except queue.Empty:
                    break
                if listing_error is not None:
                    report.listing_complete = False
                    if len(report.listing_errors) < 100:
                        report.listing_errors.append(listing_error)
                    logger.error("Incomplete listing; continuing known files: %s", listing_error)
                    continue
                entry_count += 1
                try:
                    if not isinstance(entry, dict):
                        raise InvalidParameter("File entry must be a dictionary")
                    if (entry.get("Type") or entry.get("type")) == "tree":
                        continue
                    raw_path = entry.get("Path") or entry.get("path") or entry.get("Name")
                    if isinstance(raw_path, str):
                        if (
                            include
                            and not _matches_patterns(raw_path, include)
                            or exclude
                            and _matches_patterns(raw_path, exclude)
                        ):
                            continue
                    path, size, sha256 = _validate_entry(entry, output)
                    result = state.select(run, path, size, sha256)
                except (InvalidParameter, ValueError, TypeError, OverflowError) as exc:
                    path = (
                        str(entry.get("Path", f"<invalid-entry-{entry_count}>"))
                        if isinstance(entry, dict)
                        else f"<invalid-entry-{entry_count}>"
                    )
                    if state.select(run, path, None, None) is not None:
                        state.finish(path, "FAILED", exc)
                    logger.error("Invalid file metadata; continuing: %s", safe_error(exc))
                    continue
                if result is None:
                    continue
                future = pool.submit(transfer, result)
                futures[future] = path
            if futures:
                completed, _ = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in completed:
                    futures.pop(future)
                    future.result()
    except KeyboardInterrupt:
        interrupted = True
        report.status = "CANCELLED"
        stop.set()
    except Exception as exc:
        report.global_error = safe_error(exc)
        report.listing_complete = False
        stop.set()
    finally:
        if stop.is_set():
            for future, path in futures.items():
                if future.cancel():
                    state.finish(path, "PENDING")
        pool.shutdown(wait=True, cancel_futures=True)
        stop.set()
        producer.join()
    state.report(run, report)
    logger.info(
        "Download %s: verified=%d failed=%d deferred=%d pending=%d; report=%s",
        report.status,
        report.succeeded,
        report.failed,
        report.deferred,
        report.pending,
        report.report_path,
    )
    if interrupted:
        raise KeyboardInterrupt
    return report
