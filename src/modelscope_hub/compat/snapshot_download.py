"""Legacy-compatible snapshot download wrappers.

These functions replicate the signature of the old
``modelscope.hub.snapshot_download`` module so that existing user code
and the old SDK can delegate to ``modelscope_hub`` without changes.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import requests as _requests

from .._cache_manager import _link_legacy_model_path, resolve_cached_snapshot
from ..api import HubApi
from ..config import get_cache_root
from ..constants import RepoType
from ..errors import AuthenticationError, InvalidParameter, NotExistError, PermissionDeniedError
from ..types import DownloadRetryPolicy
from ..utils import append_user_agent
from ..utils.patterns import normalize_patterns
from .file_download import _resolve_legacy_paths

if TYPE_CHECKING:
    from .._download import ProgressCallback


def snapshot_download(
    model_id: str | None = None,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_dir: str | None = None,
    allow_file_pattern: Sequence[str] | str | None = None,
    ignore_file_pattern: Sequence[str] | str | None = None,
    allow_patterns: Sequence[str] | str | None = None,
    ignore_patterns: Sequence[str] | str | None = None,
    max_workers: int = 4,
    cookies: dict | None = None,
    repo_id: str | None = None,
    repo_type: str | None = None,
    token: str | None = None,
    endpoint: str | None = None,
    local_files_only: bool = False,
    user_agent: dict | str | None = None,
    progress_callbacks: list[type[ProgressCallback]] | None = None,
    continue_on_error: bool = False,
    tracker_path: str | None = None,
    retry_failed: bool = False,
    retry_policy: DownloadRetryPolicy | None = None,
) -> str:
    """Download a repo snapshot (legacy signature).

    Parameters mirror the old ``modelscope.hub.snapshot_download.snapshot_download``.
    ``allow_patterns``/``ignore_patterns`` take priority over the
    ``allow_file_pattern``/``ignore_file_pattern`` aliases when both are set.
    ``progress_callbacks`` takes a list of :class:`ProgressCallback`
    subclasses (not instances); each is instantiated per file to report
    download progress. continue_on_error opts into durable failed-file isolation
    and returns the output path even when the persisted report is PARTIAL. Use
    HubApi.download_repo_report for direct access to verified-file results.
    """
    effective_id = repo_id or model_id
    if not effective_id:
        raise InvalidParameter("Please provide a valid model_id or repo_id")
    effective_type = repo_type or "model"

    if cookies is not None:
        warnings.warn(
            "The 'cookies' parameter is deprecated and ignored. Use 'token' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    include = _normalize_pattern(allow_patterns) or _normalize_pattern(allow_file_pattern)
    exclude = _normalize_pattern(ignore_patterns) or _normalize_pattern(ignore_file_pattern)

    if local_files_only:
        if continue_on_error or tracker_path is not None or retry_failed or retry_policy is not None:
            raise InvalidParameter("Report/recovery options cannot use local_files_only")
        return _offline_snapshot(effective_id, effective_type, revision, cache_dir, local_dir)

    api = HubApi(token=token, endpoint=endpoint, user_agent=append_user_agent("invoked_by/compat", user_agent))
    if endpoint is None and not local_files_only:
        try:
            endpoint = api.resolve_endpoint_for_read(
                effective_id,
                repo_type=effective_type,
            )
            api = HubApi(token=token, endpoint=endpoint, user_agent=append_user_agent("invoked_by/compat", user_agent))
        except Exception:
            pass
    effective_cache, effective_local = _resolve_legacy_paths(
        effective_id,
        cache_dir,
        local_dir,
        api,
    )
    if continue_on_error:
        if local_files_only:
            raise InvalidParameter("Report-mode downloads cannot use local_files_only")
        report = api.download_repo_report(
            effective_id,
            effective_type,
            revision=revision,
            cache_dir=effective_cache,
            local_dir=effective_local,
            allow_patterns=include,
            ignore_patterns=exclude,
            max_workers=max_workers,
            tracker_path=tracker_path,
            retry_failed=retry_failed,
            retry_policy=retry_policy,
            user_agent=user_agent,
            progress_callbacks=progress_callbacks,
        )
        return report.output_dir
    if tracker_path is not None or retry_failed or retry_policy is not None:
        raise InvalidParameter("Tracker/recovery options require continue_on_error=True")
    try:
        result = api.download_repo(
            effective_id,
            repo_type=effective_type,
            revision=revision,
            cache_dir=effective_cache,
            local_dir=effective_local,
            allow_patterns=include,
            ignore_patterns=exclude,
            max_workers=max_workers,
            local_files_only=local_files_only,
            user_agent=user_agent,
            progress_callbacks=progress_callbacks,
        )
    except (NotExistError, AuthenticationError, PermissionDeniedError) as e:
        raise _requests.exceptions.HTTPError(str(e), response=getattr(e, "response", None)) from e
    if effective_type == RepoType.MODEL and effective_local is None:
        # Keep legacy cache-root paths valid for callers that predate the snapshot layout.
        _link_legacy_model_path(effective_id, Path(result), effective_cache)
    return str(result)


def dataset_snapshot_download(
    dataset_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_dir: str | None = None,
    allow_file_pattern: Sequence[str] | str | None = None,
    ignore_file_pattern: Sequence[str] | str | None = None,
    allow_patterns: Sequence[str] | str | None = None,
    ignore_patterns: Sequence[str] | str | None = None,
    max_workers: int = 4,
    cookies: dict | None = None,
    token: str | None = None,
    endpoint: str | None = None,
    local_files_only: bool = False,
    user_agent: dict | str | None = None,
    continue_on_error: bool = False,
    tracker_path: str | None = None,
    retry_failed: bool = False,
    retry_policy: DownloadRetryPolicy | None = None,
) -> str:
    """Download a dataset snapshot; report-mode options match snapshot_download."""
    if cookies is not None:
        warnings.warn(
            "The 'cookies' parameter is deprecated and ignored. Use 'token' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    include = _normalize_pattern(allow_patterns) or _normalize_pattern(allow_file_pattern)
    exclude = _normalize_pattern(ignore_patterns) or _normalize_pattern(ignore_file_pattern)

    if local_files_only:
        if continue_on_error or tracker_path is not None or retry_failed or retry_policy is not None:
            raise InvalidParameter("Report/recovery options cannot use local_files_only")
        return _offline_snapshot(dataset_id, "dataset", revision, cache_dir, local_dir)

    api = HubApi(token=token, endpoint=endpoint, user_agent=append_user_agent("invoked_by/compat", user_agent))
    if endpoint is None and not local_files_only:
        try:
            endpoint = api.resolve_endpoint_for_read(dataset_id, repo_type="dataset")
            api = HubApi(token=token, endpoint=endpoint, user_agent=append_user_agent("invoked_by/compat", user_agent))
        except Exception:
            pass
    effective_cache, effective_local = _resolve_legacy_paths(
        dataset_id,
        cache_dir,
        local_dir,
        api,
    )
    if continue_on_error:
        if local_files_only:
            raise InvalidParameter("Report-mode downloads cannot use local_files_only")
        report = api.download_repo_report(
            dataset_id,
            "dataset",
            revision=revision,
            cache_dir=effective_cache,
            local_dir=effective_local,
            allow_patterns=include,
            ignore_patterns=exclude,
            max_workers=max_workers,
            tracker_path=tracker_path,
            retry_failed=retry_failed,
            retry_policy=retry_policy,
            user_agent=user_agent,
        )
        return report.output_dir
    if tracker_path is not None or retry_failed or retry_policy is not None:
        raise InvalidParameter("Tracker/recovery options require continue_on_error=True")
    try:
        result = api.download_repo(
            dataset_id,
            repo_type=RepoType.DATASET,
            revision=revision,
            cache_dir=effective_cache,
            local_dir=effective_local,
            allow_patterns=include,
            ignore_patterns=exclude,
            max_workers=max_workers,
            local_files_only=local_files_only,
            user_agent=user_agent,
        )
    except (NotExistError, AuthenticationError, PermissionDeniedError) as e:
        raise _requests.exceptions.HTTPError(str(e), response=getattr(e, "response", None)) from e
    return str(result)


def _offline_snapshot(
    repo_id: str,
    repo_type: str,
    revision: str | None,
    cache_dir: str | None,
    local_dir: str | None,
) -> str:
    """Resolve cached snapshots without constructing transport or credential state."""
    revision = revision or None
    if local_dir is None:
        return str(resolve_cached_snapshot(repo_id, repo_type, revision=revision, cache_dir=cache_dir))
    from .._download import DownloadManager

    return str(
        DownloadManager._resolve_cached_snapshot_at(
            repo_id,
            repo_type,
            revision=revision,
            cache_dir=get_cache_root(cache_dir),
            local_dir=Path(local_dir),
        )
    )


def _normalize_pattern(pattern: Sequence[str] | str | None) -> list[str] | None:
    return normalize_patterns(pattern)
