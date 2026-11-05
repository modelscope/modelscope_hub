"""Legacy-compatible single-file download wrappers.

These functions replicate the signature of the old
``modelscope.hub.file_download`` module.
"""

from __future__ import annotations

import warnings
from urllib.parse import quote_plus

import requests as _requests

from .._cache_layout import resolve_cached_file, validate_relative, validate_repo_id
from .._download_url import http_get_file as http_get_file
from .._download_url import http_get_model_file as http_get_model_file
from ..api import HubApi
from ..config import get_cache_root
from ..constants import RepoType
from ..errors import AuthenticationError, NotExistError, PermissionDeniedError
from ..utils import append_user_agent


def _resolve_legacy_paths(
    repo_id: str,
    cache_dir: str | None,
    local_dir: str | None,
    api: HubApi,
) -> tuple[str | None, str | None]:
    """Resolve cache_dir/local_dir for legacy API compatibility.

    Passes parameters through so the new API uses its standard cache
    layout (``{cache}/{type}s/{owner}--{name}/snapshots/{rev}/...``),
    consistent with CLI ``ms download`` behavior.

    When ``local_dir`` is explicitly set, files go directly there.
    Otherwise the standard cache hierarchy is used.
    """
    return cache_dir, local_dir


def model_file_download(
    model_id: str,
    file_path: str,
    revision: str | None = None,
    *,
    cache_dir: str | None = None,
    local_dir: str | None = None,
    cookies: dict | None = None,
    token: str | None = None,
    endpoint: str | None = None,
    local_files_only: bool = False,
    user_agent: dict | str | None = None,
    progress_callbacks: list[type] | None = None,
    reuse_legacy_cache: bool = True,
    release_timestamp: int | None = None,
) -> str:
    """Download a single model file (legacy signature)."""
    validate_repo_id(model_id)
    validate_relative(file_path, "file_path")
    get_cache_root(cache_dir=cache_dir)
    if local_files_only:
        return str(
            resolve_cached_file(
                model_id,
                file_path,
                "model",
                revision=revision,
                cache_dir=cache_dir,
                local_dir=local_dir,
                reuse_legacy_cache=reuse_legacy_cache,
            )
        )
    if cookies is not None:
        warnings.warn(
            "The 'cookies' parameter is deprecated and ignored. Use 'token' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    api = HubApi(token=token, endpoint=endpoint, user_agent=append_user_agent("invoked_by/compat", user_agent))
    if endpoint is None and not local_files_only:
        try:
            endpoint = api.resolve_endpoint_for_read(model_id, repo_type="model")
            api = HubApi(token=token, endpoint=endpoint, user_agent=append_user_agent("invoked_by/compat", user_agent))
        except Exception:
            pass
    if revision is None and release_timestamp is not None:
        from .hub_api import LegacyHubApi

        revision_api = LegacyHubApi.from_api(api)
        try:
            revision = revision_api.get_valid_revision_detail(
                model_id,
                release_timestamp=release_timestamp,
            ).get("Revision")
        except (NotExistError, AuthenticationError, PermissionDeniedError) as exc:
            raise _requests.exceptions.HTTPError(str(exc), response=getattr(exc, "response", None)) from exc
        except Exception:
            # 保留在线旧版本选择在服务不可用时回退 master 的兼容行为。
            revision = None
    effective_cache, effective_local = _resolve_legacy_paths(
        model_id,
        cache_dir,
        local_dir,
        api,
    )
    try:
        result = api.download_file(
            model_id,
            repo_type=RepoType.MODEL,
            file_path=file_path,
            revision=revision,
            cache_dir=effective_cache,
            local_dir=effective_local,
            local_files_only=local_files_only,
            user_agent=user_agent,
            progress_callbacks=progress_callbacks,
            reuse_legacy_cache=reuse_legacy_cache,
        )
    except (NotExistError, AuthenticationError, PermissionDeniedError) as e:
        raise _requests.exceptions.HTTPError(str(e), response=getattr(e, "response", None)) from e
    return str(result)


def get_file_download_url(model_id: str, file_path: str, revision: str, endpoint: str | None = None) -> str:
    """构造旧下载 URL；不触发下载或认证初始化。"""
    import os

    from ..constants import DEFAULT_ENDPOINT

    endpoint = endpoint or os.environ.get("MODELSCOPE_ENDPOINT") or DEFAULT_ENDPOINT
    return (
        f"{endpoint.rstrip('/')}/api/v1/models/{model_id}/repo?"
        f"Revision={quote_plus(revision)}&FilePath={quote_plus(file_path)}"
    )


def dataset_file_download(
    dataset_id: str,
    file_path: str,
    *,
    cache_dir: str | None = None,
    local_dir: str | None = None,
    revision: str | None = None,
    cookies: dict | None = None,
    token: str | None = None,
    endpoint: str | None = None,
    local_files_only: bool = False,
    user_agent: dict | str | None = None,
    progress_callbacks: list[type] | None = None,
    reuse_legacy_cache: bool = True,
) -> str:
    """Download a single dataset file (legacy signature)."""
    validate_repo_id(dataset_id)
    validate_relative(file_path, "file_path")
    get_cache_root(cache_dir=cache_dir)
    if local_files_only:
        return str(
            resolve_cached_file(
                dataset_id,
                file_path,
                "dataset",
                revision=revision,
                cache_dir=cache_dir,
                local_dir=local_dir,
                reuse_legacy_cache=reuse_legacy_cache,
            )
        )
    if cookies is not None:
        warnings.warn(
            "The 'cookies' parameter is deprecated and ignored. Use 'token' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

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
    try:
        result = api.download_file(
            dataset_id,
            repo_type=RepoType.DATASET,
            file_path=file_path,
            revision=revision,
            cache_dir=effective_cache,
            local_dir=effective_local,
            local_files_only=local_files_only,
            user_agent=user_agent,
            progress_callbacks=progress_callbacks,
            reuse_legacy_cache=reuse_legacy_cache,
        )
    except (NotExistError, AuthenticationError, PermissionDeniedError) as e:
        raise _requests.exceptions.HTTPError(str(e), response=getattr(e, "response", None)) from e
    return str(result)
