"""Legacy-compatible single-file download wrappers.

These functions replicate the signature of the old
``modelscope.hub.file_download`` module.
"""

from __future__ import annotations

import warnings

from ..api import HubApi
from ..constants import RepoType
from .constants import DEFAULT_DATASET_REVISION


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
) -> str:
    """Download a single model file (legacy signature)."""
    if cookies is not None:
        warnings.warn(
            "The 'cookies' parameter is deprecated and ignored. Use 'token' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    api = HubApi(token=token, endpoint=endpoint)
    if endpoint is None and not local_files_only:
        try:
            endpoint = api.resolve_endpoint_for_read(model_id, repo_type="model")
            api = HubApi(token=token, endpoint=endpoint)
        except Exception:
            pass
    result = api.download_file(
        model_id,
        repo_type=RepoType.MODEL,
        file_path=file_path,
        revision=revision,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_files_only=local_files_only,
        user_agent=user_agent,
    )
    return str(result)


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
) -> str:
    """Download a single dataset file (legacy signature)."""
    if cookies is not None:
        warnings.warn(
            "The 'cookies' parameter is deprecated and ignored. Use 'token' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    api = HubApi(token=token, endpoint=endpoint)
    if endpoint is None and not local_files_only:
        try:
            endpoint = api.resolve_endpoint_for_read(dataset_id, repo_type="dataset")
            api = HubApi(token=token, endpoint=endpoint)
        except Exception:
            pass
    result = api.download_file(
        dataset_id,
        repo_type=RepoType.DATASET,
        file_path=file_path,
        revision=revision or DEFAULT_DATASET_REVISION,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_files_only=local_files_only,
        user_agent=user_agent,
    )
    return str(result)
