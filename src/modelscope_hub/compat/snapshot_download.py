"""Legacy-compatible snapshot download wrappers.

These functions replicate the signature of the old
``modelscope.hub.snapshot_download`` module so that existing user code
and the old SDK can delegate to ``modelscope_hub`` without changes.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

from ..api import HubApi
from ..constants import RepoType
from .constants import DEFAULT_DATASET_REVISION


def snapshot_download(
    model_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_dir: str | None = None,
    allow_file_pattern: Sequence[str] | str | None = None,
    ignore_file_pattern: Sequence[str] | str | None = None,
    max_workers: int = 4,
    cookies: dict | None = None,
    token: str | None = None,
    endpoint: str | None = None,
) -> str:
    """Download a model snapshot (legacy signature).

    Parameters mirror the old ``modelscope.hub.snapshot_download.snapshot_download``.
    """
    if cookies is not None:
        warnings.warn(
            "The 'cookies' parameter is deprecated and ignored. Use 'token' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    api = HubApi(token=token, endpoint=endpoint)
    result = api.download_repo(
        model_id,
        repo_type=RepoType.MODEL,
        revision=revision,
        cache_dir=cache_dir,
        local_dir=local_dir,
        allow_patterns=_normalize_pattern(allow_file_pattern),
        ignore_patterns=_normalize_pattern(ignore_file_pattern),
        max_workers=max_workers,
    )
    return str(result)


def dataset_snapshot_download(
    dataset_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    local_dir: str | None = None,
    allow_file_pattern: Sequence[str] | str | None = None,
    ignore_file_pattern: Sequence[str] | str | None = None,
    max_workers: int = 4,
    cookies: dict | None = None,
    token: str | None = None,
    endpoint: str | None = None,
) -> str:
    """Download a dataset snapshot (legacy signature)."""
    if cookies is not None:
        warnings.warn(
            "The 'cookies' parameter is deprecated and ignored. Use 'token' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    api = HubApi(token=token, endpoint=endpoint)
    result = api.download_repo(
        dataset_id,
        repo_type=RepoType.DATASET,
        revision=revision or DEFAULT_DATASET_REVISION,
        cache_dir=cache_dir,
        local_dir=local_dir,
        allow_patterns=_normalize_pattern(allow_file_pattern),
        ignore_patterns=_normalize_pattern(ignore_file_pattern),
        max_workers=max_workers,
    )
    return str(result)


def _normalize_pattern(pattern: Sequence[str] | str | None) -> list[str] | None:
    if pattern is None:
        return None
    if isinstance(pattern, str):
        return [pattern]
    return list(pattern) or None
