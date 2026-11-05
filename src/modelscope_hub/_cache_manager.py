"""Cache management utilities.

Provides scanning and cleanup of the local blob/snapshot cache produced
by :class:`~._download.DownloadManager`.
"""

from __future__ import annotations

from pathlib import Path

from ._cache_clear import clear_cache as _clear_cache
from ._cache_clear import directory_size, plan_cache_clear
from ._cache_layout import (
    contained_path,
    discover_cache,
    measure_payload,
    payload_files,
    resolve_snapshot,
    validate_relative,
)
from .config import get_cache_root
from .types import CachedRepoInfo, CacheInfo, CacheVerification, VerificationMismatch
from .utils.file_utils import compute_hash
from .utils.logger import get_logger

logger = get_logger("cache")


def scan_cache(cache_dir: Path | None = None) -> CacheInfo:
    """Scan the local cache and return metadata about cached repositories.

    Parameters
    ----------
    cache_dir:
        Override for the cache directory. Defaults to the SDK config default.

    Returns
    -------
    CacheInfo
        Summary of all cached repositories, total size, etc.
    """
    root = get_cache_root(cache_dir=cache_dir)
    diagnostics: list[str] = []
    repos: list[CachedRepoInfo] = []
    for location in discover_cache(root, diagnostics=diagnostics):
        try:
            size, count = measure_payload(location.path)
            repos.append(
                CachedRepoInfo(
                    repo_id=location.repo_id or "unknown",
                    repo_type=location.repo_type,
                    revision=",".join(location.revisions) or None,
                    size_on_disk=size,
                    nb_files=count,
                    last_accessed=location.path.stat().st_atime,
                    local_path=str(location.path),
                )
            )
        except OSError as exc:
            diagnostics.append(str(exc))
    return CacheInfo(
        repos=repos, total_size=sum(r.size_on_disk for r in repos), cache_dir=str(root), warnings=diagnostics
    )


def clear_cache(
    cache_dir: Path | None = None,
    repo_type: str | None = None,
    repo_id: str | None = None,
) -> int:
    """Remove cached data from disk.

    Parameters
    ----------
    cache_dir:
        Override for the cache directory. Defaults to the SDK config default.
    repo_type:
        If given, only clear caches of this repo type.
    repo_id:
        If given, only clear the cache for this specific repository.
        Must be used with ``repo_type``.

    Returns
    -------
    int
        Number of bytes freed.

    Raises
    ------
    CacheError
        On filesystem errors.
    """
    return _clear_cache(cache_dir, repo_type, repo_id)


def verify_cache(
    repo_id: str,
    repo_type: str,
    expected_by_path: dict[str, str | None],
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
) -> CacheVerification:
    """Compare a cached snapshot or local directory with remote SHA-256 values."""
    root, resolved_revision = _resolve_verification_root(
        repo_id,
        repo_type,
        revision=revision,
        cache_dir=cache_dir,
        local_dir=local_dir,
    )
    for remote_path in expected_by_path:
        validate_relative(remote_path, "file_path")
    local_by_path = {path.relative_to(root).as_posix(): contained_path(root, path) for path in payload_files(root)}
    remote_paths = set(expected_by_path)
    local_paths = set(local_by_path)
    mismatches: list[VerificationMismatch] = []
    unverified: list[str] = []
    checked = 0

    for rel_path in sorted(remote_paths & local_paths):
        expected = expected_by_path[rel_path]
        if not expected:
            unverified.append(rel_path)
            continue
        checked += 1
        actual = compute_hash(local_by_path[rel_path], "sha256")
        if actual.lower() != expected.lower():
            mismatches.append(VerificationMismatch(rel_path, expected.lower(), actual.lower()))

    return CacheVerification(
        revision=resolved_revision,
        verified_path=str(root),
        checked_count=checked,
        mismatches=mismatches,
        missing_paths=sorted(remote_paths - local_paths),
        extra_paths=sorted(local_paths - remote_paths),
        unverified_paths=unverified,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_verification_root(
    repo_id: str,
    repo_type: str,
    *,
    revision: str | None,
    cache_dir: str | Path | None,
    local_dir: str | Path | None,
) -> tuple[Path, str]:
    root, resolved = resolve_snapshot(
        repo_id, repo_type, revision=revision, cache_dir=cache_dir, local_dir=local_dir, require_payload=False
    )
    return root, resolved or revision or "master"


def _resolve_cache_targets(root: Path, repo_id: str, repo_type: str) -> list[Path]:
    """兼容旧私有入口；仅返回已经通过整批安全校验的精确目标。"""
    return [Path(target.path) for target in plan_cache_clear(root, repo_type, repo_id).targets]


def _dir_size(path: Path) -> int:
    """Compute total size of all files under ``path`` recursively."""
    return directory_size(path)
