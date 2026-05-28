"""Cache management utilities.

Provides scanning and cleanup of the local blob/snapshot cache produced
by :class:`~._download.DownloadManager`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .config import get_default_config
from .constants import RepoType
from .errors import CacheError
from .types import CacheInfo, CachedRepoInfo
from .utils.logger import get_logger

logger = get_logger("cache")

# Repo types to scan by default
_DEFAULT_SCAN_TYPES = [RepoType.MODEL, RepoType.DATASET, RepoType.STUDIO, RepoType.MCP]


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
    config = get_default_config()
    root = Path(cache_dir) if cache_dir else config.cache_dir

    if not root.is_dir():
        return CacheInfo(repos=[], total_size=0, cache_dir=str(root))

    repos: list[CachedRepoInfo] = []
    total_size = 0

    for repo_type in _DEFAULT_SCAN_TYPES:
        segment = f"{repo_type}s"
        type_dir = root / segment
        if not type_dir.is_dir():
            continue

        for repo_dir in type_dir.iterdir():
            if not repo_dir.is_dir():
                continue

            # Compute size
            size = _dir_size(repo_dir)
            total_size += size

            # Count files
            nb_files = sum(1 for _ in repo_dir.rglob("*") if _.is_file())

            # Determine revision from snapshot dirs
            snapshots_dir = repo_dir / "snapshots"
            revision = None
            if snapshots_dir.is_dir():
                revisions = [d.name for d in snapshots_dir.iterdir() if d.is_dir()]
                revision = revisions[0] if len(revisions) == 1 else ",".join(revisions[:5])

            # Last access time
            try:
                last_accessed_ts = repo_dir.stat().st_atime
            except OSError:
                last_accessed_ts = 0

            # Decode repo_id from directory name (owner--name → owner/name)
            repo_id = repo_dir.name.replace("--", "/")

            repos.append(CachedRepoInfo(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                size_on_disk=size,
                nb_files=nb_files,
                last_accessed=last_accessed_ts if last_accessed_ts > 0 else None,
                local_path=str(repo_dir),
            ))

    return CacheInfo(
        repos=repos,
        total_size=total_size,
        cache_dir=str(root),
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
    config = get_default_config()
    root = Path(cache_dir) if cache_dir else config.cache_dir

    if not root.is_dir():
        logger.info("Cache directory does not exist: %s", root)
        return 0

    freed = 0

    if repo_id and repo_type:
        # Clear specific repo
        segment = f"{repo_type}s" if not repo_type.endswith("s") else repo_type
        safe_id = repo_id.replace("/", "--")
        target = root / segment / safe_id
        if target.is_dir():
            freed = _dir_size(target)
            _safe_rmtree(target)
            logger.info("Cleared cache for %s/%s (%d bytes)", repo_type, repo_id, freed)
    elif repo_type:
        # Clear all repos of this type
        segment = f"{repo_type}s" if not repo_type.endswith("s") else repo_type
        type_dir = root / segment
        if type_dir.is_dir():
            freed = _dir_size(type_dir)
            _safe_rmtree(type_dir)
            logger.info("Cleared all %s caches (%d bytes)", repo_type, freed)
    else:
        # Clear everything
        for repo_t in _DEFAULT_SCAN_TYPES:
            segment = f"{repo_t}s"
            type_dir = root / segment
            if type_dir.is_dir():
                freed += _dir_size(type_dir)
                _safe_rmtree(type_dir)
        logger.info("Cleared all caches (%d bytes)", freed)

    return freed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dir_size(path: Path) -> int:
    """Compute total size of all files under ``path`` recursively."""
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _safe_rmtree(path: Path) -> None:
    """Remove a directory tree, raising CacheError on failure."""
    try:
        shutil.rmtree(path)
    except OSError as exc:
        raise CacheError(f"Failed to remove {path}: {exc}") from exc
