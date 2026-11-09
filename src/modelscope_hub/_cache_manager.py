"""Cache management utilities.

Provides scanning and cleanup of the local blob/snapshot cache produced
by :class:`~._download.DownloadManager`.
"""

from __future__ import annotations

import os
import re
import stat
import uuid
from pathlib import Path

from .config import get_cache_root, get_default_config
from .constants import DEFAULT_CACHE_DIR_NAME, ENV_CACHE, RepoType
from .errors import CacheError
from .types import CachedRepoInfo, CacheInfo, CacheVerification, VerificationMismatch
from .utils.file_utils import compute_hash
from .utils.logger import get_logger

logger = get_logger("cache")

# Repo types to scan by default
_DEFAULT_SCAN_TYPES = list(RepoType)


def _validate_cache_scope(repo_type: str | None, repo_id: str | None) -> None:
    """Validate repository identifiers without treating empty values as wildcards."""
    if repo_type is not None and repo_type not in {t.value for t in RepoType}:
        raise CacheError(f"Invalid repo_type: {repo_type!r}")
    if repo_id is not None:
        if repo_type is None:
            raise CacheError("repo_type is required when repo_id is specified")
        parts = repo_id.split("/")
        if len(parts) != 2 or any(
            not re.fullmatch(r"[\w.-]+", part, flags=re.ASCII) or part in {".", ".."} for part in parts
        ):
            raise CacheError("repo_id must be a safe owner/name identifier")


def resolve_cached_snapshot(
    repo_id: str,
    repo_type: str = "model",
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return a cached snapshot without network access or filesystem writes.

    Uses the download resolver's layout and revision selection rules. Legacy
    caches have no revision dimension and are reused regardless of revision.
    The returned directory does not guarantee a complete repository download.

    Raises
    ------
    CacheError
        If an identifier or revision is invalid.
    CacheNotFound
        If the snapshot is missing or its revision is ambiguous.
    """
    from ._download import DownloadManager

    _validate_cache_scope(repo_type, repo_id)
    if revision is not None and (
        not revision
        or revision.startswith("/")
        or "\\" in revision
        or "\0" in revision
        or any(part in {".", "..", ""} for part in revision.split("/"))
    ):
        raise CacheError("revision must be a safe relative path")
    return DownloadManager._resolve_cached_snapshot_at(
        repo_id,
        repo_type,
        revision=revision,
        cache_dir=get_cache_root(cache_dir),
    )


def _link_legacy_model_path(repo_id: str, directory: Path, cache_dir: str | Path | None = None) -> None:
    """Point ``<legacy model root>/<repo_id>`` at the *directory* a download returned, best-effort.

    Consumers written for SDKs before 1.38 append a repository ID to the legacy
    model root (``MODELSCOPE_CACHE`` or ``~/.cache/modelscope/hub``, plus
    ``models``). The link follows the returned snapshot or reused legacy cache
    only when both lie inside the active cache root. It replaces only links into
    the repository's snapshots, never real directories or links of other layouts.
    """
    from ._download import DownloadManager

    legacy_cache = os.environ.get(ENV_CACHE)
    if legacy_cache == "":
        return
    try:
        _validate_cache_scope(RepoType.MODEL, repo_id)
        root = get_cache_root(cache_dir)
        legacy_root = Path(legacy_cache or Path.home() / ".cache" / DEFAULT_CACHE_DIR_NAME / "hub")
        legacy_root = legacy_root.expanduser().resolve() / "models"
        snapshots = root / "models" / repo_id.replace("/", "--") / "snapshots"
        target = Path(directory).resolve()
        if legacy_root.parent not in (root, root / "hub") or root not in target.parents:
            return
        owner, name = repo_id.split("/")
        link = legacy_root / owner / name
        if link.parent.is_symlink():
            return
        if link.is_symlink():
            if Path(os.path.realpath(link)) == target or not DownloadManager._is_compat_link(link, snapshots):
                return
        elif link.exists():
            return
        link.parent.mkdir(parents=True, exist_ok=True)
        temp = link.with_name(f".{name}.{uuid.uuid4().hex}.link")
        try:
            os.symlink(os.path.relpath(target, link.parent.resolve()), temp, target_is_directory=True)
            os.replace(temp, link)
        finally:
            if temp.is_symlink():
                temp.unlink()
    except (CacheError, OSError, RuntimeError, ValueError) as exc:
        logger.debug("Skipped legacy cache link for %s: %s", repo_id, exc)


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
    root = get_cache_root(cache_dir)

    if not root.is_dir():
        return CacheInfo(repos=[], total_size=0, cache_dir=str(root))

    repos: list[CachedRepoInfo] = []
    total_size = 0

    for repo_type in _DEFAULT_SCAN_TYPES:
        segment = f"{repo_type}s"
        for type_dir in (root / segment, root / "hub" / segment):
            if type_dir.is_symlink() or not type_dir.is_dir() or (root / "hub").is_symlink():
                continue
            candidates: list[tuple[str, Path]] = []
            for child in sorted(type_dir.iterdir()):
                if child.is_symlink() or not child.is_dir() or child.name.startswith("."):
                    continue
                if "--" in child.name:
                    candidates.append((child.name.replace("--", "/", 1), child))
                else:
                    candidates.extend(
                        (f"{child.name}/{p.name.replace('___', '.')}", p)
                        for p in sorted(child.iterdir())
                        if p.is_dir() and not p.is_symlink() and not p.name.startswith(".")
                    )
            for repo_id, repo_dir in candidates:
                size = _dir_size(repo_dir)
                total_size += size
                nb_files = sum(1 for p in repo_dir.rglob("*") if p.is_file() and not p.is_symlink())
                snapshots_dir = repo_dir / "snapshots"
                revision = None
                if snapshots_dir.is_dir():
                    revisions = [d.name for d in sorted(snapshots_dir.iterdir()) if d.is_dir()]
                    revision = revisions[0] if len(revisions) == 1 else ",".join(revisions[:5])
                try:
                    last_accessed_ts = repo_dir.stat().st_atime
                except OSError:
                    last_accessed_ts = 0
                repos.append(
                    CachedRepoInfo(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        revision=revision,
                        size_on_disk=size,
                        nb_files=nb_files,
                        last_accessed=last_accessed_ts if last_accessed_ts > 0 else None,
                        local_path=str(repo_dir),
                    )
                )

    # Scan flat layout (compat): {root}/{owner}/{name}/
    # and legacy layout: {root}/hub/{owner}/{name}/
    _scanned_paths = {r.local_path for r in repos}  # avoid double-counting

    for prefix, prefix_path in [("", root), ("hub", root / "hub")]:
        if not prefix_path.is_dir():
            continue
        for owner_dir in prefix_path.iterdir():
            if not owner_dir.is_dir():
                continue
            # Skip known non-repo directories
            if owner_dir.name in ("hub", "models", "datasets", "studios", "mcps", "skills"):
                continue
            for name_dir in owner_dir.iterdir():
                if not name_dir.is_dir():
                    continue
                if str(name_dir) in _scanned_paths:
                    continue

                size = _dir_size(name_dir)
                total_size += size
                nb_files = sum(1 for f in name_dir.rglob("*") if f.is_file())
                repo_id = f"{owner_dir.name}/{name_dir.name}"

                repos.append(
                    CachedRepoInfo(
                        repo_id=repo_id,
                        repo_type=RepoType.MODEL,  # assume model for flat layout
                        revision=None,
                        size_on_disk=size,
                        nb_files=nb_files,
                        last_accessed=None,
                        local_path=str(name_dir),
                    )
                )

    return CacheInfo(
        repos=repos,
        total_size=total_size,
        cache_dir=str(root),
    )


def clear_cache(
    cache_dir: str | Path | None = None,
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
    root, targets = _cache_clear_targets(cache_dir, repo_type, repo_id)
    freed = 0
    for target in targets:
        size = 0 if target.is_symlink() else _dir_size(target)
        _safe_rmtree(target, root)
        freed += size
        logger.info("Cleared cache at %s (%d bytes)", target, size)
    return freed


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
    local_by_path = {
        relative.as_posix(): path
        for path in root.rglob("*")
        for relative in (path.relative_to(root),)
        if path.is_file() and not any(part.startswith(".") for part in relative.parts)
    }
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
    if local_dir is not None:
        root = Path(local_dir).expanduser().resolve()
        if not root.is_dir():
            raise CacheError(f"Local directory does not exist: {root}")
        return root, revision or "master"

    cache_root = Path(cache_dir or get_default_config().cache_dir).expanduser().resolve()
    segment = f"{repo_type}s" if not repo_type.endswith("s") else repo_type
    repo_root = cache_root / segment / repo_id.replace("/", "--")
    snapshots = repo_root / "snapshots"
    if not snapshots.is_dir():
        raise CacheError(f"Repository is not present in cache: {repo_root}")

    if revision:
        snapshot = snapshots / revision
        if not snapshot.is_dir():
            raise CacheError(f"Revision '{revision}' is not present in cache: {snapshot}")
        return snapshot, revision

    for default_name in ("master", "main"):
        default_snapshot = snapshots / default_name
        if default_snapshot.is_dir():
            return default_snapshot, default_name
    candidates = sorted(path for path in snapshots.iterdir() if path.is_dir())
    if not candidates:
        raise CacheError(f"No cached revisions found in: {snapshots}")
    if len(candidates) == 1:
        return candidates[0], candidates[0].name
    raise CacheError("Cached revision is ambiguous; pass --revision explicitly")


def _resolve_cache_targets(root: Path, repo_id: str, repo_type: str) -> list[Path]:
    """Locate typed repository directories, leaving untyped flat caches untouched."""
    _validate_cache_scope(repo_type, repo_id)
    owner, name = repo_id.split("/")
    candidates: list[Path] = []
    for base in (root / f"{repo_type}s", root / "hub" / f"{repo_type}s"):
        if base.is_symlink() or (root / "hub").is_symlink():
            raise CacheError(f"Refusing symlinked cache container: {base}")
        candidates.extend((base / repo_id.replace("/", "--"), base / owner / name.replace(".", "___")))
        if "." in name:
            candidates.append(base / owner / name)
        candidates.append(base / "._____temp" / owner / name)
    return list(dict.fromkeys(p for p in candidates if p.is_dir() or p.is_symlink()))


def _check_cache_root(root: Path) -> None:
    """Reject the home directory, filesystem root, and ancestors of home."""
    home = Path.home().resolve()
    if root == Path(root.anchor) or root == home or root in home.parents:
        raise CacheError(f"Refusing unsafe cache root: {root}")


def _check_cache_target(root: Path, target: Path) -> None:
    """Reject out-of-root targets, cache containers, and symlinked parents."""
    _check_cache_root(root)
    try:
        relative = target.relative_to(root)
        resolved = target.resolve()
    except (ValueError, OSError, RuntimeError) as exc:
        raise CacheError(f"Invalid cache target: {target}") from exc
    if not relative.parts or resolved == root or root not in resolved.parents:
        raise CacheError(f"Cache target escapes the cache root: {target}")
    if any(part in {".", ".."} for part in relative.parts):
        raise CacheError(f"Invalid cache target: {target}")
    typed = relative.parts[1:] if relative.parts[0] == "hub" else relative.parts
    segments = {f"{kind.value}s" for kind in RepoType}
    if relative.parts == ("hub",) or (
        typed[0] in segments and (len(typed) == 1 or (len(typed) == 2 and "--" not in typed[1]))
    ):
        raise CacheError(f"Refusing to remove a cache container: {target}")
    parent = root
    for component in relative.parts[:-1]:
        parent /= component
        if parent.is_symlink():
            raise CacheError(f"Refusing symlinked cache parent: {parent}")


def _cache_clear_targets(
    cache_dir: str | Path | None = None,
    repo_type: str | None = None,
    repo_id: str | None = None,
) -> tuple[Path, list[Path]]:
    """Collect and validate all repository targets before deletion starts."""
    _validate_cache_scope(repo_type, repo_id)
    if cache_dir is not None and not str(cache_dir).strip():
        raise CacheError("cache_dir must not be empty")
    root = get_cache_root(cache_dir)
    _check_cache_root(root)
    targets = []
    if repo_id is not None:
        assert repo_type is not None
        targets = _resolve_cache_targets(root, repo_id, repo_type)
    else:
        for kind in [repo_type] if repo_type is not None else [t.value for t in RepoType]:
            for base in (root / f"{kind}s", root / "hub" / f"{kind}s"):
                if base.is_symlink() or (root / "hub").is_symlink():
                    raise CacheError(f"Refusing symlinked cache container: {base}")
                if not base.is_dir():
                    continue
                for child in sorted(base.iterdir()):
                    if child.is_symlink():
                        _check_cache_target(root, child)
                    if child.name.startswith(".") or not child.is_dir():
                        continue
                    if "--" in child.name and (child / "snapshots").is_dir():
                        targets.append(child)
                    else:
                        targets.extend(p for p in sorted(child.iterdir()) if p.is_dir() and not p.name.startswith("."))
    targets = list(dict.fromkeys(targets))
    for target in targets:
        _check_cache_target(root, target)
    return root, targets


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


def _remove_tree_at(parent_fd: int, name: str) -> None:
    """Remove an entry relative to a directory descriptor without following links."""
    info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode):
        os.unlink(name, dir_fd=parent_fd)
        return
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise CacheError("Cache directory changed during cleanup")
        with os.scandir(fd) as entries:
            for entry in entries:
                _remove_tree_at(fd, entry.name)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            raise CacheError("Cache directory changed during cleanup")
        os.rmdir(name, dir_fd=parent_fd)
    finally:
        os.close(fd)


def _safe_rmtree(path: Path, root: Path) -> None:
    """Remove a validated target using no-follow directory operations.

    Raises CacheError if the platform lacks the required filesystem support.
    """
    path = path.expanduser().absolute()
    _check_cache_target(root, path)
    if os.open not in os.supports_dir_fd or os.scandir not in os.supports_fd or not hasattr(os, "O_NOFOLLOW"):
        raise CacheError("Safe cache cleanup is not supported on this platform")
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in path.relative_to(root).parts[:-1]:
            child_fd = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child_fd
        _remove_tree_at(fd, path.name)
    except OSError as exc:
        raise CacheError(f"Failed to remove {path}: {exc}") from exc
    finally:
        os.close(fd)
