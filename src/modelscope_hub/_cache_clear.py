"""Clear cached repositories using exact deletion plans and filesystem boundaries."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from ._cache_layout import (
    CacheLocation,
    check_identity,
    contained_path,
    discover_cache,
    has_legacy_metadata,
    has_payload,
    legacy_candidates,
    normalize_type,
    read_identity,
    repo_cache_path,
    temp_candidates,
    validate_repo_id,
)
from .config import get_cache_root
from .errors import CacheError, InvalidParameter
from .types import CacheClearPlan, CacheClearTarget


def _fingerprint(path: Path) -> tuple[int, int, int]:
    info = path.lstat()
    return info.st_dev, info.st_ino, info.st_mode


def directory_size(path: Path) -> int:
    total = 0
    device = path.stat().st_dev
    for base, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(base) / d).is_symlink()]
        for directory in dirs:
            if (Path(base) / directory).stat().st_dev != device:
                raise CacheError(f"Refusing to traverse a mounted filesystem: {Path(base) / directory}")
        for name in files:
            try:
                total += (Path(base) / name).lstat().st_size
            except OSError as exc:
                raise CacheError(f"Cannot inspect cache: {path}") from exc
    return total


def _validate_target(root: Path, path: Path) -> None:
    contained_path(root, path)
    if not path.absolute().is_relative_to(root) or path.absolute() == root:
        raise CacheError(f"Cache target is not beneath its authorized root: {path}")
    for parent in path.parents:
        if parent == root:
            break
        if parent.is_symlink():
            raise CacheError(f"Cache deletion refuses a symlink ancestor: {parent}")
    if not path.is_symlink() and not path.is_dir():
        raise CacheError(f"Cache target is not a repository directory: {path}")


def _exact_locations(root: Path, repo_id: str, kind: str) -> list[CacheLocation]:
    locations = []
    standard = repo_cache_path(repo_id, kind, root)
    candidates = [(standard, "standard")]
    candidates.extend(legacy_candidates(repo_id, kind, root))
    candidates.extend((path, "temporary") for path in temp_candidates(repo_id, kind, root))
    for path, layout in candidates:
        # Validate links before existence checks; never silently skip dangling or external links.
        contained_path(root, path)
        if not path.exists() and not path.is_symlink():
            continue
        _validate_target(root, path)
        if layout == "standard" and not ((path / "snapshots").is_dir() or (path / ".ms_repo.json").is_file()):
            continue
        if layout in ("typed", "flat") and not has_legacy_metadata(path):
            continue
        data = read_identity(path)
        check_identity(path, repo_id, kind)
        if layout == "flat" and (not data or data.get("repo_type") is None):
            raise CacheError(f"Cannot prove repository type for flat cache: {path}")
        if layout == "standard" and repo_id.replace("/", "--").count("--") != 1 and not data:
            raise CacheError(f"Ambiguous encoded repository identity: {path}")
        locations.append(CacheLocation(path, repo_id, kind, "alias" if path.is_symlink() else layout))
    return locations


def plan_cache_clear(
    cache_dir: str | Path | None = None, repo_type: str | None = None, repo_id: str | None = None
) -> CacheClearPlan:
    """Plan without writes; reject the entire batch on boundary or ownership conflicts."""
    if repo_id is not None:
        validate_repo_id(repo_id)
        if repo_type is None:
            raise CacheError("repo_type is required when repo_id is specified")
    kind = normalize_type(repo_type) if repo_type is not None else None
    root = get_cache_root(cache_dir=cache_dir).resolve()
    if repo_id is None and root in (Path(root.anchor), Path.home().resolve(), (Path.home() / ".cache").resolve()):
        raise CacheError(f"Refusing whole-cache deletion beneath a protected root: {root}")
    if not root.exists():
        return CacheClearPlan(str(root), repo_id, kind, ())
    if not root.is_dir():
        raise CacheError(f"Cache root is not a directory: {root}")
    diagnostics: list[str] = []
    locations = (
        _exact_locations(root, repo_id, kind)
        if repo_id is not None and kind is not None
        else discover_cache(root, repo_type=kind, diagnostics=diagnostics, strict=True, include_aliases=True)
    )
    targets = []
    seen: set[Path] = set()
    physical: set[Path] = set()
    physical_sizes: dict[Path, int] = {}
    for location in locations:
        path = location.path.absolute()
        if path in seen:
            continue
        if location.layout != "temporary" and not has_payload(path):
            continue
        if location.repo_id is None or location.repo_type is None:
            raise CacheError(f"Cannot prove cache ownership/type: {path}")
        if kind is not None and location.repo_type != kind:
            raise CacheError(f"Cache target has an unauthorized repository type: {path}")
        if repo_id is not None and location.repo_id.casefold() != repo_id.casefold():
            raise CacheError(f"Cache target has an unauthorized repository identity: {path}")
        _validate_target(root, path)
        seen.add(path)
        resolved = path.resolve()
        size = 0 if path.is_symlink() else directory_size(path)
        if not path.is_symlink():
            if resolved in physical:
                continue
            physical.add(resolved)
            physical_sizes[resolved] = size
        targets.append(
            CacheClearTarget(
                str(path),
                location.repo_id,
                location.repo_type,
                location.layout,
                size,
                _fingerprint(path),
                json.dumps(read_identity(path), sort_keys=True),
            )
        )
    for target in targets:
        path = Path(target.path)
        if path.is_symlink() and path.resolve() not in physical_sizes:
            raise CacheError(f"Cache alias target is outside the selected repository scope: {path}")
    targets.sort(key=lambda entry: (not Path(entry.path).is_symlink(), entry.path))
    return CacheClearPlan(str(root), repo_id, kind, tuple(targets), _fingerprint(root), tuple(diagnostics))


def _open_directory(path: Path) -> int:
    """Open directories from the filesystem root without following substituted ancestor symlinks."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _remove_at(parent_fd: int, name: str, expected: tuple[int, int, int]) -> None:
    """Delete under pinned directory descriptors without following symlinks (Python 3.10+)."""
    info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (info.st_dev, info.st_ino, info.st_mode) != expected:
        raise CacheError(f"Cache entry changed before deletion: {name}")
    if not stat.S_ISDIR(info.st_mode):
        os.unlink(name, dir_fd=parent_fd)
        return
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(name, flags, dir_fd=parent_fd)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_mode) != expected:
            raise CacheError(f"Cache directory changed before deletion: {name}")
        with os.scandir(fd) as entries:
            for entry in entries:
                child = entry.stat(follow_symlinks=False)
                if child.st_dev != opened.st_dev:
                    raise CacheError(f"Refusing to traverse a mounted filesystem: {entry.name}")
                _remove_at(fd, entry.name, (child.st_dev, child.st_ino, child.st_mode))
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (info.st_dev, info.st_ino, info.st_mode) != expected:
            raise CacheError(f"Cache directory changed during deletion: {name}")
        os.rmdir(name, dir_fd=parent_fd)
    finally:
        os.close(fd)


def execute_cache_clear(plan: CacheClearPlan) -> int:
    """Execute confirmed targets without expanding the scope to newly discovered caches."""
    if not isinstance(plan, CacheClearPlan):
        raise InvalidParameter("Expected a CacheClearPlan")
    current = plan_cache_clear(plan.cache_dir, plan.repo_type, plan.repo_id)
    if current != plan:
        raise CacheError("Cache changed after planning; inspect a new deletion plan before proceeding")
    root = Path(plan.cache_dir)
    if not plan.targets:
        return 0
    required = (os.open, os.stat, os.unlink, os.rmdir)
    if not all(operation in os.supports_dir_fd for operation in required) or not hasattr(os, "O_NOFOLLOW"):
        raise CacheError("This platform does not support safe descriptor-relative cache deletion")
    # Revalidate all targets and parents before the first deletion to prevent partial effects.
    for target in plan.targets:
        path = Path(target.path)
        _validate_target(root, path)
        fd = _open_directory(path.parent)
        os.close(fd)
    freed = 0
    for target in plan.targets:
        path = Path(target.path)
        try:
            _validate_target(root, path)
            if _fingerprint(path) != target.fingerprint:
                raise CacheError(f"Cache target changed after planning: {path}")
            if json.dumps(read_identity(path), sort_keys=True) != target.identity:
                raise CacheError(f"Cache identity changed after planning: {path}")
            fd = _open_directory(path.parent)
            try:
                _remove_at(fd, path.name, target.fingerprint)
            finally:
                os.close(fd)
        except (OSError, CacheError) as exc:
            raise CacheError(f"Cache clear stopped after freeing {freed} bytes; failed target: {path}") from exc
        freed += target.size
    return freed


def clear_cache(cache_dir: str | Path | None = None, repo_type: str | None = None, repo_id: str | None = None) -> int:
    """Preserve direct cleanup compatibility with whole-batch scope validation before deletion."""
    return execute_cache_clear(plan_cache_clear(cache_dir, repo_type, repo_id))
