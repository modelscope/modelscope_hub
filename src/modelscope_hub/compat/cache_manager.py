"""旧 SDK 文件级缓存视图；布局发现与读取由 hub 核心实现。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .._cache_clear import clear_cache, execute_cache_clear, plan_cache_clear
from .._cache_layout import (
    CacheLocation,
    contained_path,
    discover_cache,
    find_legacy_repo_dir,
    identify_cached_repo,
    payload_files,
    read_legacy_metadata,
    resolve_cached_file,
    resolve_cached_snapshot,
    validate_relative,
)
from .._cache_manager import scan_cache
from ..config import get_cache_root
from ..errors import CacheError, CacheNotFound, CorruptedCacheException
from ..utils.format import format_timesince, tabulate


def _size(size: int) -> str:
    if size == 0:
        return "0B"
    index = min(int(math.floor(math.log(size, 1024))), 5)
    return f"{round(size / 1024**index, 2)} {('B', 'KB', 'MB', 'GB', 'TB', 'PB')[index]}"


@dataclass(frozen=True)
class CachedFileInfo:
    file_name: str
    file_path: Path
    file_revision_hash: str
    blob_path: Path
    size_on_disk: int
    blob_last_accessed: float
    blob_last_modified: float

    @property
    def blob_last_accessed_str(self) -> str:
        return format_timesince(self.blob_last_accessed)

    @property
    def blob_last_modified_str(self) -> str:
        return format_timesince(self.blob_last_modified)

    @property
    def size_on_disk_str(self) -> str:
        return _size(self.size_on_disk)


@dataclass(frozen=True)
class CachedRevisionInfo:
    commit_hash: str
    snapshot_path: Path
    size_on_disk: int
    files: frozenset[CachedFileInfo]
    last_modified: float

    @property
    def last_modified_str(self) -> str:
        return format_timesince(self.last_modified)

    @property
    def size_on_disk_str(self) -> str:
        return _size(self.size_on_disk)

    @property
    def nb_files(self) -> int:
        return len(self.files)


@dataclass(frozen=True)
class CachedRepoInfo:
    repo_id: str
    repo_type: str
    repo_path: Path
    size_on_disk: int
    nb_files: int
    revisions: frozenset[CachedRevisionInfo]
    last_accessed: float
    last_modified: float

    @property
    def last_accessed_str(self) -> str:
        return format_timesince(self.last_accessed)

    @property
    def last_modified_str(self) -> str:
        return format_timesince(self.last_modified)

    @property
    def size_on_disk_str(self) -> str:
        return _size(self.size_on_disk)


@dataclass(frozen=True)
class ModelScopeCacheInfo:
    size_on_disk: int
    repos: frozenset[CachedRepoInfo]
    warnings: list[CorruptedCacheException]

    @property
    def size_on_disk_str(self) -> str:
        return _size(self.size_on_disk)

    def export_as_table(self) -> str:
        rows = [
            [
                repo.repo_id,
                repo.repo_type,
                revision.commit_hash,
                repo.size_on_disk_str,
                repo.nb_files,
                repo.last_accessed_str,
                repo.last_modified_str,
                str(repo.repo_path),
            ]
            for repo in sorted(self.repos, key=lambda r: (r.repo_id, str(r.repo_path)))
            for revision in sorted(repo.revisions, key=lambda r: r.commit_hash)
        ]
        return tabulate(
            rows,
            [
                "REPO ID",
                "REPO TYPE",
                "REVISION",
                "SIZE ON DISK",
                "NB FILES",
                "LAST_ACCESSED",
                "LAST_MODIFIED",
                "LOCAL PATH",
            ],
        )


def _details(location: CacheLocation) -> CachedRepoInfo:
    hints: dict[str, str] = {}
    index = location.path / ".msc"
    if index.exists():
        entries = read_legacy_metadata(index)
        if not isinstance(entries, list):
            raise CacheError(f"Invalid legacy file index: {index}")
        for entry in entries:
            if not isinstance(entry, dict):
                raise CacheError(f"Invalid legacy file entry: {index}")
            value = entry.get("Path")
            if not isinstance(value, str):
                raise CacheError(f"Invalid legacy file path: {index}")
            rel = validate_relative(value, "cached file")
            hints[rel] = str(entry.get("Revision", ""))
    revisions = []
    names = location.revisions if location.layout == "standard" else ("unknown",)
    version = location.path / ".mv"
    if location.layout != "standard" and version.is_file() and not version.is_symlink():
        raw = version.read_text(encoding="utf-8").strip()
        if raw:
            names = ((raw.split("Revision:", 1)[1].split(",", 1)[0] if "Revision:" in raw else raw),)
    for name in names:
        snapshot = location.path / "snapshots" / name if location.layout == "standard" else location.path
        files = []
        for path in payload_files(snapshot):
            contained_path(snapshot, path)
            info = path.stat()
            rel = path.relative_to(snapshot).as_posix()
            files.append(
                CachedFileInfo(path.name, path, hints.get(rel, ""), path, info.st_size, info.st_atime, info.st_mtime)
            )
        revisions.append(
            CachedRevisionInfo(
                name,
                snapshot,
                sum(f.size_on_disk for f in files),
                frozenset(files),
                max((f.blob_last_modified for f in files), default=snapshot.stat().st_mtime),
            )
        )
    all_files = {f for revision in revisions for f in revision.files}
    info = location.path.stat()
    return CachedRepoInfo(
        location.repo_id or "unknown",
        location.repo_type or "unknown",
        location.path,
        sum(f.size_on_disk for f in all_files),
        len(all_files),
        frozenset(revisions),
        max((f.blob_last_accessed for f in all_files), default=info.st_atime),
        max((f.blob_last_modified for f in all_files), default=info.st_mtime),
    )


def scan_cache_dir(cache_dir: str | Path | None = None) -> ModelScopeCacheInfo:
    """只读详细扫描，保留旧 frozen 视图、warnings 与缺失目录异常。"""
    root = get_cache_root(cache_dir=cache_dir)
    if not root.exists():
        raise CacheNotFound(f"Cache directory not found: {root}", cache_dir=str(root))
    if not root.is_dir():
        raise ValueError(f"Scan cache expects a directory: {root}")
    diagnostics: list[str] = []
    repos = []
    for location in discover_cache(root, diagnostics=diagnostics):
        try:
            repos.append(_details(location))
        except (CacheError, OSError, ValueError, TypeError) as exc:
            diagnostics.append(str(exc))
    return ModelScopeCacheInfo(
        sum(r.size_on_disk for r in repos),
        frozenset(repos),
        [CorruptedCacheException(message) for message in diagnostics],
    )


def find_reusable_legacy_repo_dir(
    repo_id: str, repo_type: str = "model", cache_dir: str | Path | None = None
) -> str | None:
    """兼容旧辅助入口；仅返回主 SDK 曾额外探测的历史路径。"""
    path = find_legacy_repo_dir(repo_id, repo_type, get_cache_root(cache_dir=cache_dir), extra_only=True)
    return str(path) if path is not None else None


__all__ = [
    "CachedFileInfo",
    "CachedRevisionInfo",
    "CachedRepoInfo",
    "ModelScopeCacheInfo",
    "scan_cache_dir",
    "scan_cache",
    "clear_cache",
    "get_cache_root",
    "resolve_cached_snapshot",
    "resolve_cached_file",
    "identify_cached_repo",
    "plan_cache_clear",
    "execute_cache_clear",
    "find_reusable_legacy_repo_dir",
]
