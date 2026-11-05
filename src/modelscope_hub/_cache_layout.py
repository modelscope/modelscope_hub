"""Read-only cache layout resolution shared by downloads, scans, verification and clearing.

Repositories can live in several on-disk layouts written by different SDK
generations. This module is the single place that knows them:

* standard:  ``{root}/{type}s/{owner}--{name}/snapshots/{revision}``
* typed:     ``{root}/{type}s/{owner}/{name___}`` and ``{root}/hub/{type}s/{owner}/{name___}``
  (legacy SDK, dots in the name encoded as ``___``)
* flat:      ``{root}/{owner}/{name}`` and ``{root}/hub/{owner}/{name}`` (legacy
  ``cache_dir=...`` downloads and early ``modelscope_hub`` compat downloads)

Lookups never touch the network, never create directories and never load
credentials. Legacy pickle metadata is read with a primitive-only unpickler.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import uuid
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from filelock import FileLock, Timeout

from .config import get_cache_root
from .constants import TEMPORARY_FOLDER_NAME, RepoType
from .errors import CacheError, CacheNotFound, InvalidParameter
from .utils.logger import get_logger

logger = get_logger("cache")

IDENTITY_FILE = ".ms_repo.json"
LEGACY_METADATA_FILES = (".mdl", ".msc", ".mv")
_IDENTITY_SCHEMA = 1
_IGNORED_FILES = frozenset({IDENTITY_FILE, *LEGACY_METADATA_FILES, ".DS_Store"})
_PARTIAL_SUFFIXES = (".incomplete", ".parallel_tmp")
_SKIPPED_DIRS = frozenset({".git", ".lock", ".download_state", TEMPORARY_FOLDER_NAME})
_TYPE_SEGMENTS = {f"{kind.value}s": kind.value for kind in RepoType}
# Shared directories inside typed folders that belong to dataset tooling, not to a repository.
_NON_REPO_DIRS = frozenset({"downloads", "modules", "extracted", TEMPORARY_FOLDER_NAME})
# Top-level directories that are never ``owner`` folders of the flat layout.
_RESERVED_ROOT_DIRS = frozenset({"hub", "agent", *_TYPE_SEGMENTS})
_MAX_IDENTITY_BYTES = 1024 * 1024


@dataclass(frozen=True)
class CacheLocation:
    """A repository directory found in the cache.

    ``repo_id`` is exact when ``exact_identity`` is true (recorded by
    ``modelscope_hub``); otherwise it is decoded from legacy metadata or the
    directory name and is suitable for display only.
    """

    path: Path
    repo_id: str | None
    repo_type: str | None
    layout: str
    revisions: tuple[str, ...] = ()
    exact_identity: bool = False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_repo_id(repo_id: str) -> tuple[str, str]:
    """Return ``(owner, name)`` for a safe two-segment repository id."""
    if not isinstance(repo_id, str) or repo_id.count("/") != 1:
        raise InvalidParameter(f"repo_id must be a safe 'owner/name' identifier, got {repo_id!r}")
    owner, name = repo_id.split("/")
    for part in (owner, name):
        if not part or part in (".", "..") or any(char in part for char in ("\\", "\0", ":")):
            raise InvalidParameter(f"repo_id must be a safe 'owner/name' identifier, got {repo_id!r}")
    return owner, name


def normalize_type(repo_type: str | RepoType) -> str:
    """Return the singular repository type value, accepting plural folder names."""
    value = str(getattr(repo_type, "value", repo_type)).lower()
    if value in _TYPE_SEGMENTS:
        value = _TYPE_SEGMENTS[value]
    try:
        return RepoType(value).value
    except ValueError as exc:
        allowed = ", ".join(kind.value for kind in RepoType)
        raise InvalidParameter(f"Unknown repo_type {repo_type!r}. Expected one of: {allowed}.") from exc


def validate_relative(value: str, label: str = "path") -> str:
    """Reject paths that could escape their base directory once joined."""
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise InvalidParameter(f"Unsafe {label}: {value!r}")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive:
        raise InvalidParameter(f"Unsafe {label}: {value!r}")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise InvalidParameter(f"Unsafe {label}: {value!r}")
    return value


def contained_path(root: Path, target: Path) -> Path:
    """Return *target* unchanged after checking it resolves strictly inside *root*."""
    try:
        resolved_root = root.resolve()
        resolved = target.resolve()
    except (OSError, RuntimeError) as exc:
        raise CacheError(f"Cannot resolve cache path: {target}") from exc
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        raise CacheError(f"Cache path escapes its root {root}: {target}")
    return target


# ---------------------------------------------------------------------------
# Payload inspection
# ---------------------------------------------------------------------------
def payload_files(path: Path) -> Iterator[Path]:
    """Yield repository files below *path*, skipping SDK metadata and partial downloads.

    Directory symlinks are never followed; file symlinks are reported when they
    point at a file.
    """
    if not path.is_dir():
        return
    for base, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in _SKIPPED_DIRS and not os.path.islink(os.path.join(base, d)))
        for name in sorted(files):
            if name in _IGNORED_FILES or name.endswith(_PARTIAL_SUFFIXES):
                continue
            child = Path(base) / name
            if child.is_file():
                yield child


def has_payload(path: Path) -> bool:
    """Return True when *path* holds at least one repository file."""
    try:
        return next(payload_files(path), None) is not None
    except OSError:
        return False


def measure_payload(path: Path) -> tuple[int, int]:
    """Return ``(bytes, file_count)`` of the repository files below *path*."""
    size = count = 0
    for child in payload_files(path):
        try:
            size += child.stat().st_size
            count += 1
        except OSError:
            continue
    return size, count


def has_legacy_metadata(path: Path) -> bool:
    return any((path / name).is_file() for name in (IDENTITY_FILE, ".mdl", ".msc"))


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
class _PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"Cache metadata may only contain primitive values, found {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        raise pickle.UnpicklingError("Persistent references are not allowed in cache metadata")


def _is_primitive(value: Any, depth: int = 0) -> bool:
    if depth > 32:
        return False
    if value is None or type(value) in (str, bytes, int, float, bool):
        return True
    if type(value) in (list, tuple, set, frozenset):
        return all(_is_primitive(item, depth + 1) for item in value)
    if type(value) is dict:
        return all(_is_primitive(k, depth + 1) and _is_primitive(v, depth + 1) for k, v in value.items())
    return False


def read_legacy_metadata(path: Path) -> Any:
    """Read a legacy ``.msc``/``.mdl`` pickle without executing code or rewriting it."""
    if path.is_symlink():
        raise CacheError(f"Cache metadata must not be a symlink: {path}")
    try:
        with path.open("rb") as stream:
            value = _PrimitiveUnpickler(stream).load()
    except (OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError, RecursionError) as exc:
        raise CacheError(f"Invalid legacy cache metadata: {path}") from exc
    if not _is_primitive(value):
        raise CacheError(f"Invalid legacy cache metadata: {path}")
    return value


def _read_identity_file(marker: Path) -> dict[str, Any]:
    if marker.is_symlink():
        raise CacheError(f"Cache metadata must not be a symlink: {marker}")
    try:
        if marker.stat().st_size > _MAX_IDENTITY_BYTES:
            raise ValueError("identity file is too large")
        data = json.loads(marker.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema") != _IDENTITY_SCHEMA:
            raise ValueError("unsupported identity schema")
        validate_repo_id(data["repo_id"])
        data["repo_type"] = normalize_type(data["repo_type"])
        revisions = data.get("revisions", [])
        if not isinstance(revisions, list):
            raise ValueError("revisions must be a list")
        for revision in revisions:
            validate_relative(revision, "revision")
        data["exact"] = True
        return data
    except (OSError, ValueError, KeyError, TypeError, InvalidParameter) as exc:
        raise CacheError(f"Invalid cache identity: {marker}") from exc


def read_identity(path: Path, *, strict: bool = True) -> dict[str, Any] | None:
    """Return the recorded identity of a repository directory, if any.

    ``.ms_repo.json`` (written by ``modelscope_hub``) is exact. Legacy ``.mdl``
    stores the id with dots encoded as ``___`` and no repository type.
    """
    try:
        marker = path / IDENTITY_FILE
        if marker.exists() or marker.is_symlink():
            return _read_identity_file(marker)
        legacy = path / ".mdl"
        if legacy.exists() or legacy.is_symlink():
            data = read_legacy_metadata(legacy)
            raw = data.get("id") if isinstance(data, dict) else None
            if not isinstance(raw, str):
                raise CacheError(f"Invalid legacy cache identity: {legacy}")
            if raw == "unknown":
                return None
            validate_repo_id(raw)
            return {"repo_id": raw, "repo_type": None, "revisions": [], "exact": False}
    except (CacheError, InvalidParameter):
        if strict:
            raise
        logger.debug("Ignoring unreadable cache identity in %s", path, exc_info=True)
    return None


def _same_repo(recorded: str, repo_id: str) -> bool:
    owner, name = validate_repo_id(repo_id)
    candidates = {f"{owner}/{name}".casefold(), f"{owner}/{name.replace('.', '___')}".casefold()}
    return recorded.casefold() in candidates


def check_identity(path: Path, repo_id: str, repo_type: str) -> None:
    """Raise when *path* records a different repository than the one requested.

    Ids are compared case-insensitively so case-insensitive file systems do
    not report a conflict for the same repository.
    """
    data = read_identity(path, strict=False)
    if data is None:
        return
    matches = (
        data["repo_id"].casefold() == repo_id.casefold() if data.get("exact") else _same_repo(data["repo_id"], repo_id)
    )
    if not matches:
        raise CacheError(f"Repository identity conflict at {path}: it belongs to {data['repo_id']!r}, not {repo_id!r}")
    if data.get("repo_type") is not None and data["repo_type"] != normalize_type(repo_type):
        raise CacheError(f"Repository type conflict at {path}: it belongs to a {data['repo_type']} repository")


def _decode_legacy_id(raw: str) -> str:
    owner, name = raw.split("/")
    return f"{owner}/{name.replace('___', '.')}"


# ---------------------------------------------------------------------------
# Candidate paths
# ---------------------------------------------------------------------------
def repo_cache_path(repo_id: str, repo_type: str, root: Path) -> Path:
    """Return the standard repository directory ``{root}/{type}s/{owner}--{name}``."""
    owner, name = validate_repo_id(repo_id)
    return root / f"{normalize_type(repo_type)}s" / f"{owner}--{name}"


def legacy_candidates(repo_id: str, repo_type: str, root: Path) -> list[tuple[Path, str]]:
    """Return historical repository locations in reuse priority order with their layout."""
    owner, name = validate_repo_id(repo_id)
    kind = normalize_type(repo_type)
    safe = name.replace(".", "___")
    typed = [root / f"{kind}s" / owner / safe, root / "hub" / f"{kind}s" / owner / safe]
    flat: list[Path] = []
    if owner not in _RESERVED_ROOT_DIRS and not owner.startswith("."):
        flat = [root / owner / name, root / owner / safe, root / "hub" / owner / name, root / "hub" / owner / safe]
    unmasked = [root / f"{kind}s" / owner / name, root / "hub" / f"{kind}s" / owner / name]
    ordered = [(path, "typed") for path in typed] + [(path, "flat") for path in flat]
    ordered += [(path, "typed") for path in unmasked]
    seen: set[Path] = set()
    result = []
    for path, layout in ordered:
        if path not in seen:
            seen.add(path)
            result.append((path, layout))
    return result


def temp_candidates(repo_id: str, repo_type: str, root: Path) -> list[Path]:
    """Return legacy SDK temporary download folders that belong to one repository."""
    owner, name = validate_repo_id(repo_id)
    kind = normalize_type(repo_type)
    names = dict.fromkeys((name, name.replace(".", "___")))
    paths: list[Path] = []
    for base in (root / f"{kind}s", root / "hub" / f"{kind}s", root):
        paths.extend(base / TEMPORARY_FOLDER_NAME / owner / item for item in names)
    return list(dict.fromkeys(paths))


def snapshot_revisions(repo_path: Path, *, require_payload: bool = True) -> list[str]:
    """Return cached revision names below ``{repo_path}/snapshots``.

    Revisions containing ``/`` are known from the repository identity; their
    first path component is not reported as a separate revision.
    """
    snapshots = repo_path / "snapshots"
    if not snapshots.is_dir():
        return []
    contained_path(repo_path, snapshots)
    identity = read_identity(repo_path, strict=False)
    recorded = list(identity.get("revisions", [])) if identity and identity.get("exact") else []
    names = set(recorded)
    try:
        children = list(snapshots.iterdir())
    except OSError:
        children = []
    for child in children:
        if child.is_dir() and not any(item.startswith(child.name + "/") for item in recorded):
            names.add(child.name)
    present = []
    for name in sorted(names):
        snapshot = contained_path(repo_path, snapshots / name)
        if snapshot.is_dir() and (has_payload(snapshot) or not require_payload):
            present.append(name)
    return present


def find_legacy_repo_dir(
    repo_id: str,
    repo_type: str,
    root: Path,
    *,
    extra_only: bool = False,
) -> Path | None:
    """Return the historical repository directory that downloads should reuse.

    Priority: typed legacy folders first, then (only when no standard snapshot
    exists) flat and unmasked folders. ``extra_only`` reports only the latter
    group, which the legacy SDK used to probe on its own.
    """
    kind = normalize_type(repo_type)
    candidates = legacy_candidates(repo_id, kind, root)
    for path, _layout in candidates[:2]:
        if has_payload(contained_path(root, path)):
            check_identity(path, repo_id, kind)
            return None if extra_only else path
    if snapshot_revisions(repo_cache_path(repo_id, kind, root)):
        return None
    for path, _layout in candidates[2:]:
        if has_payload(contained_path(root, path)):
            check_identity(path, repo_id, kind)
            return path
    return None


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------
def _describe_revisions(snapshots: Path) -> str:
    if not snapshots.is_dir():
        return f"No snapshots directory found at: {snapshots}"
    try:
        children = sorted((path for path in snapshots.iterdir() if path.is_dir()), key=lambda path: path.name)
    except OSError as exc:
        return f"Unable to inspect cached revisions under {snapshots}: {exc}"
    if not children:
        return f"No cached revisions found under: {snapshots}"
    shown = [f"{path.name}{'' if has_payload(path) else ' (empty)'}" for path in children[:10]]
    if len(children) > 10:
        shown.append(f"... {len(children) - 10} more")
    return f"Cached revisions under {snapshots}: {', '.join(shown)}"


def _warn_legacy_revision(path: Path, revision: str | None) -> None:
    if revision is not None:
        warnings.warn(
            f"Reusing legacy cache {path}, which has no per-revision directory; revision {revision!r} is not "
            "verified. Pass reuse_legacy_cache=False to use revision-specific snapshots.",
            UserWarning,
            stacklevel=4,
        )


def _resolve_root(cache_dir: str | os.PathLike[str] | None) -> Path:
    return get_cache_root(cache_dir=cache_dir)


def resolve_snapshot(
    repo_id: str,
    repo_type: str = "model",
    *,
    revision: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    local_dir: str | os.PathLike[str] | None = None,
    reuse_legacy_cache: bool = True,
    require_payload: bool = True,
) -> tuple[Path, str | None]:
    """Return ``(path, revision)`` of a cached snapshot; ``revision`` is None for legacy folders."""
    validate_repo_id(repo_id)
    kind = normalize_type(repo_type)
    if revision is not None:
        validate_relative(revision, "revision")
    if local_dir is not None:
        if not os.fspath(local_dir):
            raise InvalidParameter("local_dir must not be empty")
        local = Path(os.fspath(local_dir)).expanduser().absolute()
        if local.is_dir() and (has_payload(local) or not require_payload):
            return local, revision
        raise CacheNotFound(
            "Cannot find a cached snapshot in the requested local directory and outgoing traffic has been "
            f"disabled. Local directory: {local}",
            cache_dir=str(local),
        )
    root = _resolve_root(cache_dir)
    if reuse_legacy_cache:
        legacy = find_legacy_repo_dir(repo_id, kind, root)
        if legacy is not None:
            logger.info("Found legacy cache at %s, reusing.", legacy)
            _warn_legacy_revision(legacy, revision)
            return legacy, None
    repo_path = repo_cache_path(repo_id, kind, root)
    contained_path(root, repo_path)
    if repo_path.exists():
        check_identity(repo_path, repo_id, kind)
    snapshots = repo_path / "snapshots"
    names = snapshot_revisions(repo_path, require_payload=require_payload)
    if revision is not None:
        target = contained_path(root, snapshots / revision)
        if target.is_dir() and (has_payload(target) or not require_payload):
            return target, revision
        raise CacheNotFound(
            f"Cannot find cached snapshot for {kind} repo '{repo_id}' at revision '{revision}' and outgoing traffic "
            f"has been disabled. {_describe_revisions(snapshots)}",
            cache_dir=str(snapshots / revision),
        )
    for default in ("master", "main"):
        if default in names:
            return contained_path(root, snapshots / default), default
    if len(names) == 1:
        logger.warning(
            "Using cached revision %s because no revision was provided and outgoing traffic is disabled.",
            names[0],
        )
        return contained_path(root, snapshots / names[0]), names[0]
    if names:
        shown = ", ".join(names[:10]) + (f", ... {len(names) - 10} more" if len(names) > 10 else "")
        raise CacheNotFound(
            f"Cached revision is ambiguous for {kind} repo '{repo_id}': no revision was provided and outgoing "
            f"traffic has been disabled. Available cached revisions: {shown}. Pass 'revision' to select one.",
            cache_dir=str(snapshots),
        )
    raise CacheNotFound(
        f"Cannot find cached snapshot for {kind} repo '{repo_id}' and outgoing traffic has been disabled. "
        f"{_describe_revisions(snapshots)}",
        cache_dir=str(snapshots),
    )


def resolve_cached_snapshot(
    repo_id: str,
    repo_type: str = "model",
    *,
    revision: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    local_dir: str | os.PathLike[str] | None = None,
    reuse_legacy_cache: bool = True,
) -> Path:
    """Return the existing local snapshot directory of a repository without network access.

    This is the directory ``snapshot_download(..., local_files_only=True)``
    returns. Raises :class:`CacheNotFound` when nothing is cached or the cached
    revision is ambiguous. A found directory is not proof that every file of
    the repository is present.
    """
    path, _revision = resolve_snapshot(
        repo_id,
        repo_type,
        revision=revision,
        cache_dir=cache_dir,
        local_dir=local_dir,
        reuse_legacy_cache=reuse_legacy_cache,
    )
    return path


def resolve_cached_file(
    repo_id: str,
    file_path: str,
    repo_type: str = "model",
    *,
    revision: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    local_dir: str | os.PathLike[str] | None = None,
    reuse_legacy_cache: bool = True,
) -> Path:
    """Return the existing local path of one repository file without network access."""
    validate_repo_id(repo_id)
    validate_relative(file_path, "file_path")
    kind = normalize_type(repo_type)
    if revision is not None:
        validate_relative(revision, "revision")
    if local_dir is not None:
        if not os.fspath(local_dir):
            raise InvalidParameter("local_dir must not be empty")
        local = Path(os.fspath(local_dir)).expanduser().absolute()
        target = contained_path(local, local / file_path)
        if target.is_file():
            return target
        raise CacheNotFound(
            f"Cannot find cached file '{file_path}' in the requested local directory and outgoing traffic has "
            f"been disabled. Local directory: {target.parent}",
            cache_dir=str(target.parent),
        )
    root = _resolve_root(cache_dir)
    if reuse_legacy_cache:
        legacy = find_legacy_repo_dir(repo_id, kind, root)
        if legacy is not None:
            target = contained_path(root, legacy / file_path)
            if target.is_file():
                _warn_legacy_revision(legacy, revision)
                return target
    repo_path = repo_cache_path(repo_id, kind, root)
    contained_path(root, repo_path)
    if repo_path.exists():
        check_identity(repo_path, repo_id, kind)
    snapshots = repo_path / "snapshots"
    if revision is not None:
        target = contained_path(root, snapshots / revision / file_path)
        if target.is_file():
            return target
        raise CacheNotFound(
            f"Cannot find cached file '{file_path}' for {kind} repo '{repo_id}' at revision '{revision}' and "
            f"outgoing traffic has been disabled. {_describe_revisions(snapshots)}",
            cache_dir=str(target.parent),
        )
    names = snapshot_revisions(repo_path)
    for default in ("master", "main"):
        target = contained_path(root, snapshots / default / file_path)
        if default in names and target.is_file():
            return target
    candidates = [contained_path(root, snapshots / name / file_path) for name in names]
    matches = [target for target in candidates if target.is_file()]
    if len(matches) == 1:
        logger.warning(
            "Using cached file from revision %s because no revision was provided and outgoing traffic is disabled.",
            matches[0].relative_to(snapshots).parts[0],
        )
        return matches[0]
    if matches:
        shown = ", ".join(path.relative_to(snapshots).parent.as_posix() for path in matches[:10])
        raise CacheNotFound(
            f"Cached revision is ambiguous for {kind} repo '{repo_id}': file '{file_path}' exists in cached "
            f"revisions {shown}. Pass 'revision' to select one.",
            cache_dir=str(snapshots),
        )
    raise CacheNotFound(
        f"Cannot find cached file '{file_path}' for {kind} repo '{repo_id}' and outgoing traffic has been disabled. "
        f"{_describe_revisions(snapshots)}",
        cache_dir=str(snapshots),
    )


# ---------------------------------------------------------------------------
# Download targets
# ---------------------------------------------------------------------------
def download_output(
    repo_id: str,
    repo_type: str,
    root: Path,
    revision: str,
    local_dir: Path | None,
    reuse_legacy_cache: bool,
) -> tuple[Path, bool]:
    """Return ``(directory, is_standard_snapshot)`` that a download writes into."""
    validate_repo_id(repo_id)
    validate_relative(revision, "revision")
    if local_dir is not None:
        return Path(local_dir).expanduser().absolute(), False
    kind = normalize_type(repo_type)
    if reuse_legacy_cache:
        legacy = find_legacy_repo_dir(repo_id, kind, root)
        if legacy is not None:
            logger.info("Found legacy cache at %s, reusing.", legacy)
            return legacy, False
    repo_path = repo_cache_path(repo_id, kind, root)
    contained_path(root, repo_path)
    if repo_path.exists():
        check_identity(repo_path, repo_id, kind)
        if (
            repo_path.name.count("--") != 1
            and read_identity(repo_path, strict=False) is None
            and has_payload(repo_path)
        ):
            raise CacheError(f"Cannot bind an ambiguous existing cache directory to a repository: {repo_path}")
    return contained_path(root, repo_path / "snapshots" / revision), True


def record_identity(repo_id: str, repo_type: str, root: Path, revision: str) -> None:
    """Record the repository identity once per new revision (no per-file rewrites)."""
    kind = normalize_type(repo_type)
    validate_relative(revision, "revision")
    repo_path = repo_cache_path(repo_id, kind, root)
    current = read_identity(repo_path, strict=False)
    if current is None and repo_path.name.count("--") != 1 and has_payload(repo_path):
        raise CacheError(f"Cannot bind an ambiguous existing cache directory to a repository: {repo_path}")
    if current is not None and current.get("exact") and revision in current.get("revisions", []):
        if current["repo_id"].casefold() != repo_id.casefold() or current["repo_type"] != kind:
            check_identity(repo_path, repo_id, kind)
        return
    try:
        repo_path.mkdir(parents=True, exist_ok=True)
        lock_dir = root / ".lock"
        lock_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(f"identity\0{kind}\0{repo_path.name}".encode()).hexdigest()
        with FileLock(str(lock_dir / f"identity_{key}.lock"), timeout=1):
            check_identity(repo_path, repo_id, kind)
            current = read_identity(repo_path, strict=False)
            if current is None and repo_path.name.count("--") != 1 and has_payload(repo_path):
                raise CacheError(f"Cannot bind an ambiguous existing cache directory to a repository: {repo_path}")
            revisions = list(current.get("revisions", [])) if current and current.get("exact") else []
            if revision in revisions:
                return
            data = {
                "schema": _IDENTITY_SCHEMA,
                "repo_id": current["repo_id"] if current and current.get("exact") else repo_id,
                "repo_type": kind,
                "revisions": sorted({*revisions, revision}),
            }
            temp = repo_path / f".ms_repo.{uuid.uuid4().hex}.tmp"
            try:
                temp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
                os.replace(temp, repo_path / IDENTITY_FILE)
            finally:
                temp.unlink(missing_ok=True)
    except (OSError, Timeout) as exc:
        if repo_path.name.count("--") != 1:
            raise CacheError(f"Cannot safely record an ambiguous repository identity: {repo_path}") from exc
        logger.warning("Cannot persist cache identity at %s; validated files remain reusable", repo_path)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def _legacy_revision(path: Path) -> str | None:
    marker = path / ".mv"
    if not marker.is_file() or marker.is_symlink():
        return None
    try:
        raw = marker.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if "Revision:" in raw:
        raw = raw.split("Revision:", 1)[1].split(",", 1)[0]
    return raw or None


def _make_location(path: Path, kind: str | None, layout: str, derived: str | None) -> CacheLocation:
    identity = read_identity(path)
    repo_id = derived
    exact = False
    if identity is not None:
        raw = identity["repo_id"]
        exact = bool(identity.get("exact"))
        if layout == "standard":
            if raw.replace("/", "--").casefold() != path.name.casefold():
                raise CacheError(f"Cache identity {raw!r} does not match its directory: {path}")
            repo_id = raw
        else:
            recorded_owner, recorded_name = validate_repo_id(raw)
            allowed_names = {recorded_name, recorded_name.replace(".", "___")}
            if not exact:
                allowed_names.add(recorded_name.replace("___", "."))
            if path.parent.name.casefold() != recorded_owner.casefold() or path.name.casefold() not in {
                name.casefold() for name in allowed_names
            }:
                raise CacheError(f"Cache identity {raw!r} conflicts with its legacy directory: {path}")
            repo_id = raw if exact else _decode_legacy_id(raw)
        recorded_kind = identity.get("repo_type")
        if recorded_kind is not None:
            if kind is not None and recorded_kind != kind:
                raise CacheError(f"Cache repository type {recorded_kind!r} conflicts with its directory: {path}")
            kind = recorded_kind
    if repo_id is not None:
        try:
            validate_repo_id(repo_id)
        except InvalidParameter:
            repo_id = None
    if layout == "standard":
        revisions = tuple(snapshot_revisions(path))
    else:
        legacy_revision = _legacy_revision(path)
        revisions = (legacy_revision,) if legacy_revision else ()
    return CacheLocation(path, repo_id, kind, layout, revisions, exact)


def discover_cache(
    root: Path,
    *,
    repo_type: str | None = None,
    diagnostics: list[str] | None = None,
    strict: bool = False,
    include_aliases: bool = False,
) -> list[CacheLocation]:
    """Return repository directories that can be proven to be SDK cache entries.

    Standard folders need a ``snapshots`` directory or a recorded identity;
    legacy and flat folders need legacy SDK metadata. Unknown directories are
    never returned. With ``strict`` any path escaping *root* or any identity
    conflict raises instead of being reported in ``diagnostics``.
    """
    kinds = [normalize_type(repo_type)] if repo_type is not None else [kind.value for kind in RepoType]
    found: list[CacheLocation] = []
    aliases: list[Path] = []
    seen: set[Path] = set()

    def note(message: str) -> None:
        if diagnostics is not None:
            diagnostics.append(message)

    def guard(path: Path) -> bool:
        try:
            contained_path(root, path)
            return True
        except CacheError as exc:
            if strict:
                raise
            note(str(exc))
            return False

    def add(path: Path, kind: str | None, layout: str, derived: str | None) -> None:
        if not guard(path):
            return
        try:
            location = _make_location(path, kind, layout, derived)
        except (CacheError, InvalidParameter) as exc:
            if strict:
                raise CacheError(f"Cannot prove ownership of cache directory {path}: {exc}") from exc
            note(f"Skipped {path}: {exc}")
            return
        if repo_type is not None and location.repo_type is not None and location.repo_type != repo_type:
            return
        if path not in seen:
            seen.add(path)
            found.append(location)

    def children(path: Path) -> list[Path]:
        try:
            return sorted(path.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            note(f"Cannot list {path}: {exc}")
            return []

    for prefix in (root, root / "hub"):
        if prefix != root and (not prefix.is_dir() or not guard(prefix)):
            continue
        for kind in kinds:
            typed = prefix / f"{kind}s"
            if not typed.is_dir() or not guard(typed):
                continue
            for entry in children(typed):
                if entry.name.startswith(".") or entry.name in _NON_REPO_DIRS:
                    continue
                if entry.is_symlink():
                    if guard(entry):
                        aliases.append(entry)
                    continue
                if not entry.is_dir():
                    continue
                if (entry / "snapshots").is_dir() or (entry / IDENTITY_FILE).exists():
                    derived = entry.name.replace("--", "/") if entry.name.count("--") == 1 else None
                    add(entry, kind, "standard", derived)
                    continue
                for repo in children(entry):
                    if repo.name.startswith("."):
                        continue
                    if repo.is_symlink():
                        if guard(repo):
                            aliases.append(repo)
                        continue
                    if not repo.is_dir():
                        continue
                    if has_legacy_metadata(repo):
                        add(repo, kind, "typed", f"{entry.name}/{repo.name.replace('___', '.')}")
                    elif has_payload(repo):
                        note(f"Skipped unidentified directory without SDK metadata: {repo}")
        for owner in children(prefix):
            if owner.name in _RESERVED_ROOT_DIRS or owner.name.startswith("."):
                continue
            if owner.is_symlink():
                guard(owner)
                continue
            if not owner.is_dir():
                continue
            for repo in children(owner):
                if repo.name.startswith("."):
                    continue
                if repo.is_symlink():
                    if guard(repo):
                        aliases.append(repo)
                    continue
                if not repo.is_dir():
                    continue
                if has_legacy_metadata(repo):
                    add(repo, None, "flat", f"{owner.name}/{repo.name.replace('___', '.')}")
    if include_aliases:
        targets = {location.path.resolve(): location for location in found}
        for alias in aliases:
            location = targets.get(alias.resolve())
            if location is not None:
                found.append(
                    CacheLocation(alias, location.repo_id, location.repo_type, "alias", (), location.exact_identity)
                )
    return found


def identify_cached_repo(path: str | os.PathLike[str]) -> str | None:
    """Return the repository id of a cached snapshot directory, or None if unknown.

    Uses recorded identities first and falls back to unambiguous standard
    directory names. Nothing is created or rewritten.
    """
    target = Path(path).expanduser().absolute()
    for candidate in (target, *target.parents):
        identity = read_identity(candidate, strict=False)
        if identity is not None:
            return str(identity["repo_id"]) if identity.get("exact") else _decode_legacy_id(identity["repo_id"])
        if candidate.parent.name in _TYPE_SEGMENTS and candidate.name.count("--") == 1:
            if (candidate / "snapshots").is_dir() and target.is_relative_to(candidate / "snapshots"):
                repo_id = candidate.name.replace("--", "/")
                try:
                    validate_repo_id(repo_id)
                    return repo_id
                except InvalidParameter:
                    return None
        if candidate.name in _TYPE_SEGMENTS or candidate == Path(candidate.anchor):
            break
    return None
