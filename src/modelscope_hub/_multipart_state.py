"""Sparse, private multipart recovery hints."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from filelock import FileLock, Timeout

from ._multipart_protocol import MultipartError
from .config import _write_private_text
from .utils.logger import get_logger

logger = get_logger("multipart_state")
_LOCKS: weakref.WeakValueDictionary[str, object] = weakref.WeakValueDictionary()
_LOCK_GUARD = threading.Lock()


def _key(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _safe(path: Path, *, directory: bool) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise MultipartError("Unsafe multipart state path.")
    if hasattr(os, "getuid") and (info.st_uid != os.getuid() or info.st_mode & 0o077):
        raise MultipartError("Multipart state must be owner-only.")


@dataclass
class UploadState:
    url: str
    token: str = field(default="", repr=False)
    part_size: int = 0
    generation: int = 0
    dirty: list[int] = field(default_factory=list)
    last_action: str = "status"
    ready: bool = False


class StateStore:
    def __init__(
        self,
        home: Path,
        endpoint: str,
        credential: str | None,
        repo_id: str,
        repo_type: str,
        sha256: str,
        size: int,
        enabled: bool,
        timeout: int,
        max_bytes: int,
    ) -> None:
        self.binding = {
            "endpoint": endpoint,
            "repo_id": repo_id,
            "repo_type": repo_type,
            "sha256": sha256,
            "size": size,
        }
        scope = _key([endpoint, credential])
        self.root = home / "uploads" / "multipart" / scope / _key(self.binding)
        self.enabled = enabled
        self.timeout = timeout
        self.max_bytes = max_bytes
        self._path: Path | None = None
        self._memory: UploadState | None = None

    def _target(self, url: str) -> Path:
        parsed = urlsplit(url)
        return self.root / _key([parsed.scheme, parsed.netloc, parsed.path]) / ".state.json"

    def _read(self, path: Path) -> UploadState | None:
        try:
            _safe(path.parent, directory=True)
            _safe(path, directory=False)
            if path.stat().st_size > self.max_bytes:
                return None
            with path.open(encoding="utf-8") as stream:
                body = json.load(stream)
            if not isinstance(body, dict) or body.get("version") != 1 or body.get("binding") != self.binding:
                return None
            value = body.get("state")
            if not isinstance(value, dict) or not isinstance(value.get("url"), str):
                return None
            if self._target(value["url"]) != path:
                return None
            token = value.get("token", "")
            dirty = value.get("dirty", [])
            if not isinstance(token, str) or token.strip() != token or "\r" in token or "\n" in token:
                return None
            if not isinstance(dirty, list) or len(dirty) > 10000:
                return None
            if any(type(n) is not int or not 1 <= n <= 10000 for n in dirty):
                return None
            part_size = value.get("part_size")
            generation = value.get("generation")
            if type(part_size) is not int or part_size < 0 or type(generation) is not int or generation < 0:
                return None
            if type(value.get("ready")) is not bool or value.get("last_action") not in {
                "init",
                "part",
                "status",
                "complete",
            }:
                return None
            return UploadState(**value)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError):
            logger.warning("Unreadable multipart recovery hint; authenticated status will be used.")
            return None

    def pending(self) -> UploadState | None:
        if not self.enabled or not self.root.exists():
            return None
        _safe(self.root, directory=True)
        result = []
        for index, directory in enumerate(self.root.iterdir()):
            if index >= 16:
                raise MultipartError("Too many multipart targets for one content binding; check local state.")
            value = self._read(directory / ".state.json")
            if value and not value.ready and value.token and value.last_action == "complete":
                result.append(value)
        if len(result) > 1:
            raise MultipartError("Multiple pending completions require explicit target reconciliation.")
        return result[0] if result else None

    def _prepare(self, path: Path) -> None:
        for directory in (
            self.root.parent.parent.parent,
            self.root.parent.parent,
            self.root.parent,
            self.root,
            path.parent,
        ):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            _safe(directory, directory=True)

    @contextmanager
    def lease(self, url: str, check: Callable[[], None]) -> Iterator[UploadState | None]:
        path = self._target(url)
        lock_key = str(path)
        with _LOCK_GUARD:
            local = _LOCKS.get(lock_key)
            if local is None:
                local = threading.RLock()
                _LOCKS[lock_key] = local
        deadline = time.monotonic() + self.timeout
        while not local.acquire(timeout=0.1):  # type: ignore[attr-defined]
            check()
            if time.monotonic() >= deadline:
                raise MultipartError("Multipart lock wait exceeded its configured limit.")
        disk_lock = None
        try:
            check()
            if self.enabled:
                try:
                    self._prepare(path)
                    lock_path = path.with_name(".lock")
                    if lock_path.exists() or lock_path.is_symlink():
                        _safe(lock_path, directory=False)
                    disk_lock = FileLock(str(lock_path), mode=0o600)
                    while True:
                        check()
                        try:
                            disk_lock.acquire(timeout=0)
                            break
                        except Timeout:
                            if time.monotonic() >= deadline:
                                raise MultipartError("Multipart lock wait exceeded its configured limit.") from None
                            time.sleep(0.1)
                    self._path = path
                except OSError:
                    self.enabled = False
                    logger.warning(
                        "Multipart state is not writable; merged uploads may require retransmission after exit."
                    )
            if disk_lock is None:
                logger.warning(
                    "Multipart recovery is memory-only; cross-process single-writer coordination is required."
                )
            yield self._read(path) if self.enabled and path.exists() else self._memory
        finally:
            if disk_lock is not None:
                disk_lock.release()
            local.release()  # type: ignore[attr-defined]

    def save(self, value: UploadState) -> None:
        self._memory = value
        if self._path is None or not self.enabled:
            return
        try:
            _safe(self._path.parent, directory=True)
            if self._path.exists() or self._path.is_symlink():
                _safe(self._path, directory=False)
            data = json.dumps({"version": 1, "binding": self.binding, "state": asdict(value)}, separators=(",", ":"))
            if len(data.encode("utf-8")) > self.max_bytes:
                raise MultipartError("Multipart recovery hint exceeds its size limit.")
            _write_private_text(self._path, data)
        except OSError:
            self.enabled = False
            logger.warning("Could not persist multipart token; retain this process for completion recovery.")
