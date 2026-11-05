"""旧文件缓存类的兼容实现；正常下载及只读扫描不使用此可写类。"""

from __future__ import annotations

import hashlib
import os
import pickle
import tempfile
import threading
from pathlib import Path
from shutil import move
from typing import Any

from .._cache_clear import execute_cache_clear, plan_cache_clear
from .._cache_layout import contained_path, read_legacy_metadata, validate_relative
from ..config import get_cache_root
from ..errors import CacheError, InvalidParameter
from ..utils.file_utils import compute_hash

enable_default_hash_validation = (
    os.getenv("MODELSCOPE_ENABLE_DEFAULT_HASH_VALIDATION", "False").strip().lower() == "true"
)


class FileSystemCache:
    KEY_FILE_NAME = ".msc"
    MODEL_META_FILE_NAME = ".mdl"
    MODEL_META_MODEL_ID = "id"
    MODEL_VERSION_FILE_NAME = ".mv"

    def __init__(self, cache_root_location: str, **kwargs: Any):
        if not os.fspath(cache_root_location):
            raise CacheError("cache_root_location must not be empty")
        self._cache_lock = threading.RLock()
        self.cache_root_location = str(Path(cache_root_location).expanduser().absolute())
        Path(self.cache_root_location).mkdir(parents=True, exist_ok=True)
        self.cached_files: list[dict] = []
        self.load_cache()

    def get_root_location(self):
        return self.cache_root_location

    def load_cache(self):
        self.cached_files = []
        path = Path(self.cache_root_location) / self.KEY_FILE_NAME
        if path.exists():
            try:
                entries = read_legacy_metadata(path)
                if isinstance(entries, list):
                    for entry in entries:
                        if not isinstance(entry, dict):
                            raise CacheError("Invalid cached file entry")
                        validate_relative(entry.get("Path"), "cached file")
                    self.cached_files = entries
            except (CacheError, InvalidParameter, ValueError, TypeError):
                # 不信任损坏 hint，但不删除用户文件或改名索引。
                self.cached_files = []

    def _save_cached_files_unlocked(self):
        root = Path(self.cache_root_location)
        fd, temp = tempfile.mkstemp(suffix=".tmp", dir=root)
        try:
            with os.fdopen(fd, "wb") as stream:
                pickle.dump(list(self.cached_files), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, root / self.KEY_FILE_NAME)
        finally:
            Path(temp).unlink(missing_ok=True)

    def save_cached_files(self):
        with self._cache_lock:
            self._save_cached_files_unlocked()

    def get_file(self, key):
        return None

    def put_file(self, key, location):
        return None

    def remove_key(self, key):
        with self._cache_lock:
            if key in self.cached_files:
                self.cached_files.remove(key)
                self._save_cached_files_unlocked()

    def exists(self, key):
        return key in self.cached_files

    def clear_cache(self):
        root = get_cache_root()
        location = Path(self.cache_root_location).resolve()
        meta = read_legacy_metadata(location / self.MODEL_META_FILE_NAME)
        if not isinstance(meta, dict) or not location.is_relative_to(root) or location == root:
            raise CacheError("Only identified repositories beneath the active cache root may be cleared")
        plan = plan_cache_clear(root, "model", meta.get("id"))
        if any(Path(t.path).resolve() != location for t in plan.targets):
            raise CacheError("Legacy cache clear would affect a different cache location")
        execute_cache_clear(plan)
        self.cached_files = []

    def hash_name(self, key):
        return hashlib.sha256(key.encode()).hexdigest()

    def _file_path(self, relative: str) -> Path:
        validate_relative(relative, "cached file")
        root = Path(self.cache_root_location).resolve()
        return contained_path(root, root / relative)


class ModelFileSystemCache(FileSystemCache):
    """保留 owner/name、单版本文件、.msc/.mdl/.mv 的旧存储契约。"""

    def __init__(self, cache_root, owner=None, name=None):
        if owner is None or name is None:
            super().__init__(str(cache_root))
            self.load_model_meta()
        else:
            validate_relative(owner, "owner")
            validate_relative(name, "name")
            if "/" in owner or "/" in name:
                raise CacheError("Cache owner and name must be individual path components")
            super().__init__(str(Path(cache_root) / owner / name))
            self.model_meta = {self.MODEL_META_MODEL_ID: f"{owner}/{name}"}
            self.save_model_meta()
        self.cached_model_revision = self.load_model_version()

    def load_model_meta(self):
        self.model_meta = {self.MODEL_META_MODEL_ID: "unknown"}
        path = Path(self.cache_root_location) / self.MODEL_META_FILE_NAME
        if path.exists():
            try:
                data = read_legacy_metadata(path)
                if isinstance(data, dict):
                    self.model_meta = data
            except CacheError:
                pass

    def load_model_version(self):
        path = Path(self.cache_root_location) / self.MODEL_VERSION_FILE_NAME
        return path.read_text().strip() if path.is_file() and not path.is_symlink() else None

    def save_model_version(self, revision_info):
        text = (
            f"Revision:{revision_info['Revision']},CreatedAt:{revision_info['CreatedAt']}"
            if isinstance(revision_info, dict)
            else str(revision_info)
        )
        target = self._file_path(self.MODEL_VERSION_FILE_NAME)
        if target.is_symlink():
            raise CacheError("Cache metadata must not be a symlink")
        target.write_text(text)

    def get_model_id(self):
        return self.model_meta.get(self.MODEL_META_MODEL_ID, "unknown")

    def save_model_meta(self):
        target = self._file_path(self.MODEL_META_FILE_NAME)
        if target.is_symlink():
            raise CacheError("Cache metadata must not be a symlink")
        target.write_bytes(pickle.dumps(self.model_meta))

    def get_file_by_path(self, file_path):
        return self._get_file(file_path)

    def _get_file(self, file_path, commit_id=None):
        target = self._file_path(file_path)
        for entry in list(self.cached_files):
            if entry["Path"] != file_path:
                continue
            revision = str(entry.get("Revision", ""))
            if commit_id is not None and not (revision.startswith(commit_id) or commit_id.startswith(revision)):
                continue
            if target.is_file():
                return str(target)
            self.remove_key(entry)
        return None

    def get_file_by_path_and_commit_id(self, file_path, commit_id):
        return self._get_file(file_path, commit_id)

    def get_file_by_info(self, model_file_info):
        key = {"Path": model_file_info["Path"], "Revision": model_file_info["Revision"]}
        return self.get_file_by_path(key["Path"]) if key in self.cached_files else None

    def exists(self, model_file_info):
        target = self.get_file_by_path_and_commit_id(model_file_info["Path"], model_file_info["Revision"])
        expected = model_file_info.get("Sha256")
        if target is None or expected is None:
            return False
        return not enable_default_hash_validation or compute_hash(target) == expected

    def remove_if_exists(self, model_file_info):
        for entry in list(self.cached_files):
            if entry["Path"] == model_file_info["Path"]:
                target = self._file_path(entry["Path"])
                self.remove_key(entry)
                target.unlink(missing_ok=True)
                break

    def put_file(self, model_file_info, model_file_location):
        target = self._file_path(model_file_info["Path"])
        self.remove_if_exists(model_file_info)
        target.parent.mkdir(parents=True, exist_ok=True)
        move(str(model_file_location), str(target))
        with self._cache_lock:
            self.cached_files.append({"Path": model_file_info["Path"], "Revision": model_file_info["Revision"]})
            self._save_cached_files_unlocked()
        return str(target)


__all__ = ["FileSystemCache", "ModelFileSystemCache"]
