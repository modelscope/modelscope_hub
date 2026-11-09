"""Adapt vLLM's legacy ModelScope path resolution without modifying its files."""

from __future__ import annotations

import functools
import importlib.abc
import inspect
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from ._cache_manager import resolve_cached_snapshot
from .compat.snapshot_download import snapshot_download
from .config import get_cache_root
from .errors import CacheNotFound
from .utils import append_user_agent

_TARGETS = {
    "vllm.transformers_utils.utils",
    "vllm.transformers_utils.processor",
    "vllm.transformers_utils.config",
}
_LOADERS = {"get_processor", "get_image_processor", "get_video_processor", "get_feature_extractor"}
_RESOLVERS = {"get_processor_config", "get_processor_cls_name_from_config", "get_video_processor_cls_name_from_config"}
_CACHED = {
    "cached_get_processor": "get_processor",
    "cached_get_image_processor": "get_image_processor",
    "cached_get_video_processor": "get_video_processor",
    "cached_get_feature_extractor": "get_feature_extractor",
    "_cached_get_video_processor_cls_name": "get_video_processor_cls_name_from_config",
}
_NON_WEIGHTS = [
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.gguf",
    "*.onnx",
    "*.onnx_data",
    "*.h5",
    "*.msgpack",
]


class _ResolvedModelPath(str):
    """A resolved path retaining its repo identity for later revision selection."""

    repo_id: str
    cache_dir: Path
    revision: str | None

    def __new__(cls, path: Path, repo_id: str, cache_dir: Path, revision: str | None):
        value = super().__new__(cls, str(path))
        value.repo_id = repo_id
        value.cache_dir = cache_dir
        value.revision = revision
        return value

    def __getnewargs__(self):
        return Path(str(self)), self.repo_id, self.cache_dir, self.revision


def _enabled() -> bool:
    return os.environ.get("VLLM_USE_MODELSCOPE", "").strip().lower() in {"1", "true"}


def _wrap_convert(original):
    @functools.wraps(original)
    def convert(model_repo: str, revision: str | None = None) -> str:
        if not _enabled():
            return original(model_repo)
        if Path(model_repo).exists():
            return model_repo
        try:
            cache_dir = get_cache_root()
            return _ResolvedModelPath(
                resolve_cached_snapshot(model_repo, revision=revision, cache_dir=cache_dir),
                model_repo,
                cache_dir,
                revision,
            )
        except CacheNotFound:
            # Keep cache misses as repo IDs; only loaders may retrieve files.
            return model_repo

    convert._modelscope_hub_adapter = True
    return convert


def _wrap_loader(original, *, download: bool):
    signature = inspect.signature(original)

    @functools.wraps(original)
    def load(*args: Any, **kwargs: Any):
        if not _enabled():
            return original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        model = bound.arguments.get("processor_name")
        if model is None or (not isinstance(model, _ResolvedModelPath) and Path(model).exists()):
            return original(*args, **kwargs)
        revision = bound.arguments.get("revision")
        extra = bound.arguments.get("kwargs", {})
        cache_dir = extra.get("cache_dir")
        if isinstance(model, _ResolvedModelPath):
            cache_dir = cache_dir if cache_dir is not None else model.cache_dir
            if "revision" not in bound.arguments:
                revision = model.revision
            model = model.repo_id
        from huggingface_hub.constants import HF_HUB_OFFLINE

        path: str | Path
        if not download or HF_HUB_OFFLINE or extra.get("local_files_only", False):
            path = resolve_cached_snapshot(model, revision=revision, cache_dir=cache_dir)
        else:
            token = extra.get("token")
            if isinstance(token, bool):
                token = None if token else ""
            # Resolve before vLLM substitutes its HF default revision.
            path = snapshot_download(
                model,
                revision=revision,
                cache_dir=cache_dir,
                token=token,
                ignore_patterns=_NON_WEIGHTS,
                user_agent=append_user_agent("invoked_by/vllm", extra.get("user_agent")),
            )
        bound.arguments["processor_name"] = str(path)
        return original(*bound.args, **bound.kwargs)

    setattr(load, "_modelscope_hub_adapter", True)
    return load


def _wrap_config(original):
    signature = inspect.signature(original)

    @functools.wraps(original)
    def load(*args: Any, **kwargs: Any):
        if not _enabled():
            return original(*args, **kwargs)
        from huggingface_hub.constants import HF_HUB_OFFLINE

        bound = signature.bind(*args, **kwargs)
        extra = bound.arguments.get("kwargs", {})
        if HF_HUB_OFFLINE or extra.get("local_files_only", False):
            model = bound.arguments.get("model")
            if model is not None and (isinstance(model, _ResolvedModelPath) or not Path(model).exists()):
                cache_dir = extra.get("cache_dir")
                revision = bound.arguments.get("revision")
                if isinstance(model, _ResolvedModelPath):
                    cache_dir = cache_dir if cache_dir is not None else model.cache_dir
                    if "revision" not in bound.arguments:
                        revision = model.revision
                    model = model.repo_id
                bound.arguments["model"] = str(
                    resolve_cached_snapshot(
                        str(model),
                        revision=revision,
                        cache_dir=cache_dir,
                    )
                )
        return original(*bound.args, **bound.kwargs)

    setattr(load, "_modelscope_hub_adapter", True)
    return load


def _adapt(module: ModuleType) -> None:
    if module.__name__.endswith(".utils"):
        original = getattr(module, "convert_model_repo_to_path", None)
        if original is not None and not getattr(original, "_modelscope_hub_adapter", False):
            setattr(module, "convert_model_repo_to_path", _wrap_convert(original))
    elif module.__name__.endswith(".config"):
        original = getattr(module, "get_config", None)
        if original is not None and not getattr(original, "_modelscope_hub_adapter", False):
            setattr(module, "get_config", _wrap_config(original))
    elif module.__name__.endswith(".processor"):
        utils = sys.modules.get("vllm.transformers_utils.utils")
        if utils is not None:
            setattr(module, "convert_model_repo_to_path", getattr(utils, "convert_model_repo_to_path"))
        for name in _LOADERS | _RESOLVERS:
            original = getattr(module, name, None)
            if original is not None and not getattr(original, "_modelscope_hub_adapter", False):
                setattr(module, name, _wrap_loader(original, download=name in _LOADERS))
        for name, target in _CACHED.items():
            cached, adapted = getattr(module, name, None), getattr(module, target, None)
            if cached is not None and adapted is not None and getattr(cached, "__wrapped__", None) is not adapted:
                setattr(module, name, functools.lru_cache(**cached.cache_parameters())(adapted))


class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        create = getattr(self.original, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        self.original.exec_module(module)
        _adapt(module)

    def __getattr__(self, name):
        return getattr(self.original, name)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS or not _enabled():
            return None
        # Delegate to existing finders to preserve custom import hooks.
        for finder in sys.meta_path[sys.meta_path.index(self) + 1 :]:
            find = getattr(finder, "find_spec", None)
            if find is None:
                continue
            spec = find(fullname, path, target)
            if spec is not None:
                if spec.loader is not None:
                    spec.loader = _Loader(spec.loader)
                return spec
        return None


def install() -> None:
    """Register lazy adapters without importing vLLM or changing disabled behavior."""
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    if _enabled():
        for name in (
            "vllm.transformers_utils.utils",
            "vllm.transformers_utils.config",
            "vllm.transformers_utils.processor",
        ):
            module = sys.modules.get(name)
            if module is not None:
                _adapt(module)
