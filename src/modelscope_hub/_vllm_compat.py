"""Adapt upstream vLLM ModelScope loaders without rewriting third-party files."""

from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import inspect
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from ._cache_layout import resolve_cached_snapshot
from .compat.snapshot_download import snapshot_download
from .errors import CacheNotFound
from .utils import append_user_agent

_TARGETS = frozenset(
    {"vllm.transformers_utils.utils", "vllm.transformers_utils.processor", "vllm.transformers_utils.config"}
)
_LOADERS = frozenset({"get_processor", "get_image_processor", "get_video_processor", "get_feature_extractor"})
_RESOLVERS = frozenset(
    {"get_processor_config", "get_processor_cls_name_from_config", "get_video_processor_cls_name_from_config"}
)
_CACHED = {
    "cached_get_processor": "get_processor",
    "cached_get_feature_extractor": "get_feature_extractor",
    "_cached_get_video_processor_cls_name": "get_video_processor_cls_name_from_config",
}
_NON_WEIGHTS = ["*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.gguf", "*.onnx", "*.h5"]


def _enabled() -> bool:
    return os.environ.get("VLLM_USE_MODELSCOPE", "").strip().lower() in {"1", "true"}


def _resolve(model_repo: str, revision: str | None = None) -> str:
    if not _enabled() or Path(model_repo).exists():
        return model_repo
    try:
        return str(resolve_cached_snapshot(model_repo, revision=revision))
    except CacheNotFound:
        # Keep the repo ID on cache misses; loaders handle retrieval, not nonexistent local paths.
        return model_repo


def _wrap_convert(original):
    @functools.wraps(original)
    def convert(model_repo: str, revision: str | None = None) -> str:
        if not _enabled():
            return original(model_repo)
        return _resolve(model_repo, revision)

    convert._modelscope_hub_adapter = True
    return convert


def _wrap_loader(original, *, download=True):
    signature = inspect.signature(original)

    @functools.wraps(original)
    def load(*args: Any, **kwargs: Any):
        if not _enabled():
            return original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        model = bound.arguments.get("processor_name")
        if model is None or Path(model).exists():
            return original(*args, **kwargs)
        revision = bound.arguments.get("revision")
        extra = bound.arguments.get("kwargs", {})
        cache_dir = extra.get("cache_dir")
        if not download:
            bound.arguments["processor_name"] = str(resolve_cached_snapshot(model, revision=revision))
            return original(*bound.args, **bound.kwargs)
        token = extra.get("token")
        if isinstance(token, bool):
            token = None if token else ""
        # Fetch non-weight files only in loaders; converters and offline lookups never access the network.
        from huggingface_hub.constants import HF_HUB_OFFLINE

        path = snapshot_download(
            model,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            ignore_patterns=_NON_WEIGHTS,
            user_agent=append_user_agent("invoked_by/vllm", extra.get("user_agent")),
            local_files_only=HF_HUB_OFFLINE or bool(extra.get("local_files_only", False)),
        )
        bound.arguments["processor_name"] = path
        return original(*bound.args, **bound.kwargs)

    load._modelscope_hub_adapter = True
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
        if not HF_HUB_OFFLINE and not extra.get("local_files_only", False):
            return original(*args, **kwargs)
        model = bound.arguments.get("model")
        if model is not None and not Path(model).exists():
            # Upstream vLLM's remote file_exists ignores HF offline mode; resolve the local cache first.
            bound.arguments["model"] = str(
                resolve_cached_snapshot(str(model), revision=bound.arguments.get("revision"))
            )
        return original(*bound.args, **bound.kwargs)

    load._modelscope_hub_adapter = True
    return load


def _adapt(module: ModuleType) -> None:
    if module.__name__ == "vllm.transformers_utils.utils":
        original = getattr(module, "convert_model_repo_to_path", None)
        if original is not None and not getattr(original, "_modelscope_hub_adapter", False):
            setattr(module, "convert_model_repo_to_path", _wrap_convert(original))
    elif module.__name__ == "vllm.transformers_utils.config":
        original = getattr(module, "get_config", None)
        if original is not None and not getattr(original, "_modelscope_hub_adapter", False):
            setattr(module, "get_config", _wrap_config(original))
    elif module.__name__ == "vllm.transformers_utils.processor":
        utils = sys.modules.get("vllm.transformers_utils.utils")
        if utils is not None:
            setattr(module, "convert_model_repo_to_path", getattr(utils, "convert_model_repo_to_path"))
        for name in _LOADERS | _RESOLVERS:
            original = getattr(module, name, None)
            if original is not None and not getattr(original, "_modelscope_hub_adapter", False):
                setattr(module, name, _wrap_loader(original, download=name in _LOADERS))
        for name, target in _CACHED.items():
            cached = getattr(module, name, None)
            adapted = getattr(module, target, None)
            if cached is not None and adapted is not None and cached.__wrapped__ is not adapted:
                parameters = cached.cache_parameters()
                setattr(module, name, functools.lru_cache(**parameters)(adapted))


class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        _adapt(module)

    def __getattr__(self, name):
        return getattr(self.original, name)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS or not _enabled():
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


def install() -> None:
    """Register a lazy adapter, inactive when disabled, without importing vLLM."""
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    if not _enabled():
        return
    for name in sorted(_TARGETS, reverse=True):
        module = sys.modules.get(name)
        if module is not None:
            _adapt(module)


def uninstall() -> None:
    """Remove this adapter and restore upstream functions with their original signatures."""
    sys.meta_path[:] = [finder for finder in sys.meta_path if not isinstance(finder, _Finder)]
    for name in _TARGETS:
        module = sys.modules.get(name)
        if module is None:
            continue
        for attribute, value in list(vars(module).items()):
            if attribute not in _CACHED and getattr(value, "_modelscope_hub_adapter", False):
                setattr(module, attribute, value.__wrapped__)
        if name.endswith(".processor"):
            for attribute, target in _CACHED.items():
                cached = getattr(module, attribute, None)
                original = getattr(module, target, None)
                if cached is not None and original is not None:
                    setattr(module, attribute, functools.lru_cache(**cached.cache_parameters())(original))
