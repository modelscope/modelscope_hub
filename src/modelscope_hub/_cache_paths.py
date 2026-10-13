"""Shared cache-path helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .config import HubConfig
from .constants import DEFAULT_ENDPOINT


def endpoint_cache_root(cache_dir: str | Path, endpoint: str | None) -> Path:
    """Return the cache root reserved for *endpoint*.

    Keep the historical layout for the default service so existing caches stay
    usable. Other registries receive an opaque namespace derived from their
    normalized endpoint and therefore cannot reuse each other's artifacts.
    """
    root = Path(cache_dir)
    normalized = HubConfig.normalize_endpoint(endpoint)
    if normalized.casefold() == DEFAULT_ENDPOINT.casefold():
        return root
    namespace = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return root / "endpoints" / namespace
