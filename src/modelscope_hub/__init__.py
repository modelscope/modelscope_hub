"""ModelScope Hub SDK.

An OpenAPI-first Python SDK for interacting with the ModelScope Hub platform.

The public surface is intentionally small: most callers should construct a
single :class:`HubApi` instance and call its methods. The data classes
exported alongside it provide structured return types for type-checked code.
"""

from __future__ import annotations

from ._download import ProgressCallback, TqdmCallback
from .api import HubApi
from .config import HubConfig, get_default_config, set_default_config
from .constants import License, RepoType, Visibility
from .errors import (
    APIError,
    AuthenticationError,
    CacheError,
    FileIntegrityError,
    HubError,
    NetworkError,
    NotFoundError,
    PermissionError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from .types import (
    CacheInfo,
    CachedRepoInfo,
    CommitInfo,
    FileInfo,
    PagedResult,
    RepoInfo,
    UserInfo,
)
from .version import __version__

__all__ = [
    "__version__",
    # Facade
    "HubApi",
    # Configuration
    "HubConfig",
    "get_default_config",
    "set_default_config",
    # Enums
    "License",
    "RepoType",
    "Visibility",
    # Progress callbacks
    "ProgressCallback",
    "TqdmCallback",
    # Data classes
    "CacheInfo",
    "CachedRepoInfo",
    "CommitInfo",
    "FileInfo",
    "PagedResult",
    "RepoInfo",
    "UserInfo",
    # Errors
    "APIError",
    "AuthenticationError",
    "CacheError",
    "FileIntegrityError",
    "HubError",
    "NetworkError",
    "NotFoundError",
    "PermissionError",
    "RateLimitError",
    "ServerError",
    "ValidationError",
]
