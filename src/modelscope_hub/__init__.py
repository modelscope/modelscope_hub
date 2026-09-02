"""ModelScope Hub SDK.

An OpenAPI-first Python SDK for interacting with the ModelScope Hub platform.

The public surface is intentionally small: most callers should construct a
single :class:`HubApi` instance and call its methods. The data classes
exported alongside it provide structured return types for type-checked code.
"""

from __future__ import annotations

from ._download import ProgressCallback, TqdmCallback
from .agent_idp import (
    generate_agent_key_pair,
    load_private_jwk,
    public_jwk_from_private,
    sign_agent_token_request,
    write_private_jwk,
)
from .api import HubApi
from .config import HubConfig, get_default_config, set_default_config
from .constants import License, RepoType, StudioVisibility, TokenScope, Visibility
from .errors import (
    APIError,
    AuthenticationError,
    CacheError,
    CacheNotFound,
    CorruptedCacheException,
    FileIntegrityError,
    HubError,
    InvalidParameter,
    NetworkError,
    NotExistError,
    NotFoundError,
    NotSupportedError,
    PermissionDeniedError,
    PermissionError,
    QuotaExceededError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    StorageError,
    ValidationError,
)
from .types import (
    AgentIdConfiguration,
    AgentIdentity,
    AgentIdentitySummary,
    AgentJWK,
    AgentToken,
    AgentTokenRecord,
    CachedRepoInfo,
    CacheInfo,
    CacheVerification,
    CommitInfo,
    FileInfo,
    PagedResult,
    RepoInfo,
    UserInfo,
    VerificationMismatch,
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
    "StudioVisibility",
    "TokenScope",
    "Visibility",
    # Progress callbacks
    "ProgressCallback",
    "TqdmCallback",
    # Agent-IDP local key helpers
    "generate_agent_key_pair",
    "load_private_jwk",
    "public_jwk_from_private",
    "sign_agent_token_request",
    "write_private_jwk",
    # Data classes
    "AgentIdentity",
    "AgentIdentitySummary",
    "AgentIdConfiguration",
    "AgentJWK",
    "AgentToken",
    "AgentTokenRecord",
    "CacheInfo",
    "CacheVerification",
    "CachedRepoInfo",
    "CommitInfo",
    "FileInfo",
    "PagedResult",
    "RepoInfo",
    "UserInfo",
    "VerificationMismatch",
    # Errors (canonical names per error-code spec)
    "APIError",
    "AuthenticationError",
    "CacheError",
    "CacheNotFound",
    "CorruptedCacheException",
    "FileIntegrityError",
    "HubError",
    "InvalidParameter",
    "NetworkError",
    "NotExistError",
    "NotSupportedError",
    "PermissionDeniedError",
    "QuotaExceededError",
    "RateLimitError",
    "RequestTimeoutError",
    "ServerError",
    "StorageError",
    # Backward-compatible aliases
    "NotFoundError",
    "PermissionError",
    "ValidationError",
]
