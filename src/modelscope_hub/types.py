"""Typed data containers returned by the SDK.

Every dataclass is constructible from a raw API payload via :func:`from_dict`,
which silently ignores fields the server may add in the future. This keeps the
client forward-compatible while still benefiting from static typing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Generic, TypedDict, TypeVar

from .constants import RepoType, Visibility

T = TypeVar("T")
_TDataclass = TypeVar("_TDataclass", bound="_FromDictMixin")


def _coerce_datetime(value: Any) -> datetime | None:
    """Best-effort conversion of an API timestamp into a :class:`datetime`."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        # ModelScope timestamps may arrive in seconds or milliseconds.
        seconds = value / 1000 if value > 1e12 else value
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


class _FromDictMixin:
    """Adds tolerant ``from_dict`` construction to a dataclass."""

    _field_aliases: ClassVar[dict[str, str]] = {}

    @classmethod
    def from_dict(cls: type[_TDataclass], data: Mapping[str, Any] | None) -> _TDataclass:
        if not data:
            return cls()  # type: ignore[call-arg]
        known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
        aliases = cls._field_aliases
        kwargs = {aliases.get(key, key): value for key, value in data.items() if aliases.get(key, key) in known}
        return cls(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class UserInfo(_FromDictMixin):
    _field_aliases: ClassVar[dict[str, str]] = {
        "name": "username",
        "avatar": "avatar_url",
    }
    _id_keys: ClassVar[tuple[str, ...]] = (
        "id",
        "Id",
        "ID",
        "user_id",
        "UserId",
        "userId",
        "uid",
        "Uid",
        "UID",
        "sub",
        "Sub",
    )
    _username_keys: ClassVar[tuple[str, ...]] = (
        "Username",
        "username",
        # Observed OIDC-style shape on newer /users/me responses: the login
        # handle may be in ``name`` while ``preferred_username`` can be empty or
        # a display value, so keep the same priority as get_current_username().
        "name",
        "Name",
        "preferred_username",
        "PreferredUsername",
        "preferredUsername",
        "user_name",
        "UserName",
        "login",
        "Login",
        "nickname",
        "Nickname",
    )
    _email_keys: ClassVar[tuple[str, ...]] = ("email", "Email", "mail", "Mail")
    _avatar_keys: ClassVar[tuple[str, ...]] = (
        "avatar_url",
        "avatarUrl",
        "AvatarUrl",
        "avatar",
        "Avatar",
        "picture",
        "Picture",
    )
    _description_keys: ClassVar[tuple[str, ...]] = (
        "description",
        "Description",
        "bio",
        "Bio",
        "introduction",
        "Introduction",
    )

    id: str | int | None = None
    username: str | None = None
    email: str | None = None
    avatar_url: str | None = None
    description: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> UserInfo:
        """Build user info from legacy ModelScope and newer OIDC-style keys."""
        if not isinstance(data, Mapping) or not data:
            return cls()
        return cls(
            id=cls._first_non_empty(data, cls._id_keys),
            username=cls._as_str_or_none(cls._first_non_empty(data, cls._username_keys)),
            email=cls._as_str_or_none(cls._first_non_empty(data, cls._email_keys)),
            avatar_url=cls._as_str_or_none(cls._first_non_empty(data, cls._avatar_keys)),
            description=cls._as_str_or_none(cls._first_non_empty(data, cls._description_keys)),
        )

    @staticmethod
    def _first_non_empty(data: Mapping[str, Any], keys: tuple[str, ...]) -> Any | None:
        for key in keys:
            if key not in data:
                continue
            value = data[key]
            if value is not None and value != "":
                return value
        return None

    @staticmethod
    def _as_str_or_none(value: Any | None) -> str | None:
        if value is None:
            return None
        return str(value)


# ---------------------------------------------------------------------------
# Agent-IDP
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class AgentJWK(_FromDictMixin):
    """An Ed25519 JSON Web Key used by the Agent-IDP protocol.

    ``d`` exists only in local private-key material. Server responses and public
    request payloads contain the other fields only; :meth:`to_dict` therefore
    excludes it unless explicitly requested.
    """

    kty: str = "OKP"
    crv: str = "Ed25519"
    x: str = ""
    kid: str = ""
    alg: str | None = None
    use: str | None = None
    d: str | None = field(default=None, repr=False)

    def to_dict(self, *, include_private: bool = False) -> dict[str, str]:
        """Return the public JWK, optionally including the private ``d`` value."""
        result = {"kty": self.kty, "crv": self.crv, "x": self.x, "kid": self.kid}
        if self.alg is not None:
            result["alg"] = self.alg
        if self.use is not None:
            result["use"] = self.use
        if include_private and self.d is not None:
            result["d"] = self.d
        return result


@dataclass(slots=True)
class AgentIdentity(_FromDictMixin):
    """A registered Agent-IDP identity returned by the OpenAPI service."""

    agent_id: str = ""
    agent_name: str = ""
    description: str | None = None
    token_expire_time: int | None = None
    principal: dict[str, Any] | None = None
    kid: str | None = None
    public_key: AgentJWK | None = None
    status: str | None = None
    create_time: str | None = None
    update_time: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> AgentIdentity:
        if not isinstance(data, Mapping):
            return cls()
        raw_key = data.get("public_key")
        principal = data.get("principal")
        return cls(
            agent_id=str(data.get("agent_id") or ""),
            agent_name=str(data.get("agent_name") or ""),
            description=data.get("description"),
            token_expire_time=data.get("token_expire_time"),
            principal=dict(principal) if isinstance(principal, Mapping) else None,
            kid=data.get("kid"),
            public_key=AgentJWK.from_dict(raw_key) if isinstance(raw_key, Mapping) else None,
            status=data.get("status"),
            create_time=data.get("create_time"),
            update_time=data.get("update_time"),
        )


@dataclass(slots=True)
class AgentIdentitySummary(_FromDictMixin):
    """The paginated, non-sensitive projection of an Agent-IDP identity."""

    agent_id: str = ""
    agent_name: str = ""
    kid: str | None = None
    status: str | None = None
    token_expire_time: int | None = None
    create_time: str | None = None


@dataclass(slots=True)
class AgentTokenRecord(_FromDictMixin):
    """One issued Agent JWT record returned by the service."""

    token_id: str = ""
    audience: str = ""
    issued_at: str | None = None
    expire_at: str | None = None
    status: str | None = None
    jwt: str | None = field(default=None, repr=False)


@dataclass(slots=True)
class AgentToken(_FromDictMixin):
    """A short-lived JWT issued by ``POST /agent_id/token``."""

    access_token: str = field(default="", repr=False)
    token_type: str = "Bearer"
    expire_at: int | None = None
    jti: str | None = None


@dataclass(slots=True)
class AgentIdConfiguration(_FromDictMixin):
    """OIDC discovery metadata served by Agent-IDP."""

    issuer: str | None = None
    token_endpoint: str | None = None
    jwks_uri: str | None = None
    registration_endpoint: str | None = None
    activity_endpoint: str | None = None
    id_token_signing_alg_values_supported: str | None = None


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class RepoInfo(_FromDictMixin):
    id: str | int | None = None
    owner: str | None = None
    name: str | None = None
    repo_type: RepoType | str | None = None
    visibility: Visibility | int | None = None
    description: str | None = None
    downloads: int = 0
    likes: int = 0
    created_at: datetime | str | int | None = None
    last_modified: datetime | str | int | None = None
    license: str | None = None
    tags: list[str] = field(default_factory=list)
    # OpenAPI native fields
    display_name: str | None = None
    file_size: int | None = None
    tasks: list[str] = field(default_factory=list)
    private: bool | None = None
    gated: bool | None = None
    login_required: bool | None = None
    # Studio-native fields. Carried here rather than dropped, because the Studio
    # payload's runtime configuration is the whole point of inspecting a space.
    sdk_type: str | None = None
    sdk_version: str | None = None
    base_image: str | None = None
    hardware: str | None = None
    mcp_support: bool | None = None
    runtime: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.repo_type, str):
            try:
                self.repo_type = RepoType(self.repo_type)
            except ValueError:
                pass
        if isinstance(self.visibility, int) and not isinstance(self.visibility, Visibility):
            try:
                self.visibility = Visibility(self.visibility)
            except ValueError:
                pass
        self.created_at = _coerce_datetime(self.created_at) or self.created_at
        self.last_modified = _coerce_datetime(self.last_modified) or self.last_modified

    def to_dict(self) -> dict:
        """Convert to OpenAPI-compatible dictionary.

        Excludes SDK-internal fields (owner, name, repo_type, visibility)
        and formats datetimes with Z suffix to match OpenAPI spec.
        """
        _INTERNAL_FIELDS = {"owner", "name", "repo_type", "visibility"}
        _OPTIONAL_FIELDS = {
            "display_name",
            "file_size",
            "private",
            "gated",
            "login_required",
            "sdk_type",
            "sdk_version",
            "base_image",
            "hardware",
            "mcp_support",
            "runtime",
        }
        result = {}
        for f in fields(self):  # type: ignore[arg-type]
            if f.name in _INTERNAL_FIELDS:
                continue
            val = getattr(self, f.name)
            if val is None and f.name in _OPTIONAL_FIELDS:
                continue  # skip None optional OpenAPI fields
            if isinstance(val, Enum):
                val = val.value
            elif isinstance(val, datetime):
                # OpenAPI uses Z suffix, not +00:00
                val = val.strftime("%Y-%m-%dT%H:%M:%SZ") if val.tzinfo else val.isoformat()
            elif isinstance(val, list):
                val = list(val)  # shallow copy
            result[f.name] = val
        return result

    @property
    def repo_id(self) -> str | None:
        """Canonical ``owner/name`` identifier, when both parts are known."""
        if self.owner and self.name:
            return f"{self.owner}/{self.name}"
        return None


# ---------------------------------------------------------------------------
# Files & commits
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class FileInfo(_FromDictMixin):
    path: str = ""
    size: int = 0
    blob_id: str | None = None
    type: str = "blob"  # "blob" | "tree"
    last_modified: datetime | str | int | None = None
    lfs: dict[str, Any] | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        self.last_modified = _coerce_datetime(self.last_modified) or self.last_modified

    @property
    def is_dir(self) -> bool:
        return self.type == "tree"

    @property
    def is_lfs(self) -> bool:
        return self.lfs is not None


@dataclass(slots=True)
class CommitInfo(_FromDictMixin):
    sha: str = ""
    message: str = ""
    author: str | None = None
    date: datetime | str | int | None = None

    def __post_init__(self) -> None:
        self.date = _coerce_datetime(self.date) or self.date


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class PagedResult(Generic[T]):
    items: list[T] = field(default_factory=list)
    total_count: int = 0
    page_number: int = 1
    page_size: int = 0
    collection_key: str = field(default="items", repr=False)

    @property
    def has_next(self) -> bool:
        if self.page_size <= 0:
            return False
        return self.page_number * self.page_size < self.total_count

    def to_dict(self) -> dict:
        """Convert to OpenAPI-compatible dictionary.

        Uses collection_key for the items array name (e.g. 'datasets', 'models').
        """
        return {
            self.collection_key: [item.to_dict() if hasattr(item, "to_dict") else item for item in self.items],
            "total_count": self.total_count,
            "page_number": self.page_number,
            "page_size": self.page_size,
        }

    def __iter__(self):  # pragma: no cover - convenience iteration
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class CachedRepoInfo(_FromDictMixin):
    repo_id: str = ""
    repo_type: RepoType | str | None = None
    revision: str | None = None
    size_on_disk: int = 0
    nb_files: int = 0
    last_accessed: datetime | str | int | float | None = None
    local_path: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.repo_type, str):
            try:
                self.repo_type = RepoType(self.repo_type)
            except ValueError:
                pass
        self.last_accessed = _coerce_datetime(self.last_accessed) or self.last_accessed


@dataclass(slots=True)
class CacheInfo:
    repos: list[CachedRepoInfo] = field(default_factory=list)
    total_size: int = 0
    cache_dir: str | None = None

    @property
    def total_repos(self) -> int:
        return len(self.repos)


@dataclass(slots=True)
class VerificationMismatch:
    path: str
    expected: str
    actual: str
    algorithm: str = "sha256"


@dataclass(slots=True)
class CacheVerification:
    """Result of comparing local repository files with Hub checksums."""

    revision: str
    verified_path: str
    checked_count: int = 0
    mismatches: list[VerificationMismatch] = field(default_factory=list)
    missing_paths: list[str] = field(default_factory=list)
    extra_paths: list[str] = field(default_factory=list)
    unverified_paths: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TypedDict payloads (for OpenAPI method signatures)
# ---------------------------------------------------------------------------
class CreateSkillPayload(TypedDict, total=False):
    """Payload for creating a new skill via POST /skills."""

    skill_name: str
    owner: str
    display_name: str
    source_url: str
    private: bool
    description: str
    license: str
    category: str
    tags: list[str]
    logo_url: str
    skill_file: str


class UpdateSkillSettingsPayload(TypedDict, total=False):
    """Payload for updating skill settings via PATCH /skills/{owner}/{skill_name}/settings."""

    display_name: str
    source_url: str
    private: bool
    description: str
    license: str
    category: str
    tags: list[str]
    logo_url: str
    skill_file: str


class CreateStudioPayload(TypedDict, total=False):
    """Payload for creating a new studio via POST /studios."""

    repo_name: str
    owner: str
    display_name: str
    license: str
    visibility: str
    private: bool
    description: str
    cover_image: str
    sdk_type: str
    sdk_version: str
    base_image: str
    hardware: str


class UpdateStudioSettingsPayload(TypedDict, total=False):
    """Payload for updating studio settings via PATCH /studios/{owner}/{repo_name}/settings."""

    display_name: str
    license: str
    visibility: str
    private: bool
    description: str
    cover_image: str
    sdk_type: str
    sdk_version: str
    base_image: str
    hardware: str


class DeployMcpServerPayload(TypedDict, total=False):
    """Payload for deploying an MCP server via POST /mcp/servers/{id}/deploy."""

    transport_type: str
    expiration_minutes: int
    auth_check: bool
    env_info: dict[str, str]


class JWKPayload(TypedDict, total=False):
    """Public or private Ed25519 JSON Web Key wire shape."""

    kty: str
    crv: str
    x: str
    kid: str
    alg: str
    use: str
    d: str


class CreateAgentIdentityPayload(TypedDict, total=False):
    """Payload for POST /agent_ids."""

    agent_name: str
    description: str
    public_key: JWKPayload
    key_alg_type: str
    token_expire_time: int


class UpdateAgentIdentityPayload(TypedDict, total=False):
    """Payload for PATCH /agent_ids/{agent_id}."""

    agent_name: str
    description: str
    token_expire_time: int


class ResetAgentKeyPairPayload(TypedDict, total=False):
    """Payload for PUT /agent_ids/{agent_id}/key_pairs."""

    public_key: JWKPayload
    key_alg_type: str


class PauseAgentPayload(TypedDict):
    """Payload for POST /agent_ids/{agent_id}/paused."""

    paused: bool


class TokenSignPayload(TypedDict):
    """Signed request body for anonymous POST /agent_id/token."""

    agent_id: str
    kid: str
    audience: str
    timestamp: int
    signature: str


__all__ = [
    "AgentIdentity",
    "AgentIdentitySummary",
    "AgentIdConfiguration",
    "AgentJWK",
    "AgentToken",
    "AgentTokenRecord",
    "CacheVerification",
    "CacheInfo",
    "CachedRepoInfo",
    "CommitInfo",
    "CreateAgentIdentityPayload",
    "CreateSkillPayload",
    "CreateStudioPayload",
    "DeployMcpServerPayload",
    "FileInfo",
    "JWKPayload",
    "PagedResult",
    "PauseAgentPayload",
    "RepoInfo",
    "ResetAgentKeyPairPayload",
    "TokenSignPayload",
    "UpdateAgentIdentityPayload",
    "UpdateSkillSettingsPayload",
    "UpdateStudioSettingsPayload",
    "UserInfo",
    "VerificationMismatch",
]
