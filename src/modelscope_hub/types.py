"""Typed data containers returned by the SDK.

Every dataclass is constructible from a raw API payload via :func:`from_dict`,
which silently ignores fields the server may add in the future. This keeps the
client forward-compatible while still benefiting from static typing.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Generic, TextIO, TypedDict, TypeVar

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


@dataclass(frozen=True, slots=True)
class DownloadRetryPolicy:
    """Short recovery limits for report-mode downloads, shared by all file parts.

    Healthy transfers keep the regular total file timeout. Only failed-request
    inactivity and retry waits consume recovery_timeout. Longer Retry-After
    values defer the file instead of occupying a worker or retrying early.
    """

    max_retries: int = 2
    recovery_timeout: float = 60
    idle_timeout: float = 15
    low_speed_window: float = 10
    max_retry_delay: float = 5

    def __post_init__(self) -> None:
        if self.max_retries < 0 or any(
            value <= 0
            for value in (
                self.recovery_timeout,
                self.idle_timeout,
                self.low_speed_window,
                self.max_retry_delay,
            )
        ):
            raise ValueError("Download recovery limits must be positive; max_retries may be zero")


@dataclass(slots=True)
class DownloadFileResult(_FromDictMixin):
    """A file validation result or an auxiliary recovery hint."""

    path: str = ""
    status: str = "PENDING"
    size: int | None = None
    sha256: str | None = None
    retries: int = 0
    recovery_seconds: float = 0
    error: str | None = None
    error_type: str | None = None
    retryable: bool = False


class _DownloadReportResults:
    """Own an anonymous result spool without adding it to dataclass serialization."""

    _file_results: TextIO | None = None

    def _attach_file_results(self, stream: TextIO) -> None:
        self._file_results = stream
        self._file_results_lock = threading.Lock()

    def close(self) -> None:
        """Release this report's temporary per-file results after consumption."""
        if self._file_results is not None:
            with self._file_results_lock:
                if self._file_results is not None:
                    self._file_results.close()
                    self._file_results = None

    def _iter_file_results(self) -> Iterator[dict]:
        if self._file_results is None:
            raise ValueError(
                "Per-file results are available only on an open live DownloadReport; "
                "the persisted JSON contains summary and recovery hints, not a verified-file manifest. "
                "Rerun download_repo_report to validate actual files."
            )
        offset = 0
        while True:
            with self._file_results_lock:
                if self._file_results is None:
                    raise ValueError("DownloadReport results have been closed")
                self._file_results.seek(offset)
                line = self._file_results.readline()
                offset = self._file_results.tell()
            if not line:
                break
            yield json.loads(line)


@dataclass(slots=True)
class DownloadReport(_DownloadReportResults, _FromDictMixin):
    """Report-mode completion is explicit: COMPLETE, PARTIAL, or CANCELLED.

    Counts cover this invocation's selected files, not every previous batch.
    Per-file results use an anonymous temporary spool owned by this live object,
    preserving earlier calls without permanent manifests or unbounded memory.
    One hidden JSON holds the latest summary and sparse recovery hints only.
    VERIFIED describes validation during this call, not a permanent guarantee:
    downstream users must check changed files. close() releases the spool;
    state_warnings reports unavailable or ignored auxiliary recovery hints.
    """

    repo_id: str = ""
    revision: str = ""
    output_dir: str = ""
    tracker_path: str = ""
    report_path: str = ""
    results_path: str = ""
    status: str = "PARTIAL"
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    deferred: int = 0
    pending: int = 0
    skipped_failures: int = 0
    listing_complete: bool = True
    listing_errors: list[str] = field(default_factory=list)
    global_error: str | None = None
    state_warnings: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return self.status == "COMPLETE"

    def iter_files(self, status: str | None = None) -> Iterator[DownloadFileResult]:
        """Stream this live call's results; persisted summaries cannot prove success."""
        for raw in self._iter_file_results():
            result = DownloadFileResult.from_dict(raw)
            if status is None or result.status == status:
                yield result


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
    # Zone membership. Reported by the model/dataset detail endpoints; ``None``
    # for repo kinds that have no zones (studio / skill / mcp) and for payloads
    # predating the field.
    model_category: str | None = None
    dataset_category: str | None = None
    # Approval mode of a gated repository, returned only when ``gated`` is true.
    gated_mode: str | None = None

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
            "model_category",
            "dataset_category",
            "gated_mode",
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


# ---------------------------------------------------------------------------
# Model / dataset zone structures
#
# Shapes shared by the category endpoints and the enum option endpoints. Field
# names follow the OpenAPI schemas exactly: the server rejects unknown keys, so
# a convenience alias here would turn into a 400 at the call site.
# ---------------------------------------------------------------------------
class SampleImage(TypedDict, total=False):
    """A sample image attached to an AIGC model version.

    ``url`` is required. The schema accepts only these three keys -- notably
    there is no ``text`` field.
    """

    url: str
    prompt: str
    negative_prompt: str


class WeightFileReference(TypedDict):
    """Reference to a weight file already uploaded to the repository.

    Only the repo-relative path is sent; the server resolves size and digest
    itself. The suffix must be one of :data:`WEIGHT_FILE_SUFFIXES`.
    """

    path_in_repo: str


class AigcContentCategories(TypedDict, total=False):
    """Three-level content classification of an AIGC model."""

    style: str
    function: str
    element: str


class ModelOptionItem(TypedDict):
    """One selectable value returned by an ``*/options`` endpoint.

    Submit ``id``; ``label`` is for display only. ``type`` is self-describing
    (``task``, ``license``, ``base_architecture``, ``specialization``, ...).
    """

    type: str
    id: str
    label: str


class BaseArchitectureOption(TypedDict):
    """A first-level AIGC base architecture and its second-level subtypes."""

    type: str
    id: str
    label: str
    subtypes: list[ModelOptionItem]


class ScienceSubjectOption(TypedDict):
    """A first-level science subject area and its specializations."""

    type: str
    id: str
    label: str
    specializations: list[ModelOptionItem]


# ---------------------------------------------------------------------------
# Model / dataset request payloads
#
# Each mirrors one OpenAPI request schema exactly. Fields the schema omits are
# omitted here too, because the endpoints reject unknown keys with a 400 --
# notably ``base_model_id`` (the base model is derived from the architecture
# subtype), ``login_required`` (read-only), and the category discriminator on
# the PATCH payloads (the server reads the current category from its own
# database and rejects the field).
# ---------------------------------------------------------------------------
class CreateModelPayload(TypedDict, total=False):
    """Payload for ``POST /models``. ``owner`` and ``repo_name`` are required.

    Carries general metadata only; a newly created model is ``base``. Zone
    fields are set afterwards via ``PUT /models/{owner}/{repo_name}/category``.
    """

    owner: str
    repo_name: str
    display_name: str
    license: str
    license_name: str
    license_link: str
    visibility: str
    gated_mode: str
    description: str
    # The schema declares ``string | null``: an explicit null means "not set".
    task: str | None


class CreateDatasetPayload(TypedDict, total=False):
    """Payload for ``POST /datasets``. ``owner`` and ``repo_name`` are required.

    Mirrors :class:`CreateModelPayload` without ``task``, which the dataset
    schema does not declare.
    """

    owner: str
    repo_name: str
    display_name: str
    license: str
    license_name: str
    license_link: str
    visibility: str
    gated_mode: str
    description: str


class SetModelCategoryPayload(TypedDict, total=False):
    """Payload for ``PUT /models/{owner}/{repo_name}/category``.

    ``model_category`` is required and names the *target* zone; which other
    fields are required follows from it, not from the model's current zone:

    * ``base`` -- nothing else; every zone field is cleared.
    * ``aigc`` -- ``aigc_type``, both architecture levels, ``version_name``, and
      non-empty ``weight_files`` and ``sample_images``. Creates a new version.
    * ``science`` -- ``primary_subject_area`` and ``secondary_specialization``;
      ``cover_image`` is optional.

    Passing a field that does not apply to the target zone returns 400.
    """

    model_category: str
    # AIGC
    aigc_type: str
    base_model_architecture_type: str
    base_model_architecture_subtype: str
    content_categories: AigcContentCategories
    version_name: str
    version_description: str
    trigger_words: list[str]
    sample_images: list[SampleImage]
    weight_files: list[WeightFileReference]
    # Science
    primary_subject_area: str
    secondary_specialization: str
    cover_image: str


class UpdateModelCategoryPayload(TypedDict, total=False):
    """Payload for ``PATCH /models/{owner}/{repo_name}/category``.

    Updates zone metadata in place. The server determines the zone itself, so
    ``model_category`` is not accepted; use ``PUT`` to change zones.

    For AIGC, ``version_id`` (the numeric id from a detail response's
    ``versions[]``) is required to update ``trigger_words``,
    ``version_description`` or ``sample_images``. Science updates must not send
    it. An existing version's weights are immutable: ``weight_files``,
    ``version_name`` and ``base_model_id`` are rejected -- create a new version
    with ``PUT`` instead. Omitted fields are unchanged; explicit nulls clear.
    """

    version_id: int
    # AIGC
    aigc_type: str | None
    base_model_architecture_type: str | None
    base_model_architecture_subtype: str | None
    content_categories: AigcContentCategories | None
    trigger_words: list[str] | None
    version_description: str | None
    sample_images: list[SampleImage]
    # Science
    primary_subject_area: str | None
    secondary_specialization: str | None
    cover_image: str | None


class SetDatasetCategoryPayload(TypedDict, total=False):
    """Payload for ``PUT /datasets/{owner}/{repo_name}/category``.

    ``dataset_category`` is required and is ``base`` or ``science`` -- the
    general value is ``base``, not ``basic``. ``science`` requires a matching
    subject area and specialization; ``base`` clears the science fields.
    """

    dataset_category: str
    primary_subject_area: str | None
    secondary_specialization: str | None
    cover_image: str | None


class UpdateDatasetCategoryPayload(TypedDict, total=False):
    """Payload for ``PATCH /datasets/{owner}/{repo_name}/category``.

    ``dataset_category`` is not accepted -- the server reads the current one and
    returns 400 if the field is present; use ``PUT`` to change zones.
    """

    primary_subject_area: str | None
    secondary_specialization: str | None
    cover_image: str | None


class UpdateDatasetSettingsPayload(TypedDict, total=False):
    """Payload for ``PATCH /datasets/{owner}/{repo_name}/settings``.

    General settings only; zone fields are rejected. ``login_required`` is
    deliberately absent -- the schema does not accept it, the field is only ever
    reported back by the detail endpoint.

    Omitted fields are unchanged; explicit nulls clear. ``gated_mode`` is valid
    only alongside ``visibility="gated"`` (or on a repo that is already gated).
    """

    display_name: str | None
    license: str | None
    visibility: str | None
    gated_mode: str
    description: str | None


class UpdateModelSettingsPayload(TypedDict, total=False):
    """Payload for ``PATCH /models/{owner}/{repo_name}/settings``.

    General settings only; zone fields are rejected. Mirrors
    :class:`UpdateDatasetSettingsPayload` plus ``task``.

    Omitted fields are unchanged; explicit nulls clear. ``gated_mode`` is valid
    only alongside ``visibility="gated"`` (or on a repo that is already gated).
    """

    task: str | None
    display_name: str | None
    license: str | None
    visibility: str | None
    gated_mode: str
    description: str | None


class DeleteFilesPayload(TypedDict, total=False):
    """Payload for ``DELETE /{models,datasets}/{owner}/{repo_name}/files``.

    ``file_paths`` is required: 1 to 100 distinct repo-relative file paths. No
    directories and no traversal. The whole request is one commit and is
    all-or-nothing -- a single unresolvable path fails the lot.

    ``revision`` must name a branch, not a tag.
    """

    file_paths: list[str]
    revision: str
    commit_message: str


__all__ = [
    "AigcContentCategories",
    "BaseArchitectureOption",
    "CreateDatasetPayload",
    "CreateModelPayload",
    "DeleteFilesPayload",
    "ModelOptionItem",
    "SampleImage",
    "ScienceSubjectOption",
    "SetDatasetCategoryPayload",
    "SetModelCategoryPayload",
    "UpdateDatasetCategoryPayload",
    "UpdateDatasetSettingsPayload",
    "UpdateModelCategoryPayload",
    "UpdateModelSettingsPayload",
    "WeightFileReference",
    "DownloadRetryPolicy",
    "DownloadFileResult",
    "DownloadReport",
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
