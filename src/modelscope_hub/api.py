"""High-level public API facade for the ModelScope Hub SDK.

:class:`HubApi` is the **only** entry point users should construct. It composes
the low-level :class:`OpenAPIClient`, :class:`LegacyClient`,
:class:`DownloadManager`, :class:`UploadManager` and the cache helpers into a
unified, OpenAPI-first surface.

Design principles
-----------------
* **OpenAPI-first** — every operation that has an OpenAPI counterpart goes
  through :mod:`._openapi`. Legacy endpoints are used only as a transparent
  fallback when no OpenAPI equivalent exists.
* **Unified repo pattern** — every repository operation accepts a
  ``repo_type`` parameter; there are no type-specific methods like
  ``create_model`` or ``get_dataset``.
* **Transparent fallback** — callers do not need to know which path served
  their request.
* **Lazy clients** — the underlying HTTP clients are instantiated on demand
  so that ``HubApi()`` never fails just because no token is present.
* **SOLID** — :class:`HubApi` only routes and orchestrates; concrete network
  logic lives in the injected dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping

from ._cache_manager import clear_cache as _clear_cache
from ._cache_manager import scan_cache as _scan_cache
from ._download import DownloadManager
from ._legacy_api import LegacyClient
from ._openapi import OpenAPIClient
from ._upload import UploadManager
from .config import HubConfig, get_default_config
from .constants import RepoType, Visibility
from .errors import AuthenticationError, NotFoundError
from .types import CacheInfo, FileInfo, PagedResult, RepoInfo, UserInfo
from .utils.logger import get_logger

__all__ = ["HubApi"]

logger = get_logger("api")

# Type alias for accepted repo_type inputs
RepoTypeLike = "str | RepoType"


# ---------------------------------------------------------------------------
# Routing tables — declarative dispatch keeps :class:`HubApi` free of long
# if/elif chains and makes adding new repo types a one-line change.
# ---------------------------------------------------------------------------
# Repo types that have first-class OpenAPI CRUD coverage today.
_OPENAPI_CREATE_TYPES: frozenset[RepoType] = frozenset({RepoType.STUDIO, RepoType.SKILL})
_OPENAPI_DETAIL_TYPES: frozenset[RepoType] = frozenset(
    {RepoType.MODEL, RepoType.DATASET, RepoType.STUDIO, RepoType.SKILL}
)
# Repo types whose deletion is only available on the legacy surface.
_LEGACY_DELETE_TYPES: frozenset[RepoType] = frozenset({RepoType.MODEL, RepoType.DATASET})


class HubApi:
    """ModelScope Hub SDK — unified API entry point.

    Parameters
    ----------
    config:
        Optional :class:`HubConfig`. When omitted, the process-wide default
        from :func:`get_default_config` is used.
    endpoint:
        Override the API endpoint (takes precedence over ``config.endpoint``).
    token:
        Override the API token (takes precedence over ``config.token``).
    """

    def __init__(
        self,
        config: HubConfig | None = None,
        *,
        endpoint: str | None = None,
        token: str | None = None,
    ) -> None:
        self._config = config or get_default_config()
        if endpoint is not None:
            self._config.endpoint = endpoint.rstrip("/")
        if token is not None:
            self._config.token = token

        # Lazily-instantiated clients — see the cached properties below.
        self._openapi: OpenAPIClient | None = None
        self._legacy: LegacyClient | None = None
        self._downloader: DownloadManager | None = None
        self._uploader: UploadManager | None = None

    # ==================================================================
    # Lazy client accessors
    # ==================================================================
    @property
    def openapi(self) -> OpenAPIClient:
        """Lazily-constructed OpenAPI client."""
        if self._openapi is None:
            self._openapi = OpenAPIClient(self._config)
        return self._openapi

    @property
    def legacy(self) -> LegacyClient:
        """Lazily-constructed legacy ``/api/v1`` client."""
        if self._legacy is None:
            self._legacy = LegacyClient(
                token=self._config.token,
                endpoint=self._config.endpoint,
            )
        return self._legacy

    @property
    def downloader(self) -> DownloadManager:
        if self._downloader is None:
            self._downloader = DownloadManager(self.legacy, self._config)
        return self._downloader

    @property
    def uploader(self) -> UploadManager:
        if self._uploader is None:
            # Inject the OpenAPI client so small files (≤ 5 MiB) flow through
            # ``POST /files/upload`` instead of the legacy commit endpoint.
            self._uploader = UploadManager(self.legacy, self._config, self.openapi)
        return self._uploader

    # ==================================================================
    # Static helpers
    # ==================================================================
    @staticmethod
    def _parse_repo_id(repo_id: str) -> tuple[str, str]:
        """Split a canonical ``owner/name`` identifier into its two halves."""
        if not repo_id or "/" not in repo_id:
            raise ValueError(
                f"Invalid repo_id {repo_id!r}: expected 'owner/name' format."
            )
        owner, _, name = repo_id.partition("/")
        if not owner or not name:
            raise ValueError(
                f"Invalid repo_id {repo_id!r}: owner and name must both be non-empty."
            )
        return owner, name

    @staticmethod
    def _normalize_repo_type(repo_type: RepoTypeLike) -> RepoType:
        """Coerce a ``str`` or :class:`RepoType` value to a :class:`RepoType`."""
        if isinstance(repo_type, RepoType):
            return repo_type
        try:
            return RepoType(str(repo_type).lower())
        except ValueError as exc:
            allowed = ", ".join(t.value for t in RepoType)
            raise ValueError(
                f"Unknown repo_type {repo_type!r}. Expected one of: {allowed}."
            ) from exc

    @staticmethod
    def _normalize_visibility(visibility: int | str | Visibility | None) -> int | None:
        """Normalise visibility input to its integer wire encoding."""
        if visibility is None:
            return None
        if isinstance(visibility, Visibility):
            return int(visibility)
        if isinstance(visibility, int):
            return visibility
        return int(Visibility.from_label(str(visibility)))

    @staticmethod
    def _extract_paged(payload: Any) -> tuple[list[Any], int, int, int]:
        """Decode a paginated OpenAPI response into ``(items, total, page, size)``."""
        if isinstance(payload, list):
            return payload, len(payload), 1, len(payload)
        if not isinstance(payload, dict):
            return [], 0, 1, 0
        for key in ("items", "list", "data", "results"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break
        else:
            items = []
        total = int(payload.get("total_count") or payload.get("total") or len(items))
        page = int(payload.get("page_number") or payload.get("page") or 1)
        size = int(payload.get("page_size") or payload.get("size") or len(items))
        return items, total, page, size

    @staticmethod
    def _repo_info_from_payload(
        data: Mapping[str, Any] | None,
        repo_type: RepoType,
        *,
        owner_hint: str | None = None,
        name_hint: str | None = None,
    ) -> RepoInfo:
        """Build a :class:`RepoInfo` from an arbitrary API payload.

        The legacy and OpenAPI surfaces use different field-naming conventions
        (PascalCase vs snake_case). This helper normalises both into the
        SDK's canonical dataclass.
        """
        data = dict(data or {})
        # PascalCase → snake_case shims for legacy responses.
        normalised: dict[str, Any] = {}
        aliases = {
            "Id": "id",
            "Path": "owner",
            "Name": "name",
            "Owner": "owner",
            "Visibility": "visibility",
            "License": "license",
            "Description": "description",
            "Downloads": "downloads",
            "Likes": "likes",
            "CreatedAt": "created_at",
            "UpdatedAt": "updated_at",
            "Tags": "tags",
        }
        for key, value in data.items():
            normalised[aliases.get(key, key)] = value

        normalised.setdefault("owner", owner_hint)
        normalised.setdefault("name", name_hint)
        normalised["repo_type"] = repo_type
        return RepoInfo.from_dict(normalised)

    # ==================================================================
    # Authentication
    # ==================================================================
    def login(self, token: str) -> UserInfo:
        """Persist ``token`` and return the authenticated user profile.

        We persist the token locally (so future sessions pick it up) and then
        call ``GET /users/me`` through the OpenAPI surface to confirm the
        token works.
        """
        if not token or not token.strip():
            raise ValueError("token must be a non-empty string")

        token = token.strip()
        self._config.save_token(token)
        # Reset cached clients so they observe the new credential.
        self._openapi = None
        if self._legacy is not None:
            self._legacy.token = token

        try:
            return self.whoami()
        except AuthenticationError as exc:
            # Clean up the bad token to avoid trapping the user in a loop.
            self._config.clear_token()
            raise AuthenticationError(
                "Login failed: the provided token was rejected by the server.",
                status_code=getattr(exc, "status_code", None),
            ) from exc

    def logout(self) -> None:
        """Clear the locally persisted token."""
        self._config.clear_token()
        self._openapi = None
        if self._legacy is not None:
            self._legacy.token = None

    def whoami(self) -> UserInfo:
        """Return the profile for the currently authenticated user."""
        payload = self.openapi.get_current_user()
        return UserInfo.from_dict(payload if isinstance(payload, dict) else {})

    # ==================================================================
    # Unified repo CRUD
    # ==================================================================
    def create_repo(
        self,
        repo_id: str,
        repo_type: RepoTypeLike,
        *,
        visibility: int | str | Visibility | None = None,
        license: str | None = None,
        chinese_name: str | None = None,
        description: str | None = None,
        **extra: Any,
    ) -> RepoInfo:
        """Create a new repository.

        * ``studio`` / ``skill`` → OpenAPI ``POST /studios`` / ``POST /skills``
        * ``model`` / ``dataset`` → legacy ``POST /api/v1/{type}s``
        """
        rt = self._normalize_repo_type(repo_type)
        owner, name = self._parse_repo_id(repo_id)
        vis = self._normalize_visibility(visibility)

        if rt in _OPENAPI_CREATE_TYPES:
            payload: dict[str, Any] = {
                "owner": owner,
                "name": name,
            }
            if vis is not None:
                payload["visibility"] = vis
            if license is not None:
                payload["license"] = license
            if chinese_name is not None:
                payload["chinese_name"] = chinese_name
            if description is not None:
                payload["description"] = description
            payload.update(extra)
            data = (
                self.openapi.create_studio(payload)
                if rt is RepoType.STUDIO
                else self.openapi.create_skill(payload)
            )
            return self._repo_info_from_payload(
                data, rt, owner_hint=owner, name_hint=name
            )

        # Fallback to the legacy surface for model/dataset (and anything else
        # that exposes the same shape).
        legacy_kwargs: dict[str, Any] = {}
        if vis is not None:
            legacy_kwargs["Visibility"] = vis
        if license is not None:
            legacy_kwargs["License"] = license
        if chinese_name is not None:
            legacy_kwargs["ChineseName"] = chinese_name
        if description is not None:
            legacy_kwargs["Description"] = description
        legacy_kwargs.update(extra)

        data = self.legacy.create_repo(
            repo_id=repo_id,
            repo_type=str(rt),
            visibility=vis if vis is not None else int(Visibility.PUBLIC),
            license=license or "Apache-2.0",
            **legacy_kwargs,
        )
        return self._repo_info_from_payload(
            data, rt, owner_hint=owner, name_hint=name
        )

    def get_repo(
        self,
        repo_id: str,
        repo_type: RepoTypeLike,
        *,
        revision: str | None = None,  # noqa: ARG002 - reserved for future use
    ) -> RepoInfo:
        """Fetch a repository's metadata via the OpenAPI surface."""
        rt = self._normalize_repo_type(repo_type)
        owner, name = self._parse_repo_id(repo_id)

        if rt is RepoType.MODEL:
            data = self.openapi.get_model(owner, name)
        elif rt is RepoType.DATASET:
            data = self.openapi.get_dataset(owner, name)
        elif rt is RepoType.STUDIO:
            data = self.openapi.get_studio(owner, name)
        elif rt is RepoType.SKILL:
            # Skills are addressed by id on read; fall back to listing if the
            # id form is not the canonical owner/name pair.
            data = self.openapi.get_skill(f"{owner}/{name}")
        elif rt is RepoType.MCP:
            data = self.openapi.get_mcp_server(f"{owner}/{name}")
        else:  # pragma: no cover - defensive
            raise NotImplementedError(f"get_repo not supported for {rt}")

        return self._repo_info_from_payload(
            data, rt, owner_hint=owner, name_hint=name
        )

    def list_repos(
        self,
        repo_type: RepoTypeLike,
        *,
        owner: str | None = None,
        search: str | None = None,
        sort: str | None = None,
        page_number: int = 1,
        page_size: int = 10,
        **filters: Any,
    ) -> PagedResult[RepoInfo]:
        """List repositories of the given type via OpenAPI."""
        rt = self._normalize_repo_type(repo_type)
        # Build a clean filter mapping (drop ``None`` values).
        clean_filters: dict[str, Any] = {k: v for k, v in filters.items() if v is not None}

        if rt is RepoType.MODEL:
            payload = self.openapi.list_models(
                search=search, owner=owner, sort=sort,
                page_number=page_number, page_size=page_size,
                filters=clean_filters or None,
            )
        elif rt is RepoType.DATASET:
            payload = self.openapi.list_datasets(
                search=search, owner=owner, sort=sort,
                page_number=page_number, page_size=page_size,
                filters=clean_filters or None,
            )
        elif rt is RepoType.SKILL:
            if owner:
                clean_filters.setdefault("owner", owner)
            payload = self.openapi.list_skills(
                search=search,
                page_number=page_number, page_size=page_size,
                filters=clean_filters or None,
            )
        elif rt is RepoType.MCP:
            payload = self.openapi.list_mcp_servers(
                search=search,
                page_number=page_number, page_size=page_size,
                extra=clean_filters or None,
            )
        elif rt is RepoType.STUDIO:
            # Studios have no list endpoint on the OpenAPI surface today;
            # exposing an empty page keeps the type contract uniform.
            raise NotImplementedError(
                "Listing studios is not supported by the OpenAPI surface yet."
            )
        else:  # pragma: no cover - defensive
            raise NotImplementedError(f"list_repos not supported for {rt}")

        items, total, page, size = self._extract_paged(payload)
        infos = [self._repo_info_from_payload(item, rt) for item in items]
        return PagedResult(items=infos, total_count=total, page_number=page, page_size=size)

    def delete_repo(self, repo_id: str, repo_type: RepoTypeLike) -> None:
        """Delete a repository.

        Currently only ``model`` and ``dataset`` are deletable via the legacy
        surface; other types raise :class:`NotImplementedError`.
        """
        rt = self._normalize_repo_type(repo_type)
        if rt not in _LEGACY_DELETE_TYPES:
            raise NotImplementedError(
                f"Deletion is not supported for repo_type={rt.value!r}."
            )
        # Validate the repo_id shape before issuing the network call.
        self._parse_repo_id(repo_id)
        self.legacy.delete_repo(repo_id=repo_id, repo_type=str(rt))

    def repo_exists(self, repo_id: str, repo_type: RepoTypeLike) -> bool:
        """Return ``True`` iff the repository exists and is visible to the caller."""
        try:
            self.get_repo(repo_id, repo_type)
            return True
        except NotFoundError:
            return False

    # ==================================================================
    # Files
    # ==================================================================
    def upload_file(
        self,
        repo_id: str,
        repo_type: RepoTypeLike,
        path_or_fileobj: str | Path | bytes | BinaryIO,
        path_in_repo: str,
        *,
        commit_message: str | None = None,
        revision: str | None = None,
    ) -> dict:
        """Upload a single file. Routes via :class:`UploadManager`."""
        rt = self._normalize_repo_type(repo_type)
        return self.uploader.upload_file(
            repo_id=repo_id,
            repo_type=str(rt),
            path_or_fileobj=path_or_fileobj,
            path_in_repo=path_in_repo,
            commit_message=commit_message or "Upload file",
            revision=revision or "master",
        )

    def upload_folder(
        self,
        repo_id: str,
        repo_type: RepoTypeLike,
        folder_path: str | Path,
        *,
        path_in_repo: str = "",
        commit_message: str | None = None,
        revision: str | None = None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        max_workers: int = 4,
    ) -> dict:
        """Upload a whole folder. Routes via :class:`UploadManager`."""
        rt = self._normalize_repo_type(repo_type)
        return self.uploader.upload_folder(
            repo_id=repo_id,
            repo_type=str(rt),
            folder_path=folder_path,
            path_in_repo=path_in_repo,
            commit_message=commit_message or "Upload folder",
            revision=revision or "master",
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=max_workers,
        )

    def download_file(
        self,
        repo_id: str,
        repo_type: RepoTypeLike,
        file_path: str,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        force: bool = False,
    ) -> Path:
        """Download a single file. Routes via :class:`DownloadManager`."""
        rt = self._normalize_repo_type(repo_type)
        return self.downloader.download_file(
            repo_id=repo_id,
            repo_type=str(rt),
            file_path=file_path,
            revision=revision or "master",
            cache_dir=Path(cache_dir) if cache_dir else None,
            force=force,
        )

    def download_repo(
        self,
        repo_id: str,
        repo_type: RepoTypeLike,
        *,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        max_workers: int = 4,
    ) -> Path:
        """Download an entire repo snapshot. Routes via :class:`DownloadManager`."""
        rt = self._normalize_repo_type(repo_type)
        return self.downloader.download_repo(
            repo_id=repo_id,
            repo_type=str(rt),
            revision=revision or "master",
            cache_dir=Path(cache_dir) if cache_dir else None,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=max_workers,
        )

    def list_repo_files(
        self,
        repo_id: str,
        repo_type: RepoTypeLike,
        *,
        revision: str | None = None,
        recursive: bool = True,
    ) -> list[FileInfo]:
        """List files in a repository (legacy — no OpenAPI equivalent)."""
        rt = self._normalize_repo_type(repo_type)
        raw = self.legacy.list_repo_files(
            repo_id=repo_id,
            repo_type=str(rt),
            revision=revision or "master",
            recursive=recursive,
        )
        files: list[FileInfo] = []
        for item in raw:
            normalised = {
                "path": item.get("Path") or item.get("path") or item.get("Name") or "",
                "size": int(item.get("Size") or item.get("size") or 0),
                "blob_id": item.get("BlobId") or item.get("blob_id") or item.get("Sha256"),
                "type": item.get("Type") or item.get("type") or "blob",
                "last_modified": item.get("CommittedDate") or item.get("last_modified"),
                "lfs": item.get("Lfs") or item.get("lfs"),
            }
            files.append(FileInfo.from_dict(normalised))
        return files

    def delete_files(
        self,
        repo_id: str,
        repo_type: RepoTypeLike,
        file_paths: Iterable[str],
        *,
        commit_message: str | None = None,
        revision: str | None = None,
    ) -> dict:
        """Delete one or more files via a legacy commit operation."""
        rt = self._normalize_repo_type(repo_type)
        operations = [
            {"action": "delete", "file_path": p} for p in file_paths if p
        ]
        if not operations:
            raise ValueError("file_paths must contain at least one non-empty path.")
        return self.legacy.create_commit(
            repo_id=repo_id,
            repo_type=str(rt),
            operations=operations,
            commit_message=commit_message or f"Delete {len(operations)} file(s)",
            revision=revision or "master",
        )

    # ==================================================================
    # Versioning
    # ==================================================================
    def list_repo_revisions(
        self, repo_id: str, repo_type: RepoTypeLike
    ) -> list[dict]:
        """Return branches and tags of a repository (legacy)."""
        rt = self._normalize_repo_type(repo_type)
        return self.legacy.list_revisions(repo_id=repo_id, repo_type=str(rt))

    def create_repo_tag(
        self,
        repo_id: str,
        repo_type: RepoTypeLike,
        tag: str,
        *,
        revision: str | None = None,
    ) -> dict:
        """Create a tag pointing at ``revision`` (defaults to ``master``)."""
        rt = self._normalize_repo_type(repo_type)
        return self.legacy.create_tag(
            repo_id=repo_id,
            repo_type=str(rt),
            tag=tag,
            revision=revision or "master",
        )

    # ==================================================================
    # Lifecycle (Studio / MCP)
    # ==================================================================
    def deploy_repo(
        self,
        repo_id: str,
        repo_type: RepoTypeLike = RepoType.STUDIO,
        *,
        payload: Mapping[str, Any] | None = None,
    ) -> dict:
        """Deploy a Studio space or an MCP server."""
        rt = self._normalize_repo_type(repo_type)
        if rt is RepoType.STUDIO:
            owner, name = self._parse_repo_id(repo_id)
            return self.openapi.deploy_studio(owner, name, payload)
        if rt is RepoType.MCP:
            return self.openapi.deploy_mcp_server(repo_id, payload)
        raise NotImplementedError(
            f"deploy_repo is not supported for repo_type={rt.value!r}."
        )

    def stop_repo(
        self,
        repo_id: str,
        repo_type: RepoTypeLike = RepoType.STUDIO,
    ) -> dict:
        """Stop a running Studio or undeploy an MCP server."""
        rt = self._normalize_repo_type(repo_type)
        if rt is RepoType.STUDIO:
            owner, name = self._parse_repo_id(repo_id)
            return self.openapi.stop_studio(owner, name)
        if rt is RepoType.MCP:
            return self.openapi.undeploy_mcp_server(repo_id)
        raise NotImplementedError(
            f"stop_repo is not supported for repo_type={rt.value!r}."
        )

    def get_repo_logs(
        self,
        repo_id: str,
        repo_type: RepoTypeLike = RepoType.STUDIO,
        *,
        log_type: str = "runtime",
        page_num: int = 1,
        page_size: int = 20,
        keyword: str | None = None,
    ) -> dict:
        """Fetch paginated runtime/build logs for a Studio space."""
        rt = self._normalize_repo_type(repo_type)
        if rt is not RepoType.STUDIO:
            raise NotImplementedError(
                f"get_repo_logs is currently only supported for studio (got {rt.value!r})."
            )
        owner, name = self._parse_repo_id(repo_id)
        return self.openapi.get_studio_logs(
            owner, name, log_type,
            page_num=page_num, page_size=page_size, keyword=keyword,
        )

    def update_repo_settings(
        self,
        repo_id: str,
        repo_type: RepoTypeLike,
        **settings: Any,
    ) -> dict:
        """Update repo settings (Studio or Skill)."""
        rt = self._normalize_repo_type(repo_type)
        owner, name = self._parse_repo_id(repo_id)
        if rt is RepoType.STUDIO:
            return self.openapi.update_studio_settings(owner, name, settings)
        if rt is RepoType.SKILL:
            return self.openapi.update_skill_settings(owner, name, settings)
        raise NotImplementedError(
            f"update_repo_settings is not supported for repo_type={rt.value!r}."
        )

    # ==================================================================
    # Secrets
    # ==================================================================
    def list_secrets(
        self, repo_id: str, repo_type: RepoTypeLike = RepoType.STUDIO
    ) -> list[dict]:
        rt = self._normalize_repo_type(repo_type)
        if rt is not RepoType.STUDIO:
            raise NotImplementedError(
                f"Secret management is only supported for studio (got {rt.value!r})."
            )
        owner, name = self._parse_repo_id(repo_id)
        data = self.openapi.list_studio_secrets(owner, name)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("items", "secrets", "list"):
                if isinstance(data.get(key), list):
                    return data[key]
        return []

    def add_secret(
        self,
        repo_id: str,
        key: str,
        value: str,
        repo_type: RepoTypeLike = RepoType.STUDIO,
    ) -> dict:
        rt = self._normalize_repo_type(repo_type)
        if rt is not RepoType.STUDIO:
            raise NotImplementedError("Only studio secrets are supported.")
        owner, name = self._parse_repo_id(repo_id)
        return self.openapi.add_studio_secret(owner, name, key, value)

    def update_secret(
        self,
        repo_id: str,
        key: str,
        value: str,
        repo_type: RepoTypeLike = RepoType.STUDIO,
    ) -> dict:
        rt = self._normalize_repo_type(repo_type)
        if rt is not RepoType.STUDIO:
            raise NotImplementedError("Only studio secrets are supported.")
        owner, name = self._parse_repo_id(repo_id)
        return self.openapi.update_studio_secret(owner, name, key, value)

    def delete_secret(
        self,
        repo_id: str,
        key: str,
        repo_type: RepoTypeLike = RepoType.STUDIO,
    ) -> dict:
        rt = self._normalize_repo_type(repo_type)
        if rt is not RepoType.STUDIO:
            raise NotImplementedError("Only studio secrets are supported.")
        owner, name = self._parse_repo_id(repo_id)
        return self.openapi.delete_studio_secret(owner, name, key)

    # ==================================================================
    # MCP convenience wrappers
    # ==================================================================
    def list_mcp_servers(
        self,
        *,
        search: str | None = None,
        page_number: int = 1,
        page_size: int = 10,
        **extra: Any,
    ) -> PagedResult[dict]:
        """List MCP servers (OpenAPI)."""
        payload = self.openapi.list_mcp_servers(
            search=search,
            page_number=page_number,
            page_size=page_size,
            extra={k: v for k, v in extra.items() if v is not None} or None,
        )
        items, total, page, size = self._extract_paged(payload)
        return PagedResult(items=list(items), total_count=total, page_number=page, page_size=size)

    def get_mcp_server(self, server_id: str) -> dict:
        return self.openapi.get_mcp_server(server_id)

    def deploy_mcp_server(
        self, server_id: str, *, payload: Mapping[str, Any] | None = None
    ) -> dict:
        return self.openapi.deploy_mcp_server(server_id, payload)

    def undeploy_mcp_server(self, server_id: str) -> dict:
        return self.openapi.undeploy_mcp_server(server_id)

    # ==================================================================
    # Cache
    # ==================================================================
    def scan_cache(self, cache_dir: str | Path | None = None) -> CacheInfo:
        """Inspect the local cache directory."""
        return _scan_cache(Path(cache_dir) if cache_dir else None)

    def clear_cache(
        self,
        *,
        cache_dir: str | Path | None = None,
        repo_type: RepoTypeLike | None = None,
        repo_id: str | None = None,
    ) -> int:
        """Remove cached data. Returns the number of bytes freed."""
        rt_value: str | None = None
        if repo_type is not None:
            rt_value = str(self._normalize_repo_type(repo_type))
        return _clear_cache(
            cache_dir=Path(cache_dir) if cache_dir else None,
            repo_type=rt_value,
            repo_id=repo_id,
        )
