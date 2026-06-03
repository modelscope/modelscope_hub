"""Low-level client for the ModelScope OpenAPI v1 surface.

This module is *internal*: it is consumed by :mod:`modelscope_hub.api` to build
the ergonomic public façade. Direct use is discouraged and not subject to the
SDK's stability guarantees.

Design goals
------------
* A single :meth:`OpenAPIClient._request` chokepoint owns transport concerns —
  URL composition, authentication injection, retry/back-off, error decoding
  and ``data`` envelope unwrapping.
* Each of the 25 OpenAPI endpoints maps to exactly one method, named after the
  resource it manipulates and grouped by section comments.
* Filter-style query parameters (``filter.task=...`` etc.) are accepted as a
  flat ``filters`` mapping and serialised transparently.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping
from urllib.parse import urljoin

import requests

from .config import HubConfig, get_default_config
from .constants import API_MAX_RETRIES, API_TIMEOUT, OPENAPI_PREFIX
from .errors import (
    AuthenticationError,
    NetworkError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    raise_for_status,
)
from .types import (
    CreateSkillPayload,
    CreateStudioPayload,
    DeployMcpServerPayload,
    UpdateSkillSettingsPayload,
    UpdateStudioSettingsPayload,
)
from .utils.logger import get_logger

__all__ = ["OpenAPIClient"]

_logger = get_logger("openapi")

# HTTP methods that are safe to retry without risking duplicate side-effects.
_IDEMPOTENT_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})

# Errors that warrant a transparent retry.
_RETRYABLE_EXC: tuple[type[BaseException], ...] = (
    NetworkError, ServerError, RateLimitError,
)

JSON = dict[str, Any]
QueryParams = list[tuple[str, str]]
Filters = Mapping[str, str | int | float | bool] | None


class OpenAPIClient:
    """Thin, typed wrapper around the public ``/openapi/v1`` endpoints.

    Parameters
    ----------
    config:
        Optional :class:`HubConfig` instance. When omitted the process-wide
        default returned by :func:`get_default_config` is used.
    session:
        Optional pre-configured :class:`requests.Session`. Useful for tests
        and for sharing a connection pool across multiple clients.
    timeout:
        Per-request timeout in seconds. Defaults to :data:`API_TIMEOUT`.
    max_retries:
        Maximum number of retry attempts for transient failures.
    """

    def __init__(
        self,
        config: HubConfig | None = None,
        *,
        session: requests.Session | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._config = config or get_default_config()
        self._session = session or requests.Session()
        self._timeout = float(timeout) if timeout is not None else float(API_TIMEOUT)
        self._max_retries = int(max_retries) if max_retries is not None else int(API_MAX_RETRIES)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> "OpenAPIClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Request plumbing
    # ------------------------------------------------------------------
    @property
    def base_url(self) -> str:
        """Fully-qualified OpenAPI base URL, including trailing slash."""
        return f"{self._config.endpoint.rstrip('/')}{OPENAPI_PREFIX}/"

    def _url(self, path: str) -> str:
        # ``urljoin`` treats absolute leading slashes as roots, which would
        # discard the ``/openapi/v1`` prefix. Normalise to a relative form.
        return urljoin(self.base_url, path.lstrip("/"))

    def _auth_headers(self, *, require_token: bool = False) -> dict[str, str]:
        token = self._config.token
        if not token:
            if require_token:
                raise AuthenticationError(
                    "Missing API token. Call HubApi.login(...) or set MODELSCOPE_API_TOKEN."
                )
            return {}
        return {"Authorization": f"Bearer {token}"}

    @staticmethod
    def _flatten_filters(filters: Filters) -> QueryParams:
        """Serialise a flat mapping into ``filter.key=value`` tuples."""
        if not filters:
            return []
        flat: QueryParams = []
        for key, value in filters.items():
            if value is None or value == "":
                continue
            flat.append((f"filter.{key}", str(value)))
        return flat

    @staticmethod
    def _merge_params(
        params: Mapping[str, Any] | None,
        filters: Filters = None,
    ) -> QueryParams | None:
        """Combine plain query params with filter-prefixed ones, dropping ``None``."""
        merged: QueryParams = []
        if params:
            for key, value in params.items():
                if value is None:
                    continue
                if isinstance(value, (list, tuple, set)):
                    merged.extend((key, str(item)) for item in value if item is not None)
                else:
                    merged.append((key, str(value)))
        merged.extend(OpenAPIClient._flatten_filters(filters))
        return merged or None

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | QueryParams | None = None,
        json_body: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        headers: Mapping[str, str] | None = None,
        require_token: bool = True,
        unwrap: bool = True,
        timeout: float | None = None,
    ) -> Any:
        """Execute an HTTP request and return the unwrapped ``data`` payload.

        The method centralises authentication, retries on transient errors,
        and decoding of the standard ``{"success": ..., "data": ...}`` envelope.
        """
        url = self._url(path)
        merged_headers = dict(self._auth_headers(require_token=require_token))
        if headers:
            merged_headers.update(headers)

        method_upper = method.upper()
        attempts = max(1, self._max_retries)
        last_exc: BaseException | None = None

        for attempt in range(1, attempts + 1):
            _logger.debug("%s %s", method_upper, url)
            try:
                response = self._session.request(
                    method=method_upper,
                    url=url,
                    params=params,
                    json=json_body,
                    data=data,
                    files=files,
                    headers=merged_headers,
                    timeout=timeout if timeout is not None else self._timeout,
                )
            except requests.Timeout as exc:
                last_exc = RequestTimeoutError(f"Request timed out: {exc}")
            except requests.ConnectionError as exc:
                last_exc = NetworkError(f"Connection error: {exc}")
            except requests.RequestException as exc:  # pragma: no cover - defensive
                last_exc = NetworkError(f"Request failed: {exc}")
            else:
                _logger.debug("%s %s -> %s", method_upper, url, response.status_code)
                try:
                    raise_for_status(response)
                except _RETRYABLE_EXC as exc:  # type: ignore[misc]
                    last_exc = exc
                else:
                    return self._decode(response, unwrap=unwrap)

            # Retry policy: only for idempotent methods on transient errors.
            if attempt >= attempts or method_upper not in _IDEMPOTENT_METHODS:
                break
            backoff = min(2 ** (attempt - 1), 16)
            _logger.debug(
                "Retrying %s %s after %s (attempt %d/%d)",
                method_upper, path, last_exc, attempt, attempts,
            )
            time.sleep(backoff)

        assert last_exc is not None  # for type-checkers
        raise last_exc

    @staticmethod
    def _decode(response: requests.Response, *, unwrap: bool) -> Any:
        """Decode a successful response, optionally unwrapping ``data``."""
        if response.status_code == 204 or not response.content:
            return None
        try:
            payload = response.json()
        except ValueError:
            return response.content if not unwrap else response.text
        if not unwrap or not isinstance(payload, dict):
            return payload
        # The OpenAPI envelope always carries a ``data`` field on success.
        if "data" in payload:
            return payload["data"]
        return payload

    # ==================================================================
    # User
    # ==================================================================
    def get_current_user(self) -> JSON:
        """``GET /users/me`` — fetch the authenticated user profile."""
        return self._request("GET", "/users/me")

    # ==================================================================
    # Models
    # ==================================================================
    def list_models(
        self,
        *,
        search: str | None = None,
        owner: str | None = None,
        sort: str | None = None,
        page_number: int = 1,
        page_size: int = 20,
        filters: Filters = None,
    ) -> JSON:
        """``GET /models`` — list models with pagination and filters.

        Supported filter keys: ``task``, ``library``, ``model_type``,
        ``custom_tag``, ``license``, ``deploy``.
        """
        params = self._merge_params(
            {
                "search": search,
                "owner": owner,
                "sort": sort,
                "page_number": page_number,
                "page_size": page_size,
            },
            filters,
        )
        return self._request("GET", "/models", params=params, require_token=False)

    def get_model(self, owner: str, repo_name: str) -> JSON:
        """``GET /models/{owner}/{repo_name}`` — fetch a model's metadata."""
        return self._request("GET", f"/models/{owner}/{repo_name}", require_token=False)

    # ==================================================================
    # Datasets
    # ==================================================================
    def list_datasets(
        self,
        *,
        search: str | None = None,
        owner: str | None = None,
        sort: str | None = None,
        page_number: int = 1,
        page_size: int = 20,
        filters: Filters = None,
    ) -> JSON:
        """``GET /datasets`` — list datasets. Filter keys: ``task``, ``license``."""
        params = self._merge_params(
            {
                "search": search,
                "owner": owner,
                "sort": sort,
                "page_number": page_number,
                "page_size": page_size,
            },
            filters,
        )
        return self._request("GET", "/datasets", params=params, require_token=False)

    def get_dataset(self, owner: str, repo_name: str) -> JSON:
        """``GET /datasets/{owner}/{repo_name}`` — fetch a dataset's metadata."""
        return self._request("GET", f"/datasets/{owner}/{repo_name}", require_token=False)

    # ==================================================================
    # Files
    # ==================================================================
    def upload_file(
        self,
        *,
        file: str | Path | BinaryIO,
        path_in_repo: str,
        repo_id: str,
        repo_type: str = "model",
        revision: str | None = None,
        commit_message: str | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> JSON:
        """``POST /files/upload`` — upload a single file (≤ 5 MiB).

        For larger files, use the LFS upload path exposed by :mod:`_upload`.
        """
        form: list[tuple[str, Any]] = [
            ("path", (None, path_in_repo)),
            ("repo_id", (None, repo_id)),
            ("repo_type", (None, repo_type)),
        ]
        if revision:
            form.append(("revision", (None, revision)))
        if commit_message:
            form.append(("commit_message", (None, commit_message)))
        if extra_fields:
            for key, value in extra_fields.items():
                if value is None:
                    continue
                form.append((key, (None, str(value))))

        opened: BinaryIO | None = None
        try:
            if isinstance(file, (str, Path)):
                file_path = Path(file)
                opened = file_path.open("rb")
                form.append(("file", (file_path.name, opened, "application/octet-stream")))
            else:
                form.append(("file", ("upload.bin", file, "application/octet-stream")))
            return self._request("POST", "/files/upload", files=form)
        finally:
            if opened is not None:
                opened.close()

    # ==================================================================
    # Skills
    # ==================================================================
    def list_skills(
        self,
        *,
        search: str | None = None,
        page_number: int = 1,
        page_size: int = 20,
        filters: Filters = None,
    ) -> JSON:
        """``GET /skills`` — list skills.

        Filter keys: ``developer``, ``category``, ``license``, ``custom_tag``,
        ``owner``.
        """
        params = self._merge_params(
            {
                "search": search,
                "page_number": page_number,
                "page_size": page_size,
            },
            filters,
        )
        return self._request("GET", "/skills", params=params, require_token=False)

    def create_skill(self, payload: CreateSkillPayload | Mapping[str, Any]) -> JSON:
        """``POST /skills`` — create a new skill."""
        return self._request("POST", "/skills", json_body=dict(payload))

    def get_skill(self, skill_id: str | int) -> JSON:
        """``GET /skills/{id}`` — fetch a skill by id."""
        return self._request("GET", f"/skills/{skill_id}", require_token=False)

    def update_skill_settings(
        self,
        owner: str,
        skill_name: str,
        settings: UpdateSkillSettingsPayload | Mapping[str, Any],
    ) -> JSON:
        """``PATCH /skills/{owner}/{skill_name}/settings`` — update skill settings."""
        return self._request(
            "PATCH",
            f"/skills/{owner}/{skill_name}/settings",
            json_body=dict(settings),
        )

    # ==================================================================
    # Studios
    # ==================================================================
    def create_studio(self, payload: CreateStudioPayload | Mapping[str, Any]) -> JSON:
        """``POST /studios`` — create a new Studio space."""
        return self._request("POST", "/studios", json_body=dict(payload))

    def get_studio(self, owner: str, repo_name: str) -> JSON:
        """``GET /studios/{owner}/{repo_name}`` — fetch Studio metadata."""
        return self._request("GET", f"/studios/{owner}/{repo_name}", require_token=False)

    def deploy_studio(
        self,
        owner: str,
        repo_name: str,
        payload: Mapping[str, Any] | None = None,
    ) -> JSON:
        """``POST /studios/{owner}/{repo_name}/deploy`` — trigger a deployment."""
        return self._request(
            "POST",
            f"/studios/{owner}/{repo_name}/deploy",
            json_body=dict(payload or {}),
        )

    def stop_studio(self, owner: str, repo_name: str) -> JSON:
        """``POST /studios/{owner}/{repo_name}/stop`` — stop a running Studio."""
        return self._request("POST", f"/studios/{owner}/{repo_name}/stop")

    def get_studio_logs(
        self,
        owner: str,
        repo_name: str,
        log_type: str,
        *,
        page_num: int = 1,
        page_size: int = 100,
        keyword: str | None = None,
        start_timestamp: int | None = None,
        end_timestamp: int | None = None,
    ) -> JSON:
        """``GET /studios/{owner}/{repo_name}/logs/{log_type}`` — paginated logs."""
        params = self._merge_params(
            {
                "page_num": page_num,
                "page_size": page_size,
                "keyword": keyword,
                "start_timestamp": start_timestamp,
                "end_timestamp": end_timestamp,
            }
        )
        return self._request(
            "GET",
            f"/studios/{owner}/{repo_name}/logs/{log_type}",
            params=params,
        )

    def list_studio_secrets(self, owner: str, repo_name: str) -> JSON:
        """``GET /studios/{owner}/{repo_name}/secrets`` — list configured secrets."""
        return self._request("GET", f"/studios/{owner}/{repo_name}/secrets")

    def add_studio_secret(self, owner: str, repo_name: str, key: str, value: str) -> JSON:
        """``POST /studios/{owner}/{repo_name}/secrets`` — add a new secret."""
        return self._request(
            "POST",
            f"/studios/{owner}/{repo_name}/secrets",
            json_body={"key": key, "value": value},
        )

    def update_studio_secret(self, owner: str, repo_name: str, key: str, value: str) -> JSON:
        """``PUT /studios/{owner}/{repo_name}/secrets`` — overwrite an existing secret."""
        return self._request(
            "PUT",
            f"/studios/{owner}/{repo_name}/secrets",
            json_body={"key": key, "value": value},
        )

    def delete_studio_secret(self, owner: str, repo_name: str, key: str) -> JSON:
        """``DELETE /studios/{owner}/{repo_name}/secrets`` — remove a secret by key."""
        return self._request(
            "DELETE",
            f"/studios/{owner}/{repo_name}/secrets",
            json_body={"key": key},
        )

    def update_studio_settings(
        self,
        owner: str,
        repo_name: str,
        settings: UpdateStudioSettingsPayload | Mapping[str, Any],
    ) -> JSON:
        """``PATCH /studios/{owner}/{repo_name}/settings`` — update Studio settings."""
        return self._request(
            "PATCH",
            f"/studios/{owner}/{repo_name}/settings",
            json_body=dict(settings),
        )

    # ==================================================================
    # MCP (Model Context Protocol) servers
    # ==================================================================
    def list_mcp_servers(
        self,
        *,
        search: str | None = None,
        page_number: int = 1,
        page_size: int = 20,
        extra: Mapping[str, Any] | None = None,
    ) -> JSON:
        """``PUT /mcp/servers`` — discover MCP servers (JSON body, not query)."""
        body: dict[str, Any] = {
            "search": search,
            "page_number": page_number,
            "page_size": page_size,
        }
        if extra:
            body.update(extra)
        body = {k: v for k, v in body.items() if v is not None}
        return self._request("PUT", "/mcp/servers", json_body=body, require_token=False)

    def list_operational_mcp_servers(self) -> JSON:
        """``GET /mcp/servers/operational`` — list servers deployed by the caller."""
        return self._request("GET", "/mcp/servers/operational")

    def get_mcp_server(
        self,
        server_id: str | int,
        *,
        get_operational_url: bool | None = None,
    ) -> JSON:
        """``GET /mcp/servers/{id}`` — fetch a single MCP server's manifest."""
        params = self._merge_params({"get_operational_url": get_operational_url})
        return self._request(
            "GET", f"/mcp/servers/{server_id}", params=params, require_token=False
        )

    def deploy_mcp_server(
        self,
        server_id: str | int,
        payload: DeployMcpServerPayload | Mapping[str, Any] | None = None,
    ) -> JSON:
        """``POST /mcp/servers/{id}/deploy`` — deploy an MCP server for the caller."""
        return self._request(
            "POST",
            f"/mcp/servers/{server_id}/deploy",
            json_body=dict(payload or {}),
        )

    def undeploy_mcp_server(self, server_id: str | int) -> JSON:
        """``DELETE /mcp/servers/{id}/undeploy`` — tear down a deployed MCP server."""
        return self._request("DELETE", f"/mcp/servers/{server_id}/undeploy")


# Re-export iterables-of-strings helper for parity with other modules that may
# want to treat filter keys as a closed set in the future.
def _coerce_keys(keys: Iterable[str]) -> tuple[str, ...]:  # pragma: no cover - reserved
    return tuple(sorted(set(keys)))
