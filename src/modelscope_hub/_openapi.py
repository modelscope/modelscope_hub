"""Low-level client for the ModelScope OpenAPI v1 surface.

This module is *internal*: it is consumed by :mod:`modelscope_hub.api` to build
the ergonomic public façade. Direct use is discouraged and not subject to the
SDK's stability guarantees.

Design goals
------------
* A single :meth:`OpenAPIClient._send` chokepoint owns transport concerns —
  URL composition, authentication injection, retry/back-off, error decoding
  and ``data`` envelope unwrapping. :meth:`OpenAPIClient._request` wraps it with
  the authorisation policy (required token tier, anonymous fallback) and is what
  the endpoint methods call.
* Each OpenAPI endpoint maps to exactly one method, named after the resource it
  manipulates and grouped by section comments. :data:`OPERATION_REGISTRY` records
  that mapping for the tags this client covers in full, and a test checks it
  against the vendored specification so a newly published endpoint cannot go
  unnoticed.
* Filter-style query parameters (``filter.task=...`` etc.) are accepted as a
  flat ``filters`` mapping and serialised transparently.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urljoin, urlsplit

import requests

from .config import HubConfig, get_default_config
from .constants import API_CONNECT_TIMEOUT, API_MAX_RETRIES, API_TIMEOUT, OPENAPI_PREFIX, TokenScope
from .errors import (
    APIError,
    AuthenticationError,
    InvalidParameter,
    NetworkError,
    PermissionDeniedError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    raise_for_status,
)
from .types import (
    CreateAgentIdentityPayload,
    CreateSkillPayload,
    CreateStudioPayload,
    DeployMcpServerPayload,
    PauseAgentPayload,
    ResetAgentKeyPairPayload,
    TokenSignPayload,
    UpdateAgentIdentityPayload,
    UpdateSkillSettingsPayload,
    UpdateStudioSettingsPayload,
)
from .utils.logger import get_logger

__all__ = ["OPERATION_REGISTRY", "OpenAPIClient"]

_logger = get_logger("openapi")

# HTTP methods that are safe to retry without risking duplicate side-effects.
_IDEMPOTENT_METHODS: frozenset[str] = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})

# POST endpoints that are semantically idempotent (deploy/stop are state transitions).
_RETRYABLE_POST_PATHS: frozenset[str] = frozenset({"/deploy", "/stop", "/undeploy"})

# Errors that warrant a transparent retry.
_RETRYABLE_EXC: tuple[type[BaseException], ...] = (
    NetworkError,
    ServerError,
    RateLimitError,
)

# Where a user checks which permission tier their token was issued with.
_TOKEN_SETTINGS_URL = "https://modelscope.cn/my/myaccesstoken"

# Closed value sets published by the Studios section of the specification.
# Validating client-side turns a silently-ignored typo into a named error.
_STUDIO_SORTS: tuple[str, ...] = ("default", "last_modified", "view_num", "likes")
_STUDIO_STATUS_FILTERS: tuple[str, ...] = ("running", "all")
_STUDIO_HARDWARE_TYPES: tuple[str, ...] = ("xgpu", "amd")
_STUDIO_SDK_TYPES: tuple[str, ...] = ("gradio", "streamlit", "docker", "static")
_STUDIO_LOG_TYPES: tuple[str, ...] = ("build", "run")

# ``GET /studios/{owner}/{repo}/logs/{type}`` caps page_size at 500.
_STUDIO_LOG_MAX_PAGE_SIZE = 500

# Agent-IDP declares these closed sets in its OpenAPI request schemas.
_AGENT_IDP_TOKEN_EXPIRIES: frozenset[int] = frozenset({300, 600, 1800, 3600})
_AGENT_IDENTITY_STATUSES: frozenset[str] = frozenset({"active", "paused"})

JSON = dict[str, Any]
QueryParams = list[tuple[str, str]]
Filters = Mapping[str, str | int | float | bool] | None


def _validate_choice(name: str, value: str | None, allowed: tuple[str, ...]) -> None:
    """Reject a value the endpoint's enum does not accept.

    The server ignores an unknown enum value rather than complaining, which turns
    a typo in e.g. ``sort`` into silently wrong results. Failing here names the
    parameter and lists what it accepts.
    """
    if value is not None and value not in allowed:
        raise InvalidParameter(f"{name} must be one of {', '.join(allowed)} (got {value!r}).")


def _as_wire_bool(value: bool | None) -> str | None:
    """Render a tri-state flag the way query strings expect (``true``/``false``)."""
    if value is None:
        return None
    return "true" if value else "false"


# ---------------------------------------------------------------------------
# Operation registry
#
# Maps each specification ``operationId`` to the method implementing it and the
# minimum token permission tier it needs. Two jobs:
#
# 1. ``tests/test_openapi_coverage.py`` checks it against the vendored spec, so a
#    newly published operation fails the suite by name instead of going unnoticed
#    -- which is exactly how the gaps this table closes accumulated.
# 2. It documents the required tier per operation in one place, since the spec
#    itself declares no scopes.
#
# Only the tags the SDK claims to cover are listed; the rest are enumerated as
# deferred in that test.
# ---------------------------------------------------------------------------
OPERATION_REGISTRY: dict[str, tuple[str, TokenScope]] = {
    # -- Agent-IDP ----------------------------------------------------------
    "createAgentIdentity": ("create_agent_identity", TokenScope.WRITE),
    "getAgentIdentity": ("get_agent_identity", TokenScope.READ),
    "updateAgentIdentity": ("update_agent_identity", TokenScope.WRITE),
    "deleteAgentIdentity": ("delete_agent_identity", TokenScope.WRITE),
    "resetAgentKeyPair": ("reset_agent_key_pair", TokenScope.WRITE),
    "pauseAgent": ("pause_agent", TokenScope.WRITE),
    "listAgentTokenRecords": ("list_agent_token_records", TokenScope.READ),
    "listUserAgentIdentities": ("list_user_agent_identities", TokenScope.READ),
    "issueAgentToken": ("issue_agent_token", TokenScope.READ),
    "getAgentIdConfiguration": ("get_agent_id_configuration", TokenScope.READ),
    "getAgentIdJWKS": ("get_agent_id_jwks", TokenScope.READ),
    # -- MCP ----------------------------------------------------------------
    "listMcpServers": ("list_mcp_servers", TokenScope.READ),
    "listOperationalMcpServers": ("list_operational_mcp_servers", TokenScope.READ),
    "getMcpServer": ("get_mcp_server", TokenScope.READ),
    "deployMcpServer": ("deploy_mcp_server", TokenScope.WRITE),
    "undeployMcpServer": ("undeploy_mcp_server", TokenScope.WRITE),
    # -- Studios ------------------------------------------------------------
    "listStudios": ("list_studios", TokenScope.READ),
    "createStudio": ("create_studio", TokenScope.WRITE),
    "getStudio": ("get_studio", TokenScope.READ),
    "updateStudioSettings": ("update_studio_settings", TokenScope.WRITE),
    "deployStudio": ("deploy_studio", TokenScope.WRITE),
    "stopStudio": ("stop_studio", TokenScope.WRITE),
    "getStudioLogs": ("get_studio_logs", TokenScope.READ),
    "listHardware": ("list_studio_hardware", TokenScope.READ),
    "listBaseImages": ("list_studio_base_images", TokenScope.READ),
    "listSdkVersions": ("list_studio_sdk_versions", TokenScope.READ),
    "listStudioSecrets": ("list_studio_secrets", TokenScope.READ),
    "addStudioSecret": ("add_studio_secret", TokenScope.WRITE),
    "updateStudioSecret": ("update_studio_secret", TokenScope.WRITE),
    "deleteStudioSecret": ("delete_studio_secret", TokenScope.WRITE),
    "listStudioVariables": ("list_studio_variables", TokenScope.READ),
    "addStudioVariable": ("add_studio_variable", TokenScope.WRITE),
    "updateStudioVariable": ("update_studio_variable", TokenScope.WRITE),
    "deleteStudioVariable": ("delete_studio_variable", TokenScope.WRITE),
}


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
        Per-request read timeout in seconds. When omitted, uses
        ``(API_CONNECT_TIMEOUT, API_TIMEOUT)`` as ``(connect, read)`` tuple.
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
        self._timeout: float | tuple[float, float] = (
            float(timeout) if timeout is not None else (float(API_CONNECT_TIMEOUT), float(API_TIMEOUT))
        )
        self._max_retries = int(max_retries) if max_retries is not None else int(API_MAX_RETRIES)
        # Tri-state cache for whether this deployment serves ``GET /mcp/servers``.
        # ``None`` means "not probed yet"; see :meth:`list_mcp_servers`.
        self._mcp_list_supports_get: bool | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Release the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> OpenAPIClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public request interface
    # ------------------------------------------------------------------
    def request(
        self,
        method: str,
        path: str = "",
        *,
        url: str | None = None,
        params: Mapping[str, Any] | QueryParams | None = None,
        json_body: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        headers: Mapping[str, str] | None = None,
        require_token: bool = True,
        unwrap: bool = True,
        timeout: float | None = None,
    ) -> Any:
        """Execute an HTTP request.

        When *url* is provided the request targets that absolute URL directly
        (useful for endpoints outside ``/openapi/v1``, signed OSS URLs, etc.).
        Otherwise the request is routed through ``self._url(path)``.

        When *unwrap* is ``False`` the raw :class:`requests.Response` is returned.
        """
        return self._request(
            method,
            path,
            url=url,
            params=params,
            json_body=json_body,
            data=data,
            files=files,
            headers=headers,
            require_token=require_token,
            unwrap=unwrap,
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Request plumbing (internal)
    # ------------------------------------------------------------------
    @property
    def base_url(self) -> str:
        """Fully-qualified OpenAPI base URL, including trailing slash."""
        return f"{(self._config.endpoint or '').rstrip('/')}{OPENAPI_PREFIX}/"

    def _url(self, path: str) -> str:
        # ``urljoin`` treats absolute leading slashes as roots, which would
        # discard the ``/openapi/v1`` prefix. Normalise to a relative form.
        return urljoin(self.base_url, path.lstrip("/"))

    def _resolve_token(self) -> str | None:
        """Resolve the current API token (from config or persisted credential)."""
        token = self._config.token
        if not token and not getattr(self._config, "_token_overridden", False):
            token = self._config.load_token()
            if token:
                self._config.token = token
        return token

    def _auth_headers(self, *, require_token: bool = False) -> dict[str, str]:
        token = self._resolve_token()
        if not token:
            if require_token:
                raise AuthenticationError("Missing API token. Call HubApi.login(...) or set MODELSCOPE_API_TOKEN.")
            return {}
        return {"Authorization": f"Bearer {token}"}

    def _auth_cookies(self) -> dict[str, str]:
        """Build session cookies for /api/v1/ endpoints that use cookie auth."""
        token = self._resolve_token()
        if not token:
            return {}
        return {"m_session_id": token, "modelscope_session": token}

    def _same_host_as_endpoint(self, target_url: str) -> bool:
        """True when *target_url* points at the endpoint host or its LFS sibling.

        Credentials (the ``Authorization`` header and session cookies) must only
        be attached to our own hosts. Absolute URLs pointing elsewhere -- e.g. the
        signed OSS blob-upload URLs on ``*.aliyuncs.com`` -- must never receive
        our token, otherwise it leaks to a third-party domain.

        The LFS blob endpoints live on a sibling host derived from the endpoint
        by swapping the leading label for ``lfs`` / ``pre-lfs`` (e.g. endpoint
        ``pre.modelscope.cn`` -> ``pre-lfs.modelscope.cn``, ``modelscope.cn`` ->
        ``lfs.modelscope.cn``). Those hosts DO require our credentials, so they
        are trusted here too.
        """
        try:
            target_host = (urlsplit(target_url).hostname or "").lower()
        except ValueError:
            return False
        endpoint_host = (urlsplit(self._config.endpoint or "").hostname or "").lower()
        if not target_host or not endpoint_host:
            return False
        if target_host == endpoint_host:
            return True
        # Base domain = endpoint host minus its leading sub-label (if any), so
        # ``pre.modelscope.cn`` -> ``modelscope.cn`` while ``modelscope.cn`` stays.
        labels = endpoint_host.split(".")
        base = ".".join(labels[1:]) if len(labels) >= 3 else endpoint_host
        return target_host in {f"lfs.{base}", f"pre-lfs.{base}"}

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
        path: str = "",
        *,
        url: str | None = None,
        params: Mapping[str, Any] | QueryParams | None = None,
        json_body: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        headers: Mapping[str, str] | None = None,
        require_token: bool = True,
        anonymous: bool = False,
        unwrap: bool = True,
        timeout: float | None = None,
        required_scope: TokenScope | None = None,
        anonymous_retry: bool = False,
    ) -> Any:
        """Execute a request, then interpret authorisation failures.

        Wraps :meth:`_send` (which owns transport, retries and envelope
        decoding) with the two behaviours that depend on *why* a call was made
        rather than *how*:

        *required_scope*
            The minimum token permission tier the endpoint needs. Purely
            advisory -- the Hub does not publish a token's tier, so nothing can
            be pre-validated; on a 403 it turns "permission denied" into a
            message that names the missing tier.
        *anonymous_retry*
            For endpoints whose content is public, retry once without
            credentials when the token is rejected. A read-scoped or stale token
            must not be able to hide data that any anonymous caller can see.
            Never enable this for account-private data: degrading to anonymous
            there would answer "empty" or "not found" and bury the real cause.
        """
        try:
            return self._send(
                method,
                path,
                url=url,
                params=params,
                json_body=json_body,
                data=data,
                files=files,
                headers=headers,
                require_token=require_token,
                anonymous=anonymous,
                unwrap=unwrap,
                timeout=timeout,
            )
        except (AuthenticationError, PermissionDeniedError) as exc:
            if anonymous_retry and not anonymous and self._resolve_token():
                _logger.debug(
                    "Credentials rejected for %s %s (%s); retrying anonymously",
                    method.upper(),
                    url or path,
                    exc.__class__.__name__,
                )
                return self._send(
                    method,
                    path,
                    url=url,
                    params=params,
                    json_body=json_body,
                    data=data,
                    files=files,
                    headers=headers,
                    require_token=False,
                    anonymous=True,
                    unwrap=unwrap,
                    timeout=timeout,
                )
            if required_scope is not None and isinstance(exc, PermissionDeniedError):
                # Shadow the class-level suggestion on this instance only, so the
                # error code, status, request id and traceback all stay intact.
                exc.suggestion = (
                    f"This operation requires a token with '{required_scope.value}' permission. "
                    f"Verify the token's permission level at {_TOKEN_SETTINGS_URL}"
                )
            raise

    def _send(
        self,
        method: str,
        path: str = "",
        *,
        url: str | None = None,
        params: Mapping[str, Any] | QueryParams | None = None,
        json_body: Any | None = None,
        data: Any | None = None,
        files: Any | None = None,
        headers: Mapping[str, str] | None = None,
        require_token: bool = True,
        anonymous: bool = False,
        unwrap: bool = True,
        timeout: float | None = None,
    ) -> Any:
        """Send one logical request and return the unwrapped ``data`` payload.

        Owns the transport concerns only: authentication injection, retries on
        transient errors, and decoding of the standard
        ``{"success": ..., "data": ...}`` envelope. Authorisation policy lives in
        :meth:`_request`, which is what callers should use.

        When *url* is given, it is used as-is (absolute URL). Otherwise the
        final URL is derived from *path* via :meth:`_url`.
        """
        final_url = url or self._url(path)
        # Only attach our credentials when the target is our own host. Absolute
        # URLs to a foreign host (e.g. signed OSS upload URLs) must not receive
        # the Authorization header or session cookies, which carry the token.
        if self._same_host_as_endpoint(final_url) and not anonymous:
            merged_headers = dict(self._auth_headers(require_token=require_token))
            request_cookies = self._auth_cookies()
        else:
            merged_headers = {}
            request_cookies = {}
        if headers:
            merged_headers.update(headers)

        method_upper = method.upper()
        attempts = max(1, self._max_retries)
        last_exc: BaseException | None = None

        for attempt in range(1, attempts + 1):
            _logger.debug("%s %s", method_upper, final_url)
            try:
                response = self._session.request(
                    method=method_upper,
                    url=final_url,
                    params=params,
                    json=json_body,
                    data=data,
                    files=files,
                    headers=merged_headers,
                    cookies=request_cookies,
                    timeout=timeout if timeout is not None else self._timeout,
                )
            except requests.Timeout as exc:
                last_exc = RequestTimeoutError(f"Request timed out: {exc}")
            except requests.ConnectionError as exc:
                last_exc = NetworkError(f"Connection error: {exc}")
            except requests.RequestException as exc:  # pragma: no cover - defensive
                last_exc = NetworkError(f"Request failed: {exc}")
            else:
                _logger.debug("%s %s -> %s", method_upper, final_url, response.status_code)
                if response.status_code >= 400:
                    _logger.debug(
                        "Request failed: %s %s params=%s status=%s body=%s",
                        method_upper,
                        final_url,
                        params,
                        response.status_code,
                        response.text[:500] if response.text else "",
                    )
                try:
                    raise_for_status(response)
                except _RETRYABLE_EXC as exc:  # type: ignore[misc]
                    last_exc = exc
                else:
                    if not unwrap:
                        return response
                    return self._decode(response, unwrap=True)

            # Retry policy: idempotent methods + known-idempotent POST paths.
            # A rate-limited / lock-busy response (RateLimitError) is always
            # safe to retry -- even for a non-idempotent POST -- because the
            # server rejected the request WITHOUT processing it (unlike
            # NetworkError/ServerError, which may have been partially applied).
            is_retryable = (
                isinstance(last_exc, RateLimitError)
                or method_upper in _IDEMPOTENT_METHODS
                or (method_upper == "POST" and path and any(path.endswith(p) for p in _RETRYABLE_POST_PATHS))
            )
            if attempt >= attempts or not is_retryable:
                break
            # Honor a server-provided Retry-After; otherwise exponential backoff.
            # A little jitter avoids concurrent losers retrying in lockstep and
            # colliding on the same per-repo commit lock again.
            retry_after = getattr(last_exc, "retry_after", None)
            if retry_after is not None:
                backoff: float = float(retry_after)
            else:
                backoff = min(2 ** (attempt - 1), 16) + random.uniform(0, 0.5)
            _logger.debug(
                "Retrying %s %s after %s (attempt %d/%d)",
                method_upper,
                final_url,
                last_exc,
                attempt,
                attempts,
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
        # The OpenAPI envelope carries a ``data`` (or ``Data``) field on success.
        if "data" in payload:
            return payload["data"]
        if "Data" in payload:
            return payload["Data"]
        return payload

    # ==================================================================
    # User
    # ==================================================================
    def get_current_user(self) -> JSON:
        """``GET /users/me`` — fetch the authenticated user profile."""
        return self._request("GET", "/users/me")

    def get_current_username(self) -> str:
        """The authenticated account handle, or ``""`` when unresolvable.

        Callers need the handle to build repo paths (``<owner>/<repo>``), and
        the field it arrives in has changed: the endpoint now answers with
        OIDC-style claims (``preferred_username`` / ``name``) where it used to
        return ModelScope's own ``Username``. Reading only the old key yielded
        an empty owner, which then produced confusing downstream failures
        (``path is required`` on create, then a ``//`` URL 404 on commit).

        Keys are tried in handle-before-display-name order so a server that
        populates both still gives the login handle rather than a full name.
        """
        data = self.get_current_user()
        if not isinstance(data, dict):
            return ""
        for key in ("Username", "username", "name", "preferred_username"):
            value = data.get(key)
            if value:
                return str(value)
        return ""

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
        page_size: int = 10,
        filters: Filters = None,
    ) -> JSON:
        """``GET /models`` — list models with pagination and filters.

        Supported filter keys: ``task``, ``library``, ``model_type``,
        ``custom_tag``, ``license``, ``deploy``.
        """
        if page_number * page_size > 3000:
            raise InvalidParameter(f"page_number * page_size must be <= 3000 (got {page_number * page_size}).")
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
        page_size: int = 10,
        filters: Filters = None,
    ) -> JSON:
        """``GET /datasets`` — list datasets. Filter keys: ``task``, ``license``."""
        if page_number * page_size > 3000:
            raise InvalidParameter(f"page_number * page_size must be <= 3000 (got {page_number * page_size}).")
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
        path_in_repo: str | None = None,
        repo_id: str | None = None,
        repo_type: str | None = None,
        revision: str | None = None,
        commit_message: str | None = None,
        extra_fields: Mapping[str, Any] | None = None,
    ) -> JSON:
        """``POST /files/upload`` — upload a single file (≤ 5 MiB).

        When called with ``repo_id`` / ``path_in_repo`` / ``repo_type``,
        the file is committed to the given repository.  When called with
        only ``file``, a generic upload is performed and the response
        contains the file ID (used by skill creation, etc.).
        """
        form: list[tuple[str, Any]] = []
        if path_in_repo is not None:
            form.append(("path", (None, path_in_repo)))
        if repo_id is not None:
            form.append(("repo_id", (None, repo_id)))
        if repo_type is not None:
            form.append(("repo_type", (None, repo_type)))
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
        page_size: int = 10,
        filters: Filters = None,
    ) -> JSON:
        """``GET /skills`` — list skills.

        Filter keys: ``developer``, ``category``, ``license``, ``custom_tag``,
        ``owner``.
        """
        if page_number * page_size > 3000:
            raise InvalidParameter(f"page_number * page_size must be <= 3000 (got {page_number * page_size}).")
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
    def list_studios(
        self,
        *,
        search: str | None = None,
        owner: str | None = None,
        sort: str | None = None,
        page_number: int = 1,
        page_size: int = 10,
        status: str | None = None,
        mcp_support: bool | None = None,
        hardware_type: str | None = None,
    ) -> JSON:
        """``GET /studios`` — list Studio spaces with pagination and filters.

        Note on *status*: the server defaults to ``running`` for a plain search
        but to ``all`` when ``owner`` is set. That switch is deliberately left to
        the server rather than second-guessed here, so passing nothing yields
        whichever default the endpoint considers correct for the query.

        Unlike the other list endpoints this one answers *without* the
        ``{"success": ..., "data": ...}`` envelope -- ``studios`` sits at the top
        level of the response body.
        """
        if page_number * page_size > 3000:
            raise InvalidParameter(f"page_number * page_size must be <= 3000 (got {page_number * page_size}).")
        _validate_choice("sort", sort, _STUDIO_SORTS)
        _validate_choice("status", status, _STUDIO_STATUS_FILTERS)
        _validate_choice("hardware_type", hardware_type, _STUDIO_HARDWARE_TYPES)
        params = self._merge_params(
            {
                "search": search,
                "owner": owner,
                "sort": sort,
                "page_number": page_number,
                "page_size": page_size,
                "status": status,
                "mcp_support": _as_wire_bool(mcp_support),
                "hardware_type": hardware_type,
            }
        )
        return self._request(
            "GET",
            "/studios",
            params=params,
            require_token=False,
            required_scope=TokenScope.READ,
            anonymous_retry=True,
        )

    def list_studio_hardware(
        self,
        *,
        sdk_type: str | None = None,
        studio: str | None = None,
    ) -> JSON:
        """``GET /studios/hardware`` — hardware tiers available to the caller.

        Anonymous callers get the default free tier; an authenticated caller also
        gets the paid tiers with prices. Pass *studio* (``owner/repo_name``) to
        scope the free tiers to what that space may actually use. Paid resources
        are selected as ``paid/<instance_type>`` in the ``hardware`` field of
        :meth:`create_studio` / :meth:`update_studio_settings`.
        """
        _validate_choice("sdk_type", sdk_type, _STUDIO_SDK_TYPES)
        params = self._merge_params({"sdk_type": sdk_type, "studio": studio})
        return self._request(
            "GET",
            "/studios/hardware",
            params=params,
            require_token=False,
            required_scope=TokenScope.READ,
            anonymous_retry=True,
        )

    def list_studio_base_images(self) -> JSON:
        """``GET /studios/base-images`` — base images available to Studio spaces."""
        return self._request(
            "GET",
            "/studios/base-images",
            require_token=False,
            required_scope=TokenScope.READ,
            anonymous_retry=True,
        )

    def list_studio_sdk_versions(self, *, sdk_type: str | None = None) -> JSON:
        """``GET /studios/sdk-versions`` — SDK versions available to Studio spaces.

        Only ``sdk_type="gradio"`` yields versions; every other SDK type (and
        omitting it) returns an empty list.
        """
        _validate_choice("sdk_type", sdk_type, _STUDIO_SDK_TYPES)
        params = self._merge_params({"sdk_type": sdk_type})
        return self._request(
            "GET",
            "/studios/sdk-versions",
            params=params,
            require_token=False,
            required_scope=TokenScope.READ,
            anonymous_retry=True,
        )

    def create_studio(self, payload: CreateStudioPayload | Mapping[str, Any]) -> JSON:
        """``POST /studios`` — create a new Studio space."""
        return self._request(
            "POST",
            "/studios",
            json_body=dict(payload),
            required_scope=TokenScope.WRITE,
        )

    def get_studio(self, owner: str, repo_name: str) -> JSON:
        """``GET /studios/{owner}/{repo_name}`` — fetch Studio metadata.

        Public and experience-public (``protected``) spaces are readable without
        credentials, so no token is demanded up front.
        """
        return self._request(
            "GET",
            f"/studios/{owner}/{repo_name}",
            require_token=False,
            required_scope=TokenScope.READ,
            anonymous_retry=True,
        )

    def deploy_studio(
        self,
        owner: str,
        repo_name: str,
        payload: Mapping[str, Any] | None = None,
    ) -> JSON:
        """``POST /studios/{owner}/{repo_name}/deploy`` — trigger a deployment.

        The specification defines no request body for this operation; *payload*
        is forwarded when given only so that callers written against an older
        server keep working.
        """
        return self._request(
            "POST",
            f"/studios/{owner}/{repo_name}/deploy",
            json_body=dict(payload) if payload else None,
            required_scope=TokenScope.WRITE,
        )

    def stop_studio(self, owner: str, repo_name: str) -> JSON:
        """``POST /studios/{owner}/{repo_name}/stop`` — stop a running Studio."""
        return self._request(
            "POST",
            f"/studios/{owner}/{repo_name}/stop",
            json_body=None,
            required_scope=TokenScope.WRITE,
        )

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
        """``GET /studios/{owner}/{repo_name}/logs/{log_type}`` — paginated logs.

        The response payload carries ``logs``, ``page_num``, ``page_size``,
        ``total_count`` and ``total_page_num``.
        """
        _validate_choice("log_type", log_type, _STUDIO_LOG_TYPES)
        if page_size > _STUDIO_LOG_MAX_PAGE_SIZE:
            raise InvalidParameter(f"page_size must be <= {_STUDIO_LOG_MAX_PAGE_SIZE} (got {page_size}).")
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
            required_scope=TokenScope.READ,
        )

    def list_studio_secrets(self, owner: str, repo_name: str) -> JSON:
        """``GET /studios/{owner}/{repo_name}/secrets`` — list secret keys.

        Only the keys are returned; values are never disclosed. Use
        :meth:`list_studio_variables` for the plaintext counterpart.
        """
        return self._request(
            "GET",
            f"/studios/{owner}/{repo_name}/secrets",
            required_scope=TokenScope.READ,
        )

    def add_studio_secret(self, owner: str, repo_name: str, key: str, value: str) -> JSON:
        """``POST /studios/{owner}/{repo_name}/secrets`` — add a new secret."""
        return self._request(
            "POST",
            f"/studios/{owner}/{repo_name}/secrets",
            json_body={"key": key, "value": value},
            required_scope=TokenScope.WRITE,
        )

    def update_studio_secret(self, owner: str, repo_name: str, key: str, value: str) -> JSON:
        """``PUT /studios/{owner}/{repo_name}/secrets`` — overwrite an existing secret."""
        return self._request(
            "PUT",
            f"/studios/{owner}/{repo_name}/secrets",
            json_body={"key": key, "value": value},
            required_scope=TokenScope.WRITE,
        )

    def delete_studio_secret(self, owner: str, repo_name: str, key: str) -> JSON:
        """``DELETE /studios/{owner}/{repo_name}/secrets`` — remove a secret by key."""
        return self._request(
            "DELETE",
            f"/studios/{owner}/{repo_name}/secrets",
            json_body={"key": key},
            required_scope=TokenScope.WRITE,
        )

    # -- Plaintext variables --------------------------------------------
    # Mirrors the secrets block above one-for-one. The only difference is
    # disclosure: a variable's value is public, a secret's never is.
    def list_studio_variables(self, owner: str, repo_name: str) -> JSON:
        """``GET /studios/{owner}/{repo_name}/variables`` — list plaintext variables.

        Both keys and values are returned, because unlike secrets the values of
        plaintext variables are publicly visible.
        """
        return self._request(
            "GET",
            f"/studios/{owner}/{repo_name}/variables",
            required_scope=TokenScope.READ,
        )

    def add_studio_variable(self, owner: str, repo_name: str, key: str, value: str) -> JSON:
        """``POST /studios/{owner}/{repo_name}/variables`` — add a plaintext variable.

        Both key and value are publicly visible; use
        :meth:`add_studio_secret` for anything sensitive.
        """
        return self._request(
            "POST",
            f"/studios/{owner}/{repo_name}/variables",
            json_body={"key": key, "value": value},
            required_scope=TokenScope.WRITE,
        )

    def update_studio_variable(self, owner: str, repo_name: str, key: str, value: str) -> JSON:
        """``PUT /studios/{owner}/{repo_name}/variables`` — overwrite a plaintext variable."""
        return self._request(
            "PUT",
            f"/studios/{owner}/{repo_name}/variables",
            json_body={"key": key, "value": value},
            required_scope=TokenScope.WRITE,
        )

    def delete_studio_variable(self, owner: str, repo_name: str, key: str) -> JSON:
        """``DELETE /studios/{owner}/{repo_name}/variables`` — remove a variable by key."""
        return self._request(
            "DELETE",
            f"/studios/{owner}/{repo_name}/variables",
            json_body={"key": key},
            required_scope=TokenScope.WRITE,
        )

    def update_studio_settings(
        self,
        owner: str,
        repo_name: str,
        settings: UpdateStudioSettingsPayload | Mapping[str, Any],
    ) -> JSON:
        """``PATCH /studios/{owner}/{repo_name}/settings`` — update Studio settings.

        Only the fields present in *settings* are modified. Changes to
        ``sdk_type`` / ``sdk_version`` / ``base_image`` / ``hardware`` take effect
        on the next deployment.
        """
        return self._request(
            "PATCH",
            f"/studios/{owner}/{repo_name}/settings",
            json_body=dict(settings),
            required_scope=TokenScope.WRITE,
        )

    # ==================================================================
    # MCP (Model Context Protocol) servers
    # ==================================================================
    @staticmethod
    def _flatten_mcp_list_params(body: Mapping[str, Any]) -> QueryParams:
        params: QueryParams = []
        for key, value in body.items():
            if key == "filter" and isinstance(value, Mapping):
                for filter_key, filter_value in value.items():
                    params.append(
                        (
                            f"filter.{filter_key}",
                            str(filter_value).lower() if isinstance(filter_value, bool) else str(filter_value),
                        )
                    )
            else:
                params.append((key, str(value).lower() if isinstance(value, bool) else str(value)))
        return params

    @staticmethod
    def _is_method_or_route_unsupported(exc: APIError) -> bool:
        return exc.status_code in (404, 405, 501)

    # ==================================================================
    # Agent-IDP
    # ==================================================================
    @staticmethod
    def _validate_agent_idp_page(page: int, page_size: int) -> None:
        """Validate the common Agent-IDP pagination contract (1-based, max 50)."""
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise InvalidParameter("page must be an integer >= 1.")
        if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 50:
            raise InvalidParameter("page_size must be an integer between 1 and 50.")

    @staticmethod
    def _validate_agent_public_jwk(value: Any) -> dict[str, Any]:
        """Validate the public-only Ed25519 JWK accepted by registration routes."""
        if not isinstance(value, Mapping):
            raise InvalidParameter("public_key must be an Ed25519 JWK object.")
        key = dict(value)
        if "d" in key:
            raise InvalidParameter("public_key must not contain private JWK material ('d').")
        required = {"kty", "crv", "x", "kid"}
        missing = sorted(name for name in required if not isinstance(key.get(name), str) or not key[name])
        if missing:
            raise InvalidParameter(f"public_key is missing required JWK field(s): {', '.join(missing)}.")
        if key["kty"] != "OKP" or key["crv"] != "Ed25519":
            raise InvalidParameter("public_key must use kty='OKP' and crv='Ed25519'.")
        for field in ("alg", "use"):
            if field in key and key[field] is not None and not isinstance(key[field], str):
                raise InvalidParameter(f"public_key.{field} must be a string.")
        return {field: value for field, value in key.items() if value is not None}

    @staticmethod
    def _validate_agent_token_expiry(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value not in _AGENT_IDP_TOKEN_EXPIRIES:
            allowed = ", ".join(str(item) for item in sorted(_AGENT_IDP_TOKEN_EXPIRIES))
            raise InvalidParameter(f"token_expire_time must be one of {allowed} seconds.")
        return value

    def create_agent_identity(self, payload: CreateAgentIdentityPayload | Mapping[str, Any]) -> JSON:
        """``POST /agent_ids`` — register an Agent-IDP public Ed25519 identity."""
        body = {key: value for key, value in dict(payload).items() if value is not None}
        allowed = {"agent_name", "description", "public_key", "key_alg_type", "token_expire_time"}
        unknown = sorted(set(body) - allowed)
        if unknown:
            raise InvalidParameter(f"Unsupported create Agent-IDP field(s): {', '.join(unknown)}.")
        if not isinstance(body.get("agent_name"), str) or not body["agent_name"].strip():
            raise InvalidParameter("agent_name must be a non-empty string.")
        body["public_key"] = self._validate_agent_public_jwk(body.get("public_key"))
        if "key_alg_type" in body and body["key_alg_type"] != "Ed25519":
            raise InvalidParameter("key_alg_type must be 'Ed25519'.")
        if "token_expire_time" in body:
            body["token_expire_time"] = self._validate_agent_token_expiry(body["token_expire_time"])
        return self._request("POST", "/agent_ids", json_body=body, required_scope=TokenScope.WRITE)

    def get_agent_identity(self, agent_id: str) -> JSON:
        """``GET /agent_ids/{agent_id}`` — fetch one Agent-IDP identity."""
        if not agent_id:
            raise InvalidParameter("agent_id must not be empty.")
        return self._request("GET", f"/agent_ids/{agent_id}", required_scope=TokenScope.READ)

    def update_agent_identity(
        self,
        agent_id: str,
        payload: UpdateAgentIdentityPayload | Mapping[str, Any],
    ) -> JSON:
        """``PATCH /agent_ids/{agent_id}`` — update mutable Agent-IDP metadata."""
        if not agent_id:
            raise InvalidParameter("agent_id must not be empty.")
        body = {key: value for key, value in dict(payload).items() if value is not None}
        allowed = {"agent_name", "description", "token_expire_time"}
        unknown = sorted(set(body) - allowed)
        if unknown:
            raise InvalidParameter(f"Unsupported Agent-IDP update field(s): {', '.join(unknown)}.")
        if not body:
            raise InvalidParameter("update_agent_identity requires at least one field.")
        if "agent_name" in body and (not isinstance(body["agent_name"], str) or not body["agent_name"].strip()):
            raise InvalidParameter("agent_name must be a non-empty string.")
        if "description" in body and not isinstance(body["description"], str):
            raise InvalidParameter("description must be a string.")
        if "token_expire_time" in body:
            body["token_expire_time"] = self._validate_agent_token_expiry(body["token_expire_time"])
        return self._request("PATCH", f"/agent_ids/{agent_id}", json_body=body, required_scope=TokenScope.WRITE)

    def delete_agent_identity(self, agent_id: str) -> JSON:
        """``DELETE /agent_ids/{agent_id}`` — delete an Agent-IDP identity."""
        if not agent_id:
            raise InvalidParameter("agent_id must not be empty.")
        return self._request("DELETE", f"/agent_ids/{agent_id}", required_scope=TokenScope.WRITE)

    def reset_agent_key_pair(
        self,
        agent_id: str,
        payload: ResetAgentKeyPairPayload | Mapping[str, Any],
    ) -> JSON:
        """``PUT /agent_ids/{agent_id}/key_pairs`` — replace an identity public key."""
        if not agent_id:
            raise InvalidParameter("agent_id must not be empty.")
        body = {key: value for key, value in dict(payload).items() if value is not None}
        allowed = {"public_key", "key_alg_type"}
        unknown = sorted(set(body) - allowed)
        if unknown:
            raise InvalidParameter(f"Unsupported Agent-IDP key-reset field(s): {', '.join(unknown)}.")
        body["public_key"] = self._validate_agent_public_jwk(body.get("public_key"))
        if "key_alg_type" in body and body["key_alg_type"] != "Ed25519":
            raise InvalidParameter("key_alg_type must be 'Ed25519'.")
        return self._request(
            "PUT",
            f"/agent_ids/{agent_id}/key_pairs",
            json_body=body,
            required_scope=TokenScope.WRITE,
        )

    def pause_agent(self, agent_id: str, payload: PauseAgentPayload | Mapping[str, Any]) -> JSON:
        """``POST /agent_ids/{agent_id}/paused`` — enable or pause token issuance."""
        if not agent_id:
            raise InvalidParameter("agent_id must not be empty.")
        body = dict(payload)
        if set(body) != {"paused"} or not isinstance(body.get("paused"), bool):
            raise InvalidParameter("pause_agent requires exactly a boolean 'paused' field.")
        return self._request(
            "POST",
            f"/agent_ids/{agent_id}/paused",
            json_body=body,
            required_scope=TokenScope.WRITE,
        )

    def list_agent_token_records(self, agent_id: str, *, page: int = 1, page_size: int = 20) -> JSON:
        """``GET /agent_ids/{agent_id}/jwt_id_tokens`` — list issued-token records."""
        if not agent_id:
            raise InvalidParameter("agent_id must not be empty.")
        self._validate_agent_idp_page(page, page_size)
        return self._request(
            "GET",
            f"/agent_ids/{agent_id}/jwt_id_tokens",
            params=self._merge_params({"page": page, "page_size": page_size}),
            required_scope=TokenScope.READ,
        )

    def list_user_agent_identities(
        self,
        username: str,
        *,
        status: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> JSON:
        """``GET /users/{username}/agent_ids`` — list a user's Agent-IDP identities."""
        if not username:
            raise InvalidParameter("username must not be empty.")
        if status is not None and status not in _AGENT_IDENTITY_STATUSES:
            allowed = ", ".join(sorted(_AGENT_IDENTITY_STATUSES))
            raise InvalidParameter(f"status must be one of {allowed}.")
        self._validate_agent_idp_page(page, page_size)
        return self._request(
            "GET",
            f"/users/{username}/agent_ids",
            params=self._merge_params({"status": status, "page": page, "page_size": page_size}),
            required_scope=TokenScope.READ,
        )

    def issue_agent_token(self, payload: TokenSignPayload | Mapping[str, Any]) -> JSON:
        """``POST /agent_id/token`` — exchange a locally signed request for a JWT.

        The signature itself authenticates this public endpoint. Explicit
        anonymous transport ensures an ambient Hub token is never attached.
        """
        body = dict(payload)
        required = {"agent_id", "kid", "audience", "timestamp", "signature"}
        if set(body) != required:
            raise InvalidParameter("issue_agent_token requires agent_id, kid, audience, timestamp and signature.")
        for field in ("agent_id", "kid", "audience", "signature"):
            if not isinstance(body[field], str) or not body[field]:
                raise InvalidParameter(f"{field} must be a non-empty string.")
        if isinstance(body["timestamp"], bool) or not isinstance(body["timestamp"], int) or body["timestamp"] <= 0:
            raise InvalidParameter("timestamp must be a positive Unix timestamp in seconds.")
        return self._request("POST", "/agent_id/token", json_body=body, require_token=False, anonymous=True)

    def get_agent_id_configuration(self) -> JSON:
        """Fetch anonymous Agent-IDP OIDC discovery metadata."""
        return self._request(
            "GET",
            "/agent_id/.well-known/agentid-configuration",
            require_token=False,
            anonymous=True,
        )

    def get_agent_id_jwks(self) -> JSON:
        """Fetch the anonymous Agent-IDP JWT validation key set."""
        return self._request(
            "GET",
            "/agent_id/.well-known/agentid-jwks",
            require_token=False,
            anonymous=True,
        )

    # ==================================================================
    # MCP
    # ==================================================================
    def list_mcp_servers(
        self,
        *,
        search: str | None = None,
        page_number: int = 1,
        page_size: int = 20,
        filter: Mapping[str, Any] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> JSON:
        """``PUT /mcp/servers`` — discover MCP servers.

        ``PUT`` is the only method the specification defines for this route, so it
        is tried first. A ``GET`` variant is probed only after a ``PUT`` failure,
        because probing first cost every single call a wasted round trip to a 404.
        Whichever verb a deployment turns out to serve is remembered for the life
        of this client, so the loser is attempted at most once.

        Parameters
        ----------
        filter : dict, optional
            Nested filter object. Supported keys: ``category``, ``is_hosted``.
        """
        if page_number * page_size > 100:
            # The service enforces this itself, answering 403 QuotaLimitExceed with
            # exactly this rule. Checking here spares the round trip and reports it
            # as what it is -- a caller parameter mistake, not an exhausted quota.
            raise InvalidParameter(
                f"page_number * page_size must be <= 100 for MCP servers (got {page_number * page_size})."
            )
        body: dict[str, Any] = {
            "search": search,
            "page_number": page_number,
            "page_size": page_size,
        }
        if filter:
            body["filter"] = dict(filter)
        if extra:
            body.update(extra)
        body = {k: v for k, v in body.items() if v is not None}

        def _get() -> JSON:
            return self._request(
                "GET",
                "/mcp/servers",
                params=self._flatten_mcp_list_params(body),
                require_token=False,
                required_scope=TokenScope.READ,
                anonymous_retry=True,
            )

        if self._mcp_list_supports_get:
            # This deployment already answered GET and refused PUT, so leading
            # with PUT again would just repeat a known 404 on every call.
            return _get()

        try:
            return self._request(
                "PUT",
                "/mcp/servers",
                json_body=body,
                require_token=False,
                required_scope=TokenScope.READ,
                anonymous_retry=True,
            )
        except APIError as exc:
            if self._mcp_list_supports_get is False or not self._is_method_or_route_unsupported(exc):
                raise

        try:
            data = _get()
        except APIError as exc:
            if self._is_method_or_route_unsupported(exc):
                # Serves neither verb: stop probing GET on subsequent calls.
                self._mcp_list_supports_get = False
            raise
        # Serves GET but not PUT: skip the PUT attempt from now on.
        self._mcp_list_supports_get = True
        return data

    def list_operational_mcp_servers(self) -> JSON:
        """``GET /mcp/servers/operational`` — list servers deployed by the caller.

        Answers with the caller's own hosting, so no anonymous fallback: degrading
        to an anonymous request would report "nothing deployed" for what is really
        a credential problem.
        """
        return self._request(
            "GET",
            "/mcp/servers/operational",
            required_scope=TokenScope.READ,
        )

    def get_mcp_server(
        self,
        server_id: str | int,
        *,
        get_operational_url: bool | None = None,
    ) -> JSON:
        """``GET /mcp/servers/{id}`` — fetch a single MCP server's manifest."""
        params = self._merge_params({"get_operational_url": _as_wire_bool(get_operational_url)})
        return self._request(
            "GET",
            f"/mcp/servers/{server_id}",
            params=params,
            require_token=False,
            required_scope=TokenScope.READ,
            anonymous_retry=True,
        )

    def deploy_mcp_server(
        self,
        server_id: str | int,
        payload: DeployMcpServerPayload | Mapping[str, Any] | None = None,
    ) -> JSON:
        """``POST /mcp/servers/{id}/deploy`` — deploy an MCP server for the caller."""
        # Drop explicit None values so they never reach the wire, then apply
        # the platform default transport.
        body = {k: v for k, v in dict(payload or {}).items() if v is not None}
        body.setdefault("transport_type", "sse")
        return self._request(
            "POST",
            f"/mcp/servers/{server_id}/deploy",
            json_body=body,
            required_scope=TokenScope.WRITE,
        )

    def undeploy_mcp_server(self, server_id: str | int) -> JSON:
        """``DELETE /mcp/servers/{id}/undeploy`` — tear down a deployed MCP server."""
        return self._request(
            "DELETE",
            f"/mcp/servers/{server_id}/undeploy",
            required_scope=TokenScope.WRITE,
        )


# Re-export iterables-of-strings helper for parity with other modules that may
# want to treat filter keys as a closed set in the future.
def _coerce_keys(keys: Iterable[str]) -> tuple[str, ...]:  # pragma: no cover - reserved
    return tuple(sorted(set(keys)))
