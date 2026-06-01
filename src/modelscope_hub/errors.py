"""Exception hierarchy for the ModelScope Hub SDK.

The hierarchy is intentionally shallow and protocol-agnostic so callers can
catch broad categories (e.g. :class:`APIError`) without coupling to HTTP
status codes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type-only imports
    from requests import Response


class HubError(Exception):
    """Base class for every error raised by :mod:`modelscope_hub`."""


class APIError(HubError):
    """Error returned by the ModelScope Hub HTTP API."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        response_body: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.request_id = request_id
        self.response_body = response_body

    def __str__(self) -> str:
        parts: list[str] = []
        if self.status_code is not None:
            parts.append(f"[{self.status_code}]")
        parts.append(self.message)
        if self.request_id:
            parts.append(f"(request_id={self.request_id})")
        if self.response_body is not None and self.message.startswith("HTTP "):
            body_str = str(self.response_body)
            if len(body_str) > 500:
                body_str = body_str[:500] + "..."
            parts.append(f"| body={body_str}")
        return " ".join(parts)


class AuthenticationError(APIError):
    """Raised on HTTP 401 — missing or invalid credentials."""


class PermissionError(APIError):  # noqa: A001 - intentional shadow of builtin
    """Raised on HTTP 403 — authenticated but not authorised."""


class NotFoundError(APIError):
    """Raised on HTTP 404 — target resource does not exist."""


class ValidationError(APIError):
    """Raised on HTTP 400 — malformed or invalid request payload."""


class RateLimitError(APIError):
    """Raised on HTTP 429 — client should back off and retry later."""


class ServerError(APIError):
    """Raised on HTTP 5xx — upstream service failure."""


class NetworkError(HubError):
    """Raised when the request could not reach the server."""


class FileIntegrityError(HubError):
    """Raised when a downloaded or uploaded file fails integrity validation."""


class CacheError(HubError):
    """Raised on local cache filesystem or metadata corruption."""


# ---------------------------------------------------------------------------
# Status-code → exception mapping
# ---------------------------------------------------------------------------
_STATUS_MAP: dict[int, type[APIError]] = {
    400: ValidationError,
    401: AuthenticationError,
    403: PermissionError,
    404: NotFoundError,
    429: RateLimitError,
}


def _extract_payload(response: "Response") -> tuple[str, str | None, Any | None]:
    """Best-effort extraction of (message, request_id, body) from a response."""
    request_id = response.headers.get("x-request-id") or response.headers.get("X-Request-Id")
    body: Any | None = None
    message = f"HTTP {response.status_code}"
    try:
        body = response.json()
    except ValueError:
        body = response.text or None
        if isinstance(body, str) and body.strip():
            message = body.strip().splitlines()[0][:500]
        return message, request_id, body

    if isinstance(body, dict):
        for key in ("message", "Message", "msg", "Msg", "error", "Error", "detail", "Detail"):
            value = body.get(key)
            if isinstance(value, str) and value.strip():
                message = value.strip()
                break
        request_id = (
            body.get("request_id") or body.get("requestId")
            or body.get("RequestId") or request_id
        )
    return message, request_id, body


def raise_for_status(response: "Response") -> None:
    """Inspect ``response`` and raise the most specific exception on failure.

    Parameters
    ----------
    response:
        A :class:`requests.Response` instance returned by the SDK transport.

    Raises
    ------
    APIError
        Or a subclass thereof when ``response.status_code`` indicates failure.
    """
    status = response.status_code
    if status < 400:
        return

    message, request_id, body = _extract_payload(response)

    if status >= 500:
        exc_cls: type[APIError] = ServerError
    else:
        exc_cls = _STATUS_MAP.get(status, APIError)

    raise exc_cls(
        message,
        status_code=status,
        request_id=request_id,
        response_body=body,
    )


__all__ = [
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
    "raise_for_status",
]
