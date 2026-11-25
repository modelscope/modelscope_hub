"""Strict Mirror multipart response contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from .errors import APIError, HubError, _redact_body, raise_for_status

# Business codes are service contracts, not new global E codes.
_ERROR_ROWS = (
    ("INVALID_REQUEST", "fix_request", "E3021"),
    ("TOKEN_INVALID", "reinitialize", "E9001"),
    ("PART_INTEGRITY", "reupload_part", "E2020"),
    ("PARTS_MISSING", "upload_missing_parts", "E1003"),
    ("HASH_CONTEXT_MISMATCH", "reupload_all_parts", "E2020"),
    ("FILE_SHA256_MISMATCH", "reinitialize", "E2020"),
    ("UPLOAD_SESSION_GONE", "reinitialize", "E1003"),
    ("COMPLETION_UNVERIFIED", "reinitialize", "E2020"),
    ("CAPACITY_LIMIT", "retry_later", "E1021"),
    ("UPSTREAM_FAILURE", "check_status", "E1002"),
    ("UPLOAD_CONFLICT", "check_status", "E1003"),
    ("OBJECT_INTEGRITY", "contact_support", "E2020"),
    ("STORAGE_CONFIGURATION", "contact_support", "E1003"),
    ("FEATURE_UNAVAILABLE", "contact_support", "E3023"),
    ("AUTHENTICATION_REQUIRED", "reauthenticate", "E3001"),
    ("REPOSITORY_UNAVAILABLE", "check_access", "E3020"),
    ("INTERNAL_FAILURE", "contact_support", "E1002"),
    ("PART_SIZE_CONFLICT", "resume_original_plan", "E3021"),
    ("UPLOAD_NOT_FOUND", "reinitialize", "E3020"),
    ("TOKEN_REFRESH_REQUIRED", "query_status_without_token", "E9001"),
    ("SOURCE_CLEANUP_FAILED", "retry_complete", "E1003"),
)
_BY_CODE = {10030001001 + i: row for i, row in enumerate(_ERROR_ROWS)}
_BY_NAME = {row[0]: row for row in _ERROR_ROWS}


class MultipartError(APIError):
    """A multipart outcome owned by the inner recovery loop."""

    retryable = False
    recovery_handled = True
    suggestion = "Preserve the upload state and retry manually after resolving the reported cause."


class MultipartFaultError(MultipartError):
    def __init__(self, name: str, recovery: str, ecode: str, *, detail: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(f"Multipart {name} during {detail.get('action', 'upload')}.", **kwargs)
        self.name = name
        self.recovery = recovery
        self.error_code = ecode
        self.part_number = detail.get("partNumber")
        self.missing_parts = detail.get("missingParts")
        self.retry_after: float | None = None


def terminal(error: BaseException) -> MultipartError:
    if isinstance(error, MultipartError):
        return error
    result = MultipartError(
        str(error),
        status_code=getattr(error, "status_code", None),
        request_id=getattr(error, "request_id", None),
        response_body=_redact_body(getattr(error, "response_body", None)),
    )
    result.error_code = getattr(error, "error_code", "E1003") or "E1003"
    return result


def positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise MultipartError(f"Invalid multipart {name}.")
    return value


@dataclass(frozen=True)
class MultipartPlan:
    token: str = field(repr=False)
    sha256: str
    size: int
    part_size: int
    total_parts: int
    parallelism: int
    uploaded: frozenset[int]

    def length(self, number: int) -> int:
        if type(number) is not int or not 1 <= number <= self.total_parts:
            raise MultipartError("Multipart part number is outside the plan.")
        return min(self.part_size, self.size - (number - 1) * self.part_size)

    @classmethod
    def parse(cls, data: dict[str, Any], sha256: str, size: int) -> MultipartPlan:
        token = data.get("uploadToken")
        if not isinstance(token, str) or not token or "\r" in token or "\n" in token:
            raise MultipartError("Missing or invalid multipart upload token.")
        if data.get("resumableMultipart") is not True or data.get("state") != "UPLOADING":
            raise MultipartError("Invalid multipart state.")
        if data.get("sha256") != sha256 or type(data.get("size")) is not int or data["size"] != size:
            raise MultipartError("Multipart response does not match the source content.")
        part_size = positive_int(data.get("partSize"), "partSize")
        total = positive_int(data.get("totalParts"), "totalParts")
        parallelism = positive_int(data.get("maxParallelism"), "maxParallelism")
        if not 100 * 1024 <= part_size <= 64 * 1024**2 or part_size % 64:
            raise MultipartError("Server part size is outside the supported protocol range.")
        if total > 10000 or total != (size + part_size - 1) // part_size:
            raise MultipartError("Inconsistent multipart part count.")
        items = data.get("uploadedParts")
        if not isinstance(items, list):
            raise MultipartError("Missing multipart uploadedParts.")
        plan = cls(token, sha256, size, part_size, total, parallelism, frozenset())
        uploaded: set[int] = set()
        uploaded_bytes = 0
        for item in items:
            if not isinstance(item, dict):
                raise MultipartError("Invalid multipart part metadata.")
            number = positive_int(item.get("partNumber"), "partNumber")
            if number in uploaded or type(item.get("size")) is not int or item["size"] != plan.length(number):
                raise MultipartError("Duplicate or inconsistent multipart part metadata.")
            uploaded.add(number)
            uploaded_bytes += item["size"]
        if type(data.get("uploadedBytes")) is not int or data["uploadedBytes"] != uploaded_bytes:
            raise MultipartError("Inconsistent multipart uploadedBytes.")
        return cls(token, sha256, size, part_size, total, parallelism, frozenset(uploaded))


def _retry_after(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        try:
            value = (parsedate_to_datetime(raw) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, value) if value < float("inf") else None


def parse_response(response: Any, action: str, sha256: str, size: int) -> MultipartPlan | None:
    try:
        body = response.json()
    except ValueError:
        if response.status_code in (429, 500, 502, 503, 504):
            name = "CAPACITY_LIMIT" if response.status_code == 429 else "UPSTREAM_FAILURE"
            fault_row = _BY_NAME[name]
            fault = MultipartFaultError(*fault_row, detail={"action": action}, status_code=response.status_code)
            fault.retry_after = _retry_after(response.headers.get("Retry-After"))
            raise fault from None
        try:
            raise_for_status(response)
        except HubError as exc:
            raise terminal(exc) from exc
        raise MultipartError("Multipart response is not a JSON object.") from None
    if not isinstance(body, dict):
        raise MultipartError("Multipart response is not a JSON object.")
    code = body.get("Code")
    data = body.get("Data")
    success = 200 <= response.status_code < 300 and body.get("Success") is True and code in (200, "200")
    if success:
        if not isinstance(data, dict):
            raise MultipartError("Missing multipart Data object.")
        completed = data == {} and action in {"status", "complete"}
        expected = "completed" if completed else "upload_required"
        header = response.headers.get("X-Upload-Status")
        if header is not None and header != expected:
            raise MultipartError("Multipart status header conflicts with its response.")
        if completed:
            return None
        if action == "complete" or not data:
            raise MultipartError("Unexpected multipart completion or empty plan.")
        return MultipartPlan.parse(data, sha256, size)
    detail = data if isinstance(data, dict) else {}
    try:
        numeric = int(code) if code is not None and not isinstance(code, bool) else None
    except (TypeError, ValueError):
        numeric = None
    row = _BY_CODE.get(numeric) if numeric is not None else None
    error_name = detail.get("errorCode")
    if row is None and numeric is None and isinstance(error_name, str):
        row = _BY_NAME.get(error_name)
    if row is not None:
        if (error_name is not None and error_name != row[0]) or detail.get("recoveryAction", row[1]) != row[1]:
            raise MultipartError("Conflicting multipart recovery metadata.")
        if detail.get("action", action) != action:
            raise MultipartError("Multipart error action conflicts with the request.")
        fault = MultipartFaultError(
            *row,
            detail={**detail, "action": action},
            status_code=response.status_code,
            request_id=body.get("RequestId"),
            response_body=_redact_body(body),
        )
        fault.retry_after = _retry_after(response.headers.get("Retry-After"))
        raise fault
    # Authentication may still use the existing permission envelope.
    if response.status_code in (401, 403, 404):
        try:
            raise_for_status(response)
        except HubError as exc:
            raise terminal(exc) from exc
    if numeric in (None, 200) and response.status_code in (429, 500, 502, 503, 504):
        row = _BY_NAME["CAPACITY_LIMIT" if response.status_code == 429 else "UPSTREAM_FAILURE"]
        fault = MultipartFaultError(
            *row,
            detail={"action": action},
            status_code=response.status_code,
            request_id=body.get("RequestId"),
            response_body=_redact_body(body),
        )
        fault.retry_after = _retry_after(response.headers.get("Retry-After"))
        raise fault
    raise MultipartError(
        "Unknown multipart failure; no automatic recovery was attempted.",
        status_code=response.status_code,
        request_id=body.get("RequestId"),
        response_body=_redact_body(body),
    )
