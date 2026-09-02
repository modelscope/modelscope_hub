"""Classification tests for the OpenAPI error envelope.

The ``/openapi/v1`` surface reports failures as ``{"success": false, "code":
"<StringEnum>", "message": ..., "request_id": ...}``. The SDK only understood the
legacy *numeric* business code, so every one of these string codes fell through
to the bare HTTP status -- collapsing distinctions the status cannot express:
403 covers both "insufficient token permission" and "quota exhausted", and a 409
duplicate had no mapping at all.
"""

from __future__ import annotations

import json

import pytest
import requests

from modelscope_hub._openapi import _RETRYABLE_EXC
from modelscope_hub.errors import (
    AlreadyExistsError,
    APIError,
    AuthenticationError,
    InvalidParameter,
    NotExistError,
    PermissionDeniedError,
    QuotaExceededError,
    RateLimitError,
    ServerError,
    raise_for_status,
)

_URL = "https://modelscope.cn/openapi/v1/studios"


def _openapi_error(status: int, code: str, message: str = "failed") -> requests.Response:
    """Build the exact envelope the OpenAPI surface returns on failure."""
    resp = requests.Response()
    resp.status_code = status
    resp._content = json.dumps(
        {
            "success": False,
            "code": code,
            "message": message,
            "request_id": "4d3f4b7d-95be-4e6f-95eb-d75178375bd2",
        }
    ).encode()
    resp.headers["Content-Type"] = "application/json"
    resp.url = _URL
    return resp


def _plain_error(status: int) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp._content = b'{"success": false, "message": "no code"}'
    resp.headers["Content-Type"] = "application/json"
    resp.url = _URL
    return resp


# ---------------------------------------------------------------------------
# String code -> exception
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("status", "code", "expected"),
    [
        (400, "InputParameterError", InvalidParameter),
        (401, "InvalidAuthentication", AuthenticationError),
        (403, "OperationNotAllowed", PermissionDeniedError),
        (404, "ResourceNotFound", NotExistError),
        (409, "DuplicateEntity", AlreadyExistsError),
        (403, "QuotaLimitExceed", QuotaExceededError),
        (429, "RateLimitExceed", RateLimitError),
    ],
)
def test_string_code_selects_the_exception(status, code, expected):
    with pytest.raises(expected) as excinfo:
        raise_for_status(_openapi_error(status, code))
    # AlreadyExistsError subclasses InvalidParameter, so assert the exact type.
    assert type(excinfo.value) is expected


def test_quota_and_permission_share_403_but_not_the_exception():
    """403 is overloaded upstream; the code is the only thing that separates them."""
    with pytest.raises(PermissionDeniedError):
        raise_for_status(_openapi_error(403, "OperationNotAllowed"))
    with pytest.raises(QuotaExceededError):
        raise_for_status(_openapi_error(403, "QuotaLimitExceed"))


def test_message_request_id_and_status_survive_classification():
    with pytest.raises(QuotaExceededError) as excinfo:
        raise_for_status(_openapi_error(403, "QuotaLimitExceed", "magicube balance exhausted"))
    exc = excinfo.value
    assert exc.status_code == 403
    assert exc.request_id == "4d3f4b7d-95be-4e6f-95eb-d75178375bd2"
    assert "magicube balance exhausted" in exc.message


@pytest.mark.parametrize("code", ["ServiceUnavailable", "InternalServerError"])
@pytest.mark.parametrize("status", [500, 503])
def test_server_side_codes_stay_retryable(status, code):
    """A 5xx must never be reclassified: doing so would drop ``retryable``."""
    with pytest.raises(ServerError) as excinfo:
        raise_for_status(_openapi_error(status, code))
    assert type(excinfo.value) is ServerError
    assert excinfo.value.retryable is True


def test_unknown_string_code_falls_back_to_the_status():
    with pytest.raises(NotExistError):
        raise_for_status(_openapi_error(404, "SomeCodeWeHaveNeverSeen"))


# ---------------------------------------------------------------------------
# Status-code table gaps this work closed
# ---------------------------------------------------------------------------
def test_409_maps_to_already_exists_even_without_a_code():
    """createStudio / addStudioSecret / addStudioVariable all answer bare 409."""
    with pytest.raises(AlreadyExistsError) as excinfo:
        raise_for_status(_plain_error(409))
    assert type(excinfo.value) is AlreadyExistsError


def test_413_maps_to_invalid_parameter():
    """POST /files/upload answers 413 when the body exceeds 5 MiB."""
    with pytest.raises(InvalidParameter):
        raise_for_status(_plain_error(413))


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------
def test_quota_exceeded_is_not_retryable():
    """Reusing RateLimitError here would make the transport retry a hard failure."""
    assert QuotaExceededError.retryable is False
    assert not issubclass(QuotaExceededError, RateLimitError)
    assert QuotaExceededError not in _RETRYABLE_EXC
    assert not issubclass(QuotaExceededError, _RETRYABLE_EXC)


def test_rate_limit_exceeded_stays_retryable():
    assert RateLimitError.retryable is True
    assert issubclass(RateLimitError, _RETRYABLE_EXC)


def test_quota_error_is_an_api_error():
    assert issubclass(QuotaExceededError, APIError)
    assert QuotaExceededError.error_code == "E3027"


def test_quota_error_is_exported_from_the_package_root():
    import modelscope_hub

    assert modelscope_hub.QuotaExceededError is QuotaExceededError
    assert "QuotaExceededError" in modelscope_hub.__all__


def test_retry_after_still_honoured_for_a_string_rate_limit_code():
    """RateLimitError carries retry_after regardless of how it was selected."""
    resp = _openapi_error(429, "RateLimitExceed")
    resp.headers["Retry-After"] = "7"
    with pytest.raises(RateLimitError) as excinfo:
        raise_for_status(resp)
    assert excinfo.value.retry_after == 7
