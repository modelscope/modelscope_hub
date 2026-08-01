# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression guards for the ``HubApi.login`` failure paths.

These tests stub the ``requests`` transport only, so the real
:class:`~modelscope_hub.config.HubConfig`, :class:`~modelscope_hub.api.HubApi`,
:class:`~modelscope_hub._legacy_api.LegacyClient` and error-translation layers
all take part. Mocking any higher would hide precisely the defects covered
here: a rejected login used to surface as a fabricated "token rejected"
message -- losing the server's own explanation and request id -- while
deleting the credential the caller already had on disk.

The payloads below are the ones ``POST /api/v1/login`` actually returns.
"""

from __future__ import annotations

import json

import pytest
import requests
import responses

from modelscope_hub.api import HubApi
from modelscope_hub.config import HubConfig
from modelscope_hub.constants import DEFAULT_ENDPOINT, DEFAULT_INTL_ENDPOINT
from modelscope_hub.errors import (
    AlreadyExistsError,
    AuthenticationError,
    InvalidParameter,
    NetworkError,
    ServerError,
    raise_for_status,
)

CN_LOGIN = f"{DEFAULT_ENDPOINT}/api/v1/login"
AI_LOGIN = f"{DEFAULT_INTL_ENDPOINT}/api/v1/login"

# Both sites answer an unknown token with this same business code, which is why
# the server cannot tell "invalid token" apart from "token issued elsewhere".
TOKEN_REJECTED_BODY = {
    "Code": 10010103009,
    "Message": "登录失败，AccessToken错误，请从用户中心获取AccessToken或刷新",
    "RequestId": "8a039827-3f7c-4378-9b7c-3f8341b73649",
    "Success": False,
}

LOGIN_OK_BODY = {
    "Code": 200,
    "Data": {"AccessToken": "git-token", "Email": "alice@example.com", "Username": "alice"},
    "Message": "success",
    "RequestId": "b008966a-942f-4c05-8f8a-696d1b6cc2e2",
    "Success": True,
}

PRIOR_TOKEN = "ms-previously-working"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Redirect credential storage and drop ambient endpoint/token overrides."""
    for name in ("MODELSCOPE_ENDPOINT", "MODELSCOPE_API_TOKEN", "MODELSCOPE_DOMAIN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODELSCOPE_HOME", str(tmp_path))
    return tmp_path


def unpinned_api(home) -> HubApi:
    """Build a HubApi with no explicit endpoint, so the peer probe may run."""
    return HubApi(config=HubConfig(config_dir=home))


def seed_stored_credential(home) -> None:
    """Persist a working credential the way a previous login would have."""
    config = HubConfig(config_dir=home)
    config.save_token(PRIOR_TOKEN)
    assert config.load_token() == PRIOR_TOKEN


def stored_token(home) -> str | None:
    return HubConfig(config_dir=home).load_token()


# ---------------------------------------------------------------------------
# Faithful attribution
# ---------------------------------------------------------------------------
@responses.activate
def test_rejected_token_keeps_server_explanation(isolated_home):
    """The server's message, business code and request id must survive."""
    responses.add(responses.POST, CN_LOGIN, json=TOKEN_REJECTED_BODY, status=400)
    responses.add(responses.POST, AI_LOGIN, json=TOKEN_REJECTED_BODY, status=400)

    with pytest.raises(AuthenticationError) as excinfo:
        unpinned_api(isolated_home).login("ms-bad-token")

    exc = excinfo.value
    # Business code 10010103009 outranks the HTTP 400 the server chose.
    assert exc.error_code == "E3001"
    assert exc.status_code == 400
    assert exc.request_id == TOKEN_REJECTED_BODY["RequestId"]
    assert TOKEN_REJECTED_BODY["Message"] in exc.message


@responses.activate
def test_network_failure_is_not_reported_as_bad_token(isolated_home):
    """Transport errors keep their own identity instead of blaming the token."""
    seed_stored_credential(isolated_home)
    responses.add(responses.POST, CN_LOGIN, body=requests.ConnectionError("connection reset"))

    with pytest.raises(NetworkError) as excinfo:
        HubApi(config=HubConfig(config_dir=isolated_home)).login("ms-any-token")

    assert not isinstance(excinfo.value, AuthenticationError)
    assert stored_token(isolated_home) == PRIOR_TOKEN


@responses.activate
def test_server_error_propagates_unchanged(isolated_home):
    """A 5xx is a server outage, not a credential problem."""
    responses.add(responses.POST, CN_LOGIN, json={"Code": 500, "Message": "internal"}, status=500)

    with pytest.raises(ServerError) as excinfo:
        unpinned_api(isolated_home).login("ms-any-token")

    assert excinfo.value.error_code == "E1002"


# ---------------------------------------------------------------------------
# Stored credentials are never collateral damage
# ---------------------------------------------------------------------------
@responses.activate
def test_failed_login_keeps_stored_credential(isolated_home):
    """A mistyped token must not log the user out of a working session."""
    seed_stored_credential(isolated_home)
    responses.add(responses.POST, CN_LOGIN, json=TOKEN_REJECTED_BODY, status=400)
    responses.add(responses.POST, AI_LOGIN, json=TOKEN_REJECTED_BODY, status=400)

    api = HubApi(config=HubConfig(config_dir=isolated_home))
    with pytest.raises(AuthenticationError):
        api.login("ms-bad-token")

    assert stored_token(isolated_home) == PRIOR_TOKEN
    # The instance is also rewound, not left holding the rejected token.
    assert api._config.token == PRIOR_TOKEN


# ---------------------------------------------------------------------------
# Site disambiguation
# ---------------------------------------------------------------------------
@responses.activate
def test_token_valid_on_peer_site_yields_endpoint_hint(isolated_home):
    """A token issued by the other site gets an actionable hint, not a verdict."""
    responses.add(responses.POST, CN_LOGIN, json=TOKEN_REJECTED_BODY, status=400)
    responses.add(responses.POST, AI_LOGIN, json=LOGIN_OK_BODY, status=200)

    with pytest.raises(AuthenticationError) as excinfo:
        unpinned_api(isolated_home).login("ms-intl-token")

    message = excinfo.value.message
    assert DEFAULT_INTL_ENDPOINT in message
    assert "--endpoint" in message


@responses.activate
def test_pinned_endpoint_is_not_second_guessed(isolated_home):
    """An explicit endpoint is respected: no peer probe and no hint."""
    responses.add(responses.POST, CN_LOGIN, json=TOKEN_REJECTED_BODY, status=400)

    api = HubApi(config=HubConfig(config_dir=isolated_home), endpoint=DEFAULT_ENDPOINT)
    with pytest.raises(AuthenticationError) as excinfo:
        api.login("ms-bad-token")

    assert "--endpoint" not in excinfo.value.message
    assert all(AI_LOGIN not in call.request.url for call in responses.calls)


# ---------------------------------------------------------------------------
# Classification table
# ---------------------------------------------------------------------------
def json_response(status: int, body: dict) -> requests.Response:
    """Build a minimal response the error layer can classify."""
    resp = requests.Response()
    resp.status_code = status
    resp._content = json.dumps(body).encode()
    resp.headers["Content-Type"] = "application/json"
    resp.url = CN_LOGIN
    return resp


@pytest.mark.parametrize(
    "body, expected",
    [
        ({"Code": 10010103009, "Message": "token bad"}, AuthenticationError),
        ({"Code": 10010101001, "Message": "model exists"}, AlreadyExistsError),
        ({"Code": 99999999999, "Message": "something else"}, InvalidParameter),
        ({"Message": "no code at all"}, InvalidParameter),
    ],
)
def test_business_code_outranks_http_status(body, expected):
    """A published business code classifies the failure; status is the fallback."""
    with pytest.raises(expected) as excinfo:
        raise_for_status(json_response(400, body))
    # AlreadyExistsError subclasses InvalidParameter, so assert the exact type.
    assert type(excinfo.value) is expected


@pytest.mark.parametrize("code", [10010103009, 10010101001])
def test_business_code_does_not_override_a_server_outage(code):
    """A 5xx stays a retryable ServerError even when the body carries a known code.

    Reclassifying it would flip ``retryable`` to False and silently turn a
    transient outage into a permanent failure.
    """
    with pytest.raises(ServerError) as excinfo:
        raise_for_status(json_response(500, {"Code": code, "Message": "upstream failure"}))

    assert type(excinfo.value) is ServerError
    assert excinfo.value.retryable is True


# ---------------------------------------------------------------------------
# Endpoint normalisation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "given, expected",
    [
        ("modelscope.ai", "https://modelscope.ai"),
        ("https://modelscope.cn/", "https://modelscope.cn"),
        ("  modelscope.cn  ", "https://modelscope.cn"),
        ("http://localhost:8080", "http://localhost:8080"),
        # URI schemes are case-insensitive: recognise, do not prefix again.
        ("HTTPS://modelscope.cn", "HTTPS://modelscope.cn"),
        ("Http://localhost:8080", "Http://localhost:8080"),
    ],
)
def test_bare_endpoint_gains_a_scheme(isolated_home, given, expected):
    """Bare domains used to reach the transport layer and fail there."""
    api = HubApi(config=HubConfig(config_dir=isolated_home), endpoint=given)
    assert api._config.endpoint == expected


@responses.activate
def test_bare_endpoint_reaches_the_expected_url(isolated_home):
    """End-to-end proof that a scheme-less endpoint now resolves correctly."""
    url = "https://modelscope.ai/api/v1/login"
    responses.add(responses.POST, url, json=TOKEN_REJECTED_BODY, status=400)

    api = HubApi(config=HubConfig(config_dir=isolated_home), endpoint="modelscope.ai")
    with pytest.raises(AuthenticationError):
        api.login("ms-bad-token")

    assert responses.calls[0].request.url == url
