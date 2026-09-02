"""Login behaviour for permission-tiered API tokens.

Tokens are issued as read / write / admin. The legacy ``POST /api/v1/login``
endpoint exists to mint *git* credentials, which a read-only token is not
entitled to -- yet such a token authenticates fine and is all a caller needs to
browse and download. Reporting that refusal as "invalid token" sends the user
after the wrong remedy (re-issuing a token that was never the problem).

The transport is stubbed with ``responses`` so the real config, HubApi,
LegacyClient and error-translation layers all take part.
"""

from __future__ import annotations

from unittest import mock

import pytest
import requests
import responses

from modelscope_hub.api import HubApi
from modelscope_hub.config import HubConfig
from modelscope_hub.constants import DEFAULT_ENDPOINT, DEFAULT_INTL_ENDPOINT, USER_INFO_FILE_NAME
from modelscope_hub.errors import AuthenticationError, NetworkError

CN_LOGIN = f"{DEFAULT_ENDPOINT}/api/v1/login"
AI_LOGIN = f"{DEFAULT_INTL_ENDPOINT}/api/v1/login"
CN_USERS_ME = f"{DEFAULT_ENDPOINT}/openapi/v1/users/me"

READONLY_TOKEN = "ms-readonly-token"

# What the legacy login endpoint answers for a token it will not mint git
# credentials for.
LOGIN_REFUSED_BODY = {
    "Code": 10010103009,
    "Message": "登录失败，AccessToken错误，请从用户中心获取AccessToken或刷新",
    "RequestId": "8a039827-3f7c-4378-9b7c-3f8341b73649",
    "Success": False,
}

USERS_ME_BODY = {
    "success": True,
    "data": {"username": "alice", "email": "alice@example.com", "nickname": "Alice"},
    "request_id": "4d3f4b7d-95be-4e6f-95eb-d75178375bd2",
}


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Redirect credential storage and drop ambient endpoint/token overrides."""
    for name in ("MODELSCOPE_ENDPOINT", "MODELSCOPE_API_TOKEN", "MODELSCOPE_DOMAIN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODELSCOPE_HOME", str(tmp_path))
    return tmp_path


def _api(home) -> HubApi:
    return HubApi(config=HubConfig(config_dir=home), endpoint=DEFAULT_ENDPOINT)


# ---------------------------------------------------------------------------
# The token is valid, only its tier is limited
# ---------------------------------------------------------------------------
@responses.activate
def test_scoped_token_logs_in_successfully(isolated_home):
    responses.add(responses.POST, CN_LOGIN, json=LOGIN_REFUSED_BODY, status=400)
    responses.add(responses.GET, CN_USERS_ME, json=USERS_ME_BODY, status=200)

    user = _api(isolated_home).login(READONLY_TOKEN)

    assert user.username == "alice"
    assert user.email == "alice@example.com"


@responses.activate
def test_scoped_login_persists_the_token(isolated_home):
    responses.add(responses.POST, CN_LOGIN, json=LOGIN_REFUSED_BODY, status=400)
    responses.add(responses.GET, CN_USERS_ME, json=USERS_ME_BODY, status=200)

    api = _api(isolated_home)
    api.login(READONLY_TOKEN)

    assert api._config.token == READONLY_TOKEN
    assert HubConfig(config_dir=isolated_home).load_token() == READONLY_TOKEN


@responses.activate
def test_scoped_login_records_the_user_but_no_git_token(isolated_home):
    """Cookies and the git token are unavailable at this tier, so they stay unset."""
    responses.add(responses.POST, CN_LOGIN, json=LOGIN_REFUSED_BODY, status=400)
    responses.add(responses.GET, CN_USERS_ME, json=USERS_ME_BODY, status=200)

    _api(isolated_home).login(READONLY_TOKEN)

    config = HubConfig(config_dir=isolated_home)
    user_file = config.credentials_dir / USER_INFO_FILE_NAME
    assert user_file.read_text(encoding="utf-8") == "alice:alice@example.com"
    assert not config.load_git_token()


@responses.activate
def test_scoped_login_warns_about_write_operations(isolated_home):
    responses.add(responses.POST, CN_LOGIN, json=LOGIN_REFUSED_BODY, status=400)
    responses.add(responses.GET, CN_USERS_ME, json=USERS_ME_BODY, status=200)

    # The SDK logger deliberately does not propagate, so caplog cannot see it.
    with mock.patch("modelscope_hub.api.logger.warning") as warn:
        _api(isolated_home).login(READONLY_TOKEN)

    warn.assert_called_once()
    rendered = warn.call_args[0][0] % warn.call_args[0][1:]
    assert "reduced permissions" in rendered
    assert "write" in rendered
    assert "alice" in rendered


# ---------------------------------------------------------------------------
# A genuinely bad token must still fail, exactly as before
# ---------------------------------------------------------------------------
@responses.activate
def test_invalid_token_still_raises(isolated_home):
    responses.add(responses.POST, CN_LOGIN, json=LOGIN_REFUSED_BODY, status=400)
    responses.add(
        responses.GET,
        CN_USERS_ME,
        json={"success": False, "code": "InvalidAuthentication", "message": "bad token"},
        status=401,
    )

    with pytest.raises(AuthenticationError) as excinfo:
        _api(isolated_home).login("ms-bad-token")

    assert LOGIN_REFUSED_BODY["Message"] in excinfo.value.message
    assert excinfo.value.request_id == LOGIN_REFUSED_BODY["RequestId"]


@responses.activate
def test_invalid_token_leaves_the_stored_credential_alone(isolated_home):
    config = HubConfig(config_dir=isolated_home)
    config.save_token("ms-previously-working")

    responses.add(responses.POST, CN_LOGIN, json=LOGIN_REFUSED_BODY, status=400)
    responses.add(responses.GET, CN_USERS_ME, json={"success": False}, status=401)

    api = _api(isolated_home)
    with pytest.raises(AuthenticationError):
        api.login("ms-bad-token")

    assert HubConfig(config_dir=isolated_home).load_token() == "ms-previously-working"
    assert api._config.token == "ms-previously-working"


@responses.activate
def test_probe_returning_no_username_is_not_treated_as_success(isolated_home):
    """A 200 with an empty profile cannot confirm the token; the refusal stands."""
    responses.add(responses.POST, CN_LOGIN, json=LOGIN_REFUSED_BODY, status=400)
    responses.add(responses.GET, CN_USERS_ME, json={"success": True, "data": {}}, status=200)

    with pytest.raises(AuthenticationError):
        _api(isolated_home).login("ms-odd-token")


@responses.activate
def test_probe_transport_failure_does_not_mask_the_refusal(isolated_home):
    """The probe is advisory: if it cannot run, the original error is surfaced."""
    responses.add(responses.POST, CN_LOGIN, json=LOGIN_REFUSED_BODY, status=400)
    responses.add(responses.GET, CN_USERS_ME, body=requests.ConnectionError("reset"))

    with pytest.raises(AuthenticationError):
        _api(isolated_home).login("ms-bad-token")


# ---------------------------------------------------------------------------
# Non-authentication failures are untouched by this path
# ---------------------------------------------------------------------------
@responses.activate
def test_network_failure_never_reaches_the_probe(isolated_home):
    """A transport error is not a credential verdict, so no probe is attempted."""
    responses.add(responses.POST, CN_LOGIN, body=requests.ConnectionError("connection reset"))

    with pytest.raises(NetworkError):
        _api(isolated_home).login("ms-any-token")

    assert all(CN_USERS_ME not in call.request.url for call in responses.calls)


@responses.activate
def test_successful_login_never_reaches_the_probe(isolated_home):
    responses.add(
        responses.POST,
        CN_LOGIN,
        json={
            "Code": 200,
            "Data": {"AccessToken": "git-token", "Email": "a@b.c", "Username": "alice"},
            "Success": True,
        },
        status=200,
    )
    responses.add(responses.GET, CN_USERS_ME, json=USERS_ME_BODY, status=200)

    api = _api(isolated_home)
    assert api.login("ms-write-token").username == "alice"
    assert HubConfig(config_dir=isolated_home).load_git_token() == "git-token"
