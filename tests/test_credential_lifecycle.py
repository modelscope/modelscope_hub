# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regression guards for credential teardown.

Covers two defects that left the SDK half-authenticated:

* ``clear_token`` deleted only the session cookie, so the git token and the
  cached identity outlived the login they belonged to.
* :attr:`HubApi.legacy` refused to propagate a *cleared* token to an already
  constructed client, which then kept authenticating with the revoked
  credential.
"""

from __future__ import annotations

import pytest

from modelscope_hub.api import HubApi
from modelscope_hub.config import _CREDENTIAL_FILE_NAMES, HubConfig
from modelscope_hub.constants import (
    COOKIES_FILE_NAME,
    GIT_TOKEN_FILE_NAME,
    SESSION_FILE_NAME,
    USER_INFO_FILE_NAME,
)

TOKEN = "ms-token-under-test"
GIT_TOKEN = "git-token-value"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Redirect credential storage and drop ambient endpoint/token overrides."""
    for name in ("MODELSCOPE_ENDPOINT", "MODELSCOPE_API_TOKEN", "MODELSCOPE_DOMAIN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODELSCOPE_HOME", str(tmp_path))
    return tmp_path


def fully_logged_in(home) -> HubConfig:
    """Write every artefact that a successful login leaves behind."""
    config = HubConfig(config_dir=home)
    config.save_token(TOKEN)
    config.save_git_token(GIT_TOKEN)
    config.save_user_info("alice", "alice@example.com")
    config.get_session_id()  # materialises the session file
    for name in (COOKIES_FILE_NAME, GIT_TOKEN_FILE_NAME, USER_INFO_FILE_NAME, SESSION_FILE_NAME):
        assert (config.credentials_dir / name).exists(), name
    return config


# ---------------------------------------------------------------------------
# All-or-nothing teardown
# ---------------------------------------------------------------------------
def test_clear_token_removes_every_credential_artefact(isolated_home):
    """A partial wipe used to leave a working git token behind."""
    config = fully_logged_in(isolated_home)

    config.clear_token()

    for name in _CREDENTIAL_FILE_NAMES:
        assert not (config.credentials_dir / name).exists(), name
    assert config.load_token() is None
    assert config.load_git_token() is None


def test_clear_token_keeps_the_anonymous_session_id(isolated_home):
    """The session id is an install identifier, not a credential."""
    config = fully_logged_in(isolated_home)
    session_before = config.get_session_id()

    config.clear_token()

    assert (config.credentials_dir / SESSION_FILE_NAME).exists()
    assert config.get_session_id() == session_before


def test_logout_clears_persisted_state(isolated_home):
    """``HubApi.logout`` goes through the same all-or-nothing teardown."""
    fully_logged_in(isolated_home)
    api = HubApi(config=HubConfig(config_dir=isolated_home))
    assert api._config.token == TOKEN

    api.logout()

    reloaded = HubConfig(config_dir=isolated_home)
    assert reloaded.load_token() is None
    assert reloaded.load_git_token() is None


# ---------------------------------------------------------------------------
# Cached client stays in step with the configured token
# ---------------------------------------------------------------------------
def test_cleared_token_propagates_to_the_cached_legacy_client(isolated_home):
    """A cached client must not keep using a credential that was revoked."""
    fully_logged_in(isolated_home)
    api = HubApi(config=HubConfig(config_dir=isolated_home))
    assert api.legacy.token == TOKEN  # materialise the client

    api._config.clear_token()

    assert api.legacy.token is None


def test_rotated_token_propagates_to_the_cached_legacy_client(isolated_home):
    """The pre-existing propagation path keeps working."""
    fully_logged_in(isolated_home)
    api = HubApi(config=HubConfig(config_dir=isolated_home))
    assert api.legacy.token == TOKEN

    api._config.token = "ms-rotated"

    assert api.legacy.token == "ms-rotated"
