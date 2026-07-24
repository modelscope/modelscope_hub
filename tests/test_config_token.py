"""Unit tests for API-token resolution precedence.

Guards the rule: an explicitly provided token -- constructor arg *or* the
``MODELSCOPE_API_TOKEN`` env var -- wins even when empty. An explicit ``""``
means "use no token" and must never silently fall back to the credential
persisted by ``ms login``. Only a completely unset env var falls back.

Regression test for the bug where ``MODELSCOPE_API_TOKEN="" ms-hub agent
upload ...`` uploaded successfully by silently reusing the stored credential.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from modelscope_hub._openapi import OpenAPIClient
from modelscope_hub.config import ENV_TOKEN, HubConfig

_STORED = "ms-STORED-CRED"
_ENDPOINT = "https://pre.modelscope.cn"


@pytest.fixture(autouse=True)
def _persisted_credential():
    """Simulate a credential persisted by ``ms login``.

    Patched at the class level so it also works for the slotted dataclass.
    """
    with patch.object(HubConfig, "load_token", return_value=_STORED):
        yield


@pytest.fixture(autouse=True)
def _clear_token_env(monkeypatch):
    """Start every test from a clean env (no MODELSCOPE_API_TOKEN)."""
    monkeypatch.delenv(ENV_TOKEN, raising=False)


class TestTokenPrecedence:
    def test_env_empty_does_not_fall_back(self, monkeypatch):
        """Explicit empty env var -> token stays '' (no persisted fallback)."""
        monkeypatch.setenv(ENV_TOKEN, "")
        cfg = HubConfig(endpoint=_ENDPOINT)
        assert cfg.token == ""
        assert cfg._token_overridden is True
        assert OpenAPIClient(cfg)._resolve_token() == ""

    def test_env_real_value_is_used(self, monkeypatch):
        """A real env value is used verbatim (and marks the token overridden)."""
        monkeypatch.setenv(ENV_TOKEN, "ms-ENVVALUE")
        cfg = HubConfig(endpoint=_ENDPOINT)
        assert cfg.token == "ms-ENVVALUE"
        assert cfg._token_overridden is True
        assert OpenAPIClient(cfg)._resolve_token() == "ms-ENVVALUE"

    def test_env_unset_falls_back_to_persisted(self):
        """No env var and no explicit arg -> use the persisted credential."""
        cfg = HubConfig(endpoint=_ENDPOINT)
        assert cfg.token == _STORED
        assert cfg._token_overridden is False
        assert OpenAPIClient(cfg)._resolve_token() == _STORED

    def test_explicit_empty_arg_does_not_fall_back(self):
        """Explicit token='' constructor arg -> no persisted fallback."""
        cfg = HubConfig(endpoint=_ENDPOINT, token="")
        assert cfg.token == ""
        assert cfg._token_overridden is True
        assert OpenAPIClient(cfg)._resolve_token() == ""

    def test_resolve_token_lazy_loads_when_not_overridden(self):
        """A non-overridden empty token still lazy-loads the persisted
        credential, preserving the "construct config, then log in" flow."""
        cfg = HubConfig(endpoint=_ENDPOINT)
        # Simulate a config built before login: empty token, not overridden.
        cfg.token = ""
        cfg._token_overridden = False
        assert OpenAPIClient(cfg)._resolve_token() == _STORED
