# Copyright (c) Alibaba, Inc. and its affiliates.
"""Anonymous (credential-free) access tests for the agent read-only APIs.

Regression tests for the bug where ``download_repo_file`` (and the whole
download chain) raised a local ``AuthenticationError`` before any request
was sent when no token was available -- breaking anonymous downloads of
public repos.

Also covers the "explicit empty token must never fall back to persisted
credentials" contract for both the OpenAPI client and ``HubApi.get_cookies``.

All tests are offline: the HTTP session is mocked and the persisted
credential loaders are patched with sentinels that must never leak into
outgoing requests.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from requests.cookies import RequestsCookieJar

from modelscope_hub.agent import AgentApi
from modelscope_hub.api import HubApi
from modelscope_hub.config import HubConfig
from modelscope_hub.errors import AuthenticationError

ENDPOINT = "https://pre.modelscope.cn"
# Sentinel persisted credential: tests assert it never reaches a request.
STORED = "ms-STORED-MUST-NOT-LEAK"


def _mock_response(json_data=None, content=b""):
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {}
    payload = json_data or {"Code": 200, "Data": {}, "Success": True}
    resp.json.return_value = payload
    # _decode() short-circuits to None on empty content, so keep it non-empty.
    resp.content = content or json.dumps(payload).encode()
    resp.text = content.decode() if content else ""
    return resp


@pytest.fixture()
def anon_api(monkeypatch):
    """AgentApi with an explicitly empty token and persisted creds present."""
    monkeypatch.delenv("MODELSCOPE_API_TOKEN", raising=False)
    with (
        patch.object(HubConfig, "load_token", return_value=STORED),
        patch.object(HubConfig, "load_cookies", return_value=None),
    ):
        yield AgentApi(endpoint=ENDPOINT, token="", timeout=5)


def _sent_credentials(mock_req):
    kw = mock_req.call_args.kwargs
    return (kw.get("headers") or {}).get("Authorization"), kw.get("cookies") or {}


class TestAnonymousReadOnly:
    """Read-only agent APIs must work without any credentials."""

    def test_repo_info_sends_request_without_credentials(self, anon_api):
        resp = _mock_response({"Code": 200, "Data": {"Name": "demo"}, "Success": True})
        with patch.object(anon_api._openapi._session, "request", return_value=resp) as m:
            info = anon_api.repo_info("someone", "public-repo")
        assert info is not None
        auth, cookies = _sent_credentials(m)
        assert auth is None
        assert cookies == {}

    def test_list_repo_files_sends_request_without_credentials(self, anon_api):
        resp = _mock_response(
            {
                "Code": 200,
                "Success": True,
                "Data": {
                    "Trees": [
                        {"Path": "AGENTS.md", "Type": "blob", "Sha256": "abc", "IsLfs": False},
                    ]
                },
            }
        )
        with patch.object(anon_api._openapi._session, "request", return_value=resp) as m:
            files = anon_api.list_repo_files("someone", "public-repo")
        assert files == ["AGENTS.md"]
        auth, cookies = _sent_credentials(m)
        assert auth is None
        assert cookies == {}

    def test_download_repo_file_sends_request_without_credentials(self, anon_api):
        resp = _mock_response(content=b"# hello")
        with patch.object(anon_api._openapi._session, "request", return_value=resp) as m:
            data = anon_api.download_repo_file("someone", "public-repo", "AGENTS.md", binary=True)
        assert data == b"# hello"
        auth, cookies = _sent_credentials(m)
        assert auth is None
        assert cookies == {}

    def test_repo_info_falls_back_to_public_probe_on_401(self, anon_api):
        """Server rejects anonymous /openapi metadata -> probe /api/v1 tree."""
        rejected = MagicMock(status_code=401, headers={})
        rejected.json.return_value = {
            "success": False,
            "code": "InvalidAuthentication",
            "message": "Invalid authentication: user not authenticated",
        }
        rejected.content = b'{"success": false}'
        tree_ok = _mock_response({"Code": 200, "Success": True, "Data": {"Trees": []}})
        with patch.object(anon_api._openapi._session, "request", side_effect=[rejected, tree_ok]) as m:
            info = anon_api.repo_info("someone", "public-repo")
        assert info == {}
        assert m.call_count == 2
        probe_url = m.call_args_list[1].kwargs.get("url") or m.call_args_list[1].args[1]
        assert "/api/v1/agents/someone/public-repo/repo/files" in probe_url

    def test_repo_info_fallback_returns_none_for_missing_repo(self, anon_api):
        rejected = MagicMock(status_code=401, headers={})
        rejected.json.return_value = {
            "success": False,
            "code": "InvalidAuthentication",
            "message": "user not authenticated",
        }
        rejected.content = b'{"success": false}'
        missing = MagicMock(status_code=404, headers={})
        missing.json.return_value = {"Code": 10025801007, "Message": "Agent不存在", "Success": False}
        missing.content = b'{"Success": false}'
        with patch.object(anon_api._openapi._session, "request", side_effect=[rejected, missing]):
            assert anon_api.repo_info("someone", "no-such-repo") is None

    def test_read_ops_attach_token_when_available(self, monkeypatch):
        """require_token=False must NOT strip credentials: with a token
        configured (private-repo scenario) read-only calls still send it."""
        monkeypatch.delenv("MODELSCOPE_API_TOKEN", raising=False)
        with (
            patch.object(HubConfig, "load_token", return_value=None),
            patch.object(HubConfig, "load_cookies", return_value=None),
        ):
            api = AgentApi(endpoint=ENDPOINT, token="ms-PRIVATE-TOKEN", timeout=5)
        resp = _mock_response(content=b"secret file")
        with patch.object(api._openapi._session, "request", return_value=resp) as m:
            api.download_repo_file("someone", "private-repo", "AGENTS.md", binary=True)
        auth, cookies = _sent_credentials(m)
        assert auth == "Bearer ms-PRIVATE-TOKEN"
        assert cookies.get("m_session_id") == "ms-PRIVATE-TOKEN"

    def test_write_ops_still_require_token(self, anon_api):
        """Uploads keep the local guard: no token -> immediate auth error."""
        with patch.object(anon_api._openapi._session, "request") as m:
            with pytest.raises(AuthenticationError):
                anon_api.create_repo("someone", "new-repo")
            with pytest.raises(AuthenticationError):
                anon_api.commit_files("someone", "repo", [])
        m.assert_not_called()


class TestGetCookiesNoSilentFallback:
    """HubApi.get_cookies must honor the explicit-empty-token override."""

    def _hub_api(self, monkeypatch, token):
        monkeypatch.delenv("MODELSCOPE_API_TOKEN", raising=False)
        with patch.object(HubConfig, "load_token", return_value=None):
            return HubApi(endpoint=ENDPOINT, token=token)

    def test_explicit_empty_token_does_not_load_cookies(self, monkeypatch):
        api = self._hub_api(monkeypatch, token="")
        jar = RequestsCookieJar()
        jar.set("m_session_id", STORED)
        with patch.object(HubConfig, "load_cookies", return_value=jar):
            assert api.get_cookies() is None
            with pytest.raises(AuthenticationError):
                api.get_cookies(cookies_required=True)

    def test_no_token_at_all_falls_back_to_persisted_cookies(self, monkeypatch):
        api = self._hub_api(monkeypatch, token=None)
        jar = RequestsCookieJar()
        jar.set("m_session_id", STORED)
        with patch.object(HubConfig, "load_cookies", return_value=jar):
            got = api.get_cookies()
        assert got is jar
