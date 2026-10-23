"""Verify SDK UA, stable sessions and channel propagation using PreparedRequest, without remote access."""

from __future__ import annotations

import importlib
import io
from argparse import Namespace
from unittest.mock import Mock

import pytest
import requests
import responses

from modelscope_hub import HubApi, __version__
from modelscope_hub import config as config_module
from modelscope_hub._download_control import _DownloadControl
from modelscope_hub._legacy_api import LegacyClient
from modelscope_hub._openapi import OpenAPIClient
from modelscope_hub.agent import AgentApi
from modelscope_hub.cli.base import make_api
from modelscope_hub.cli.main import run_cmd
from modelscope_hub.compat import (
    dataset_file_download,
    dataset_snapshot_download,
    model_file_download,
    snapshot_download,
)
from modelscope_hub.compat.hub_api import LegacyHubApi
from modelscope_hub.config import HubConfig
from modelscope_hub.constants import DEFAULT_ENDPOINT as CN
from modelscope_hub.constants import DEFAULT_INTL_ENDPOINT as AI
from modelscope_hub.errors import CacheError
from modelscope_hub.utils import append_user_agent, build_user_agent

TOKEN = "ms-ua-test-dummy"
USER = {"success": True, "data": {"username": "alice"}}


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    for name in ("MODELSCOPE_ENDPOINT", "MODELSCOPE_DOMAIN", "MODELSCOPE_API_TOKEN", "MODELSCOPE_PREFER_AI_SITE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MODELSCOPE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(config_module, "_default_config", None)


@pytest.fixture
def config():
    return HubConfig(token=TOKEN)


def _assert_sdk(ua):
    assert ua.startswith(f"modelscope_hub/{__version__}; ")
    assert "session_id/" in ua
    assert "python/" in ua
    assert TOKEN not in ua
    return ua


def _session_id(ua):
    return next(part.removeprefix("session_id/") for part in ua.split("; ") if part.startswith("session_id/"))


@pytest.mark.parametrize("extra", [None, {"application": "example"}, "example-client/1"])
def test_build_user_agent_config_and_extra(config, extra):
    first = _assert_sdk(build_user_agent(config=config, extra=extra))
    assert first == build_user_agent(config=config, extra=extra)
    assert _session_id(first) == config.get_session_id()
    if extra:
        assert ("application/example" if isinstance(extra, dict) else extra) in first
    assert append_user_agent(first, extra) == first


@pytest.mark.parametrize("failure", [CacheError("read-only"), PermissionError("read-only"), OSError("read-only")])
@responses.activate
def test_readonly_home_keeps_requests_and_session_stable(config, monkeypatch, failure):
    monkeypatch.setattr(HubConfig, "get_session_id", Mock(side_effect=failure))
    responses.add(responses.GET, f"{CN}/openapi/v1/users/me", json=USER)
    api = HubApi(config=config)
    api.whoami()
    api.whoami()
    uas = [call.request.headers["User-Agent"] for call in responses.calls]
    uas.extend([api.legacy._user_agent, api.downloader._build_user_agent(), OpenAPIClient(config)._user_agent])
    assert len({_session_id(_assert_sdk(ua)) for ua in uas}) == 1
    assert not config.credentials_dir.exists()


def test_hubapi_constructor_remains_lazy(config, monkeypatch):
    lookup = Mock(side_effect=AssertionError("不应提前访问用户目录"))
    monkeypatch.setattr(HubConfig, "get_session_id", lookup)
    HubApi(config=config, user_agent={"application": "example"})
    lookup.assert_not_called()


@pytest.mark.parametrize("read_session", [False, True])
@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "DELETE"])
@responses.activate
def test_all_http_methods_and_both_sessions(config, method, read_session):
    url = f"{CN}/openapi/v1/ua-test"
    responses.add(method, url, json={"success": True, "data": {}})
    with OpenAPIClient(config, user_agent={"application": "example"}) as client:
        if read_session:
            with _DownloadControl(total_timeout=5) as control, client.read_context(control):
                client.request(method, "/ua-test")
        else:
            client.request(method, "/ua-test")
    ua = _assert_sdk(responses.calls[0].request.headers["User-Agent"])
    assert "application/example" in ua
    assert "python-requests/" not in ua


@pytest.mark.parametrize("name", ["User-Agent", "user-agent", "USER-AGENT"])
@pytest.mark.parametrize("value", [None, "", "other-client/2"])
@responses.activate
def test_custom_header_is_appended_case_insensitively(config, name, value):
    responses.add(responses.GET, f"{CN}/openapi/v1/users/me", json=USER)
    headers = {name: value, "X-Example": "preserved"}
    OpenAPIClient(config).request("GET", "/users/me", headers=headers)
    prepared = responses.calls[0].request.headers
    ua = _assert_sdk(prepared["User-Agent"])
    assert sum(key.lower() == "user-agent" for key in prepared) == 1
    assert prepared["X-Example"] == "preserved"
    assert headers == {name: value, "X-Example": "preserved"}
    if value:
        assert value in ua


@pytest.mark.parametrize("session_ua", [None, "", "custom-session/1", requests.utils.default_user_agent()])
@responses.activate
def test_injected_session_has_sdk_ua_without_mutating_headers(config, session_ua):
    session = requests.Session()
    if session_ua is None:
        session.headers.pop("User-Agent", None)
    else:
        session.headers["User-Agent"] = session_ua
    original = dict(session.headers)
    responses.add(responses.GET, f"{CN}/openapi/v1/users/me", json=USER)
    with OpenAPIClient(config, session=session) as client:
        client.get_current_user()
        with _DownloadControl(total_timeout=5) as control, client.read_context(control):
            client.get_current_user()
    uas = [_assert_sdk(call.request.headers["User-Agent"]) for call in responses.calls]
    assert uas[0] == uas[1]
    assert dict(session.headers) == original
    if session_ua == "custom-session/1":
        assert session_ua in uas[0]


@pytest.mark.parametrize("status", [429, 503])
@responses.activate
def test_retry_ua_is_stable(config, monkeypatch, status):
    sleeps = Mock()
    monkeypatch.setattr("modelscope_hub._openapi.time.sleep", sleeps)
    url = f"{CN}/openapi/v1/users/me"
    responses.add(responses.GET, url, json={"message": "retry"}, status=status, headers={"Retry-After": "2"})
    responses.add(responses.GET, url, json=USER)
    with OpenAPIClient(config, max_retries=2) as client:
        client.get_current_user()
    assert len(responses.calls) == 2
    assert (
        _assert_sdk(responses.calls[0].request.headers["User-Agent"])
        == responses.calls[1].request.headers["User-Agent"]
    )
    sleeps.assert_called_once()
    if status == 429:
        sleeps.assert_called_with(2.0)


@responses.activate
def test_anonymous_fallback_keeps_ua_and_drops_auth(config):
    url = f"{CN}/openapi/v1/studios"
    responses.add(responses.GET, url, json={"message": "forbidden"}, status=403)
    responses.add(responses.GET, url, json={"studios": []})
    OpenAPIClient(config).list_studios()
    first, second = [call.request.headers for call in responses.calls]
    assert _assert_sdk(first["User-Agent"]) == second["User-Agent"]
    assert "Authorization" in first and "Cookie" in first
    assert "Authorization" not in second and "Cookie" not in second


@pytest.mark.parametrize("path", ["/agent_id/.well-known/agentid-configuration", "/agent_id/.well-known/agentid-jwks"])
@responses.activate
def test_anonymous_idp_metadata_has_ua(config, path):
    responses.add(responses.GET, f"{CN}/openapi/v1{path}", json={"data": {}})
    client = OpenAPIClient(config)
    if path.endswith("agentid-jwks"):
        client.get_agent_id_jwks()
    else:
        client.get_agent_id_configuration()
    headers = responses.calls[0].request.headers
    _assert_sdk(headers["User-Agent"])
    assert "Authorization" not in headers and "Cookie" not in headers


@responses.activate
def test_multipart_absolute_and_raw_response_have_ua(config):
    responses.add(responses.POST, f"{CN}/openapi/v1/files/upload", json={"data": {"id": "f"}})
    responses.add(responses.PUT, "https://storage.example.test/blob", status=204)
    with OpenAPIClient(config) as client:
        client.upload_file(file=io.BytesIO(b"x"))
        response = client.request("PUT", url="https://storage.example.test/blob", data=b"x", unwrap=False)
    assert response.status_code == 204
    assert "multipart/form-data" in responses.calls[0].request.headers["Content-Type"]
    for call in responses.calls:
        _assert_sdk(call.request.headers["User-Agent"])
    assert "Authorization" not in responses.calls[1].request.headers
    assert "Cookie" not in responses.calls[1].request.headers


@responses.activate
def test_login_and_scoped_probe_share_channel_and_session(config):
    responses.add(responses.POST, f"{CN}/api/v1/login", json={"message": "forbidden"}, status=403)
    responses.add(responses.GET, f"{CN}/openapi/v1/users/me", json=USER)
    assert HubApi(config=config, user_agent={"invoked_by": "cli"}).login(TOKEN).username == "alice"
    assert len(responses.calls) == 2
    uas = [_assert_sdk(call.request.headers["User-Agent"]) for call in responses.calls]
    assert uas[0] == uas[1]
    assert "invoked_by/cli" in uas[0]


@responses.activate
def test_peer_login_probe_has_sdk_ua():
    responses.add(responses.POST, f"{AI}/api/v1/login", json={"Code": 200, "Data": {"Username": "alice"}})
    assert HubApi._token_valid_on(TOKEN, AI)
    _assert_sdk(responses.calls[0].request.headers["User-Agent"])


@responses.activate
def test_read_site_probes_keep_custom_config_channel_and_session(config):
    responses.add(responses.GET, f"{CN}/openapi/v1/models/alice/m", status=404)
    responses.add(responses.GET, f"{CN}/api/v1/models/alice/m", status=404)
    responses.add(responses.GET, f"{AI}/openapi/v1/models/alice/m", json={"data": {"name": "m"}})
    api = HubApi(config=config, user_agent={"invoked_by": "compat", "application": "example"})
    assert api.resolve_endpoint_for_read("alice/m") == AI
    assert len(responses.calls) == 3
    uas = [_assert_sdk(call.request.headers["User-Agent"]) for call in responses.calls]
    assert len(set(uas)) == 1
    assert _session_id(uas[0]) == config.get_session_id()
    assert "invoked_by/compat" in uas[0] and "application/example" in uas[0]


@responses.activate
def test_agent_openapi_legacy_and_blob_have_sdk_ua(config):
    responses.add(responses.GET, f"{CN}/openapi/v1/agents/alice/a", json={"data": {}})
    responses.add(responses.GET, f"{CN}/api/v1/agents/alice/a/repo/files", json={"data": {"Trees": []}})
    responses.add(responses.PUT, "https://storage.example.test/blob", status=204)
    api = AgentApi(config=config, user_agent={"invoked_by": "cli"})
    api.repo_info("alice", "a")
    api.list_repo_files("alice", "a")
    api.lfs_upload_blob("https://storage.example.test/blob", b"x")
    for call in responses.calls:
        assert "invoked_by/cli" in _assert_sdk(call.request.headers["User-Agent"])


@pytest.mark.parametrize("read_session", [False, True])
@responses.activate
def test_legacy_default_custom_headers_and_blob(config, read_session):
    responses.add(responses.GET, f"{CN}/api/v1/models/alice/m", json={"Data": {}})
    responses.add(responses.PUT, f"{CN}/blob", json={})
    client = LegacyClient(token=TOKEN, endpoint=CN)
    if read_session:
        with _DownloadControl(total_timeout=5) as control, client.read_context(control):
            client._request("GET", "models/alice/m", headers={"user-agent": "custom-read/1"})
    else:
        client._request("GET", "models/alice/m", headers={"USER-AGENT": "custom-read/1"})
    client.upload_blob(f"{CN}/blob", b"x", size=1, headers={"user-agent": "custom-blob/1"})
    assert "custom-read/1" in _assert_sdk(responses.calls[0].request.headers["User-Agent"])
    assert "custom-blob/1" in _assert_sdk(responses.calls[1].request.headers["User-Agent"])


@responses.activate
def test_cli_whoami_and_explicit_download_endpoint_have_channel():
    responses.add(responses.GET, f"{CN}/openapi/v1/users/me", json=USER)
    assert run_cmd(["--token", TOKEN, "whoami"]) == 0
    assert "invoked_by/cli" in _assert_sdk(responses.calls[0].request.headers["User-Agent"])
    from modelscope_hub.cli.download import DownloadCommand

    command = DownloadCommand(Namespace(endpoint=CN, token=TOKEN))
    assert "invoked_by/cli" in _assert_sdk(command._make_api_with_endpoint().openapi._user_agent)
    assert "invoked_by/cli" in make_api(Namespace(token=TOKEN)).legacy._user_agent


@responses.activate
def test_cli_agent_username_and_repo_calls_have_channel(monkeypatch):
    cli_agent = importlib.import_module("modelscope_hub.cli.agent")
    responses.add(responses.GET, f"{CN}/openapi/v1/users/me", json=USER)
    responses.add(responses.PUT, f"{CN}/api/v1/dolphin/agents", json={"Data": {"AgentList": []}})

    def list_after_username(**kwargs):
        return cli_agent._cmd_list(owner="alice", page_number=1, page_size=10, endpoint=CN, token=TOKEN)

    monkeypatch.setattr(cli_agent, "_cmd_upload", list_after_username)
    assert run_cmd(["--token", TOKEN, "agent", "upload", "--repo", "alice/a", "--local-dir", "."]) == 0
    assert responses.calls
    for call in responses.calls:
        assert "invoked_by/cli" in _assert_sdk(call.request.headers["User-Agent"])


@responses.activate
def test_compat_default_and_override_clients_have_channel():
    responses.add(responses.GET, f"{CN}/openapi/v1/models/alice/m", json={"data": {"name": "m"}})
    api = LegacyHubApi(endpoint=CN, token=TOKEN)
    api.repo_exists("alice/m", repo_type="model")
    api.repo_exists("alice/m", repo_type="model", token="other-test-token")
    for call in responses.calls:
        assert "invoked_by/compat" in _assert_sdk(call.request.headers["User-Agent"])


@pytest.mark.parametrize(
    "download", [model_file_download, dataset_file_download, snapshot_download, dataset_snapshot_download]
)
def test_compat_download_user_agent_applies_to_all_transports(download, monkeypatch):
    seen = []

    def fake_download(self, *args, **kwargs):
        ua = self.openapi._user_agent
        assert "invoked_by/compat" in _assert_sdk(ua) and "application/example" in ua
        assert self.legacy._user_agent == ua
        assert self.downloader._build_user_agent(kwargs["user_agent"]) == ua
        seen.append(ua)
        return "local-path"

    monkeypatch.setattr(HubApi, "download_file", fake_download)
    monkeypatch.setattr(HubApi, "download_repo", fake_download)
    kwargs = {"local_files_only": True, "user_agent": {"application": "example"}}
    if download in (model_file_download, dataset_file_download):
        assert download("alice/m", "x", **kwargs) == "local-path"
    else:
        assert download("alice/m", **kwargs) == "local-path"
    assert len(seen) == 1
