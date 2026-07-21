"""Unit tests for the OpenAPIClient (mock-based, no network).

Covers fixes from audit items 2,3,4,5,6,8,10 and Section III risks.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from modelscope_hub._openapi import OpenAPIClient, _RETRYABLE_POST_PATHS
from modelscope_hub.api import HubApi
from modelscope_hub.config import HubConfig
from modelscope_hub.errors import InvalidParameter, ServerError


@pytest.fixture
def config():
    return HubConfig(token="test-token", endpoint="https://modelscope.cn")


@pytest.fixture
def client(config):
    return OpenAPIClient(config)


def _mock_response(status_code=200, json_data=None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.content = b'{"data": {}}' if json_data is None else b"x"
    resp.headers = {}
    resp.request = MagicMock()
    resp.request.method = "GET"
    resp.request.path_url = "/test"
    resp.request.url = "https://modelscope.cn/test"
    resp.url = "https://modelscope.cn/test"
    if json_data is not None:
        resp.json.return_value = json_data
        resp.content = b"x"
    else:
        resp.json.return_value = {"success": True, "data": {}}
    return resp


# ==================================================================
# Item 2: list_mcp_servers filter param
# ==================================================================
class TestListMcpServersFilter:
    def test_filter_included_in_body(self, client):
        resp = _mock_response(json_data={"success": True, "data": {"mcp_server_list": [], "total": 0}})
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.list_mcp_servers(filter={"category": "tools", "is_hosted": True})
        call_kwargs = mock_req.call_args.kwargs
        body = call_kwargs["json"]
        assert body["filter"] == {"category": "tools", "is_hosted": True}

    def test_filter_none_not_in_body(self, client):
        resp = _mock_response(json_data={"success": True, "data": {"mcp_server_list": [], "total": 0}})
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.list_mcp_servers()
        call_kwargs = mock_req.call_args.kwargs
        body = call_kwargs["json"]
        assert "filter" not in body


# ==================================================================
# Item 2 + Section III: MCP pagination limit
# ==================================================================
class TestMcpPaginationLimit:
    def test_within_limit(self, client):
        resp = _mock_response(json_data={"success": True, "data": {"mcp_server_list": [], "total": 0}})
        with patch.object(client._session, "request", return_value=resp):
            client.list_mcp_servers(page_number=5, page_size=20)

    def test_exceeds_limit(self, client):
        with pytest.raises(InvalidParameter, match="<= 100"):
            client.list_mcp_servers(page_number=11, page_size=10)

    def test_at_boundary(self, client):
        resp = _mock_response(json_data={"success": True, "data": {"mcp_server_list": [], "total": 0}})
        with patch.object(client._session, "request", return_value=resp):
            client.list_mcp_servers(page_number=10, page_size=10)


# ==================================================================
# Item 3: deploy_studio no empty body
# ==================================================================
class TestDeployStudioBody:
    def test_no_payload_sends_none(self, client):
        resp = _mock_response()
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.deploy_studio("org", "demo")
        call_kwargs = mock_req.call_args.kwargs
        assert call_kwargs["json"] is None

    def test_with_payload_sends_dict(self, client):
        resp = _mock_response()
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.deploy_studio("org", "demo", payload={"instance_count": 2})
        call_kwargs = mock_req.call_args.kwargs
        assert call_kwargs["json"] == {"instance_count": 2}


# ==================================================================
# Item 4: _extract_paged MCP passthrough
# ==================================================================
class TestExtractPagedMcp:
    def test_mcp_server_list_key_recognized(self):
        payload = {
            "mcp_server_list": [{"id": 1}, {"id": 2}],
            "total": 50,
        }
        items, total, page, size = HubApi._extract_paged(payload)
        assert len(items) == 2
        assert total == 50
        assert page == 1
        assert size == 2  # fallback since response has no page_size

    def test_list_repos_mcp_overrides_page_size(self):
        api = HubApi(config=HubConfig(token="t", endpoint="https://modelscope.cn"))
        mcp_response = {
            "mcp_server_list": [{"id": "1", "name": "test"}],
            "total": 30,
        }
        with patch.object(api.openapi, "list_mcp_servers", return_value=mcp_response):
            result = api.list_repos("mcp", page_number=2, page_size=10)
        assert result.page_number == 2
        assert result.page_size == 10
        assert result.total_count == 30


# ==================================================================
# Item 5: stop_studio no body
# ==================================================================
class TestStopStudioBody:
    def test_no_json_body_sent(self, client):
        resp = _mock_response()
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.stop_studio("org", "demo")
        call_kwargs = mock_req.call_args.kwargs
        assert call_kwargs["json"] is None


# ==================================================================
# Item 6: get_studio requires token
# ==================================================================
class TestGetStudioAuth:
    def test_requires_token_raises_without_token(self):
        from modelscope_hub.errors import AuthenticationError
        config = HubConfig(token="placeholder", endpoint="https://modelscope.cn")
        config.token = None
        client = OpenAPIClient(config)
        with patch.object(HubConfig, "load_token", return_value=None):
            with pytest.raises(AuthenticationError, match="Missing API token"):
                client.get_studio("org", "demo")

    def test_sends_auth_header_when_token_present(self, client):
        resp = _mock_response()
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.get_studio("org", "demo")
        call_kwargs = mock_req.call_args.kwargs
        assert "Authorization" in call_kwargs["headers"]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-token"


# ==================================================================
# Item 8: page_size defaults
# ==================================================================
class TestPageSizeDefaults:
    def test_list_models_default_page_size(self, client):
        resp = _mock_response(json_data={"success": True, "data": {"models": [], "total_count": 0}})
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.list_models()
        call_kwargs = mock_req.call_args.kwargs
        params = call_kwargs["params"]
        param_dict = dict(params) if isinstance(params, list) else params
        assert param_dict.get("page_size") == "10"

    def test_list_datasets_default_page_size(self, client):
        resp = _mock_response(json_data={"success": True, "data": {"datasets": [], "total_count": 0}})
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.list_datasets()
        call_kwargs = mock_req.call_args.kwargs
        params = call_kwargs["params"]
        param_dict = dict(params) if isinstance(params, list) else params
        assert param_dict.get("page_size") == "10"

    def test_list_skills_default_page_size(self, client):
        resp = _mock_response(json_data={"success": True, "data": {"skills": [], "total_count": 0}})
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.list_skills()
        call_kwargs = mock_req.call_args.kwargs
        params = call_kwargs["params"]
        param_dict = dict(params) if isinstance(params, list) else params
        assert param_dict.get("page_size") == "10"

    def test_list_mcp_servers_default_page_size(self, client):
        resp = _mock_response(json_data={"success": True, "data": {"mcp_server_list": [], "total": 0}})
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.list_mcp_servers()
        call_kwargs = mock_req.call_args.kwargs
        body = call_kwargs["json"]
        assert body["page_size"] == 10


# ==================================================================
# Section III: models/datasets pagination limit
# ==================================================================
class TestModelDatasetPaginationLimit:
    def test_list_models_exceeds_limit(self, client):
        with pytest.raises(InvalidParameter, match="<= 3000"):
            client.list_models(page_number=61, page_size=50)

    def test_list_datasets_exceeds_limit(self, client):
        with pytest.raises(InvalidParameter, match="<= 3000"):
            client.list_datasets(page_number=61, page_size=50)

    def test_list_skills_exceeds_limit(self, client):
        with pytest.raises(InvalidParameter, match="<= 3000"):
            client.list_skills(page_number=61, page_size=50)

    def test_list_models_at_boundary(self, client):
        resp = _mock_response(json_data={"success": True, "data": {"models": [], "total_count": 0}})
        with patch.object(client._session, "request", return_value=resp):
            client.list_models(page_number=60, page_size=50)

    def test_list_models_page1_large_size_ok(self, client):
        resp = _mock_response(json_data={"success": True, "data": {"models": [], "total_count": 0}})
        with patch.object(client._session, "request", return_value=resp):
            client.list_models(page_number=1, page_size=50)


# ==================================================================
# Item 10: retry logic for idempotent POSTs
# ==================================================================
class TestRetryIdempotentPost:
    def test_retryable_post_paths_defined(self):
        assert "/deploy" in _RETRYABLE_POST_PATHS
        assert "/stop" in _RETRYABLE_POST_PATHS
        assert "/undeploy" in _RETRYABLE_POST_PATHS

    def test_deploy_studio_retried_on_server_error(self, client):
        error_resp = _mock_response(status_code=500, json_data={"message": "Internal error"})
        success_resp = _mock_response(status_code=200, json_data={"success": True, "data": {"status": "deploying"}})
        with patch.object(client._session, "request", side_effect=[error_resp, success_resp]) as mock_req:
            result = client.deploy_studio("org", "demo")
        assert mock_req.call_count == 2

    def test_stop_studio_retried_on_server_error(self, client):
        error_resp = _mock_response(status_code=500, json_data={"message": "Internal error"})
        success_resp = _mock_response(status_code=200, json_data={"success": True, "data": {"status": "stopped"}})
        with patch.object(client._session, "request", side_effect=[error_resp, success_resp]) as mock_req:
            result = client.stop_studio("org", "demo")
        assert mock_req.call_count == 2

    def test_create_skill_not_retried(self, client):
        error_resp = _mock_response(status_code=500, json_data={"message": "Internal error"})
        with patch.object(client._session, "request", return_value=error_resp) as mock_req:
            with pytest.raises(ServerError):
                client.create_skill({"owner": "org", "skill_name": "test"})
        assert mock_req.call_count == 1

    def test_deploy_mcp_server_retried(self, client):
        error_resp = _mock_response(status_code=500, json_data={"message": "Internal error"})
        success_resp = _mock_response(status_code=200, json_data={"success": True, "data": {"status": "running"}})
        with patch.object(client._session, "request", side_effect=[error_resp, success_resp]) as mock_req:
            result = client.deploy_mcp_server("123")
        assert mock_req.call_count == 2


# ==================================================================
# Credential isolation: never leak token to a foreign host via an
# absolute URL (e.g. signed OSS blob-upload URLs from the LFS batch API).
# ==================================================================
class TestForeignHostCredentialIsolation:
    def test_same_host_absolute_url_gets_auth(self, client):
        resp = _mock_response()
        url = "https://modelscope.cn/api/v1/repos/agents/o/r/commit/master"
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.request("POST", url=url, json_body={})
        call_kwargs = mock_req.call_args.kwargs
        assert call_kwargs["headers"].get("Authorization") == "Bearer test-token"
        assert call_kwargs["cookies"] == {
            "m_session_id": "test-token", "modelscope_session": "test-token"}

    def test_foreign_host_absolute_url_strips_auth_and_cookies(self, client):
        resp = _mock_response()
        url = "https://oss-cn-hangzhou.aliyuncs.com/bucket/obj?sig=abc"
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.request(
                "PUT", url=url, data=b"blob",
                headers={"Content-Type": "application/octet-stream"},
                require_token=False, unwrap=False)
        call_kwargs = mock_req.call_args.kwargs
        assert "Authorization" not in call_kwargs["headers"]
        assert call_kwargs["cookies"] == {}
        # Caller-supplied headers (not credentials) must still be sent.
        assert call_kwargs["headers"]["Content-Type"] == "application/octet-stream"
