"""Mock-based contract tests for the Agent-IDP OpenAPI client and facade."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from modelscope_hub._openapi import OpenAPIClient
from modelscope_hub.api import HubApi
from modelscope_hub.config import HubConfig
from modelscope_hub.errors import InvalidParameter
from modelscope_hub.types import AgentIdentity, AgentToken


def _response(data: object) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = 200
    response.content = b"x"
    response.headers = {}
    response.request = MagicMock()
    response.request.method = "GET"
    response.request.path_url = "/test"
    response.request.url = "https://modelscope.cn/openapi/v1/test"
    response.url = response.request.url
    response.json.return_value = {"success": True, "data": data}
    return response


@pytest.fixture
def client() -> OpenAPIClient:
    return OpenAPIClient(HubConfig(token="test-token", endpoint="https://modelscope.cn"))


@pytest.fixture
def public_jwk() -> dict[str, str]:
    return {"kty": "OKP", "crv": "Ed25519", "x": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "kid": "key-1"}


class TestAgentIdpOpenApi:
    def test_create_identity_uses_openapi_payload(self, client, public_jwk):
        with patch.object(client._session, "request", return_value=_response({"agent_id": "agent-1"})) as request:
            client.create_agent_identity({"agent_name": "builder", "public_key": public_jwk, "token_expire_time": 600})
        call = request.call_args.kwargs
        assert call["method"] == "POST"
        assert call["url"].endswith("/openapi/v1/agent_ids")
        assert call["json"]["public_key"] == public_jwk
        assert call["headers"]["Authorization"] == "Bearer test-token"

    def test_update_and_key_reset_use_spec_methods(self, client, public_jwk):
        with patch.object(client._session, "request", side_effect=[_response({}), _response({})]) as request:
            client.update_agent_identity("agent-1", {"description": "updated"})
            client.reset_agent_key_pair("agent-1", {"public_key": public_jwk})
        update, reset = request.call_args_list
        assert update.kwargs["method"] == "PATCH"
        assert update.kwargs["url"].endswith("/agent_ids/agent-1")
        assert reset.kwargs["method"] == "PUT"
        assert reset.kwargs["url"].endswith("/agent_ids/agent-1/key_pairs")

    def test_delete_and_pause_use_spec_routes(self, client):
        with patch.object(client._session, "request", side_effect=[_response({}), _response({})]) as request:
            client.delete_agent_identity("agent-1")
            client.pause_agent("agent-1", {"paused": True})
        delete, pause = request.call_args_list
        assert delete.kwargs["method"] == "DELETE"
        assert delete.kwargs["url"].endswith("/agent_ids/agent-1")
        assert pause.kwargs["method"] == "POST"
        assert pause.kwargs["url"].endswith("/agent_ids/agent-1/paused")
        assert pause.kwargs["json"] == {"paused": True}

    def test_list_routes_use_spec_page_fields(self, client):
        with patch.object(client._session, "request", side_effect=[_response({}), _response({})]) as request:
            client.list_agent_token_records("agent-1", page=2, page_size=10)
            client.list_user_agent_identities("alice", status="paused", page=3, page_size=20)
        tokens, identities = request.call_args_list
        assert tokens.kwargs["url"].endswith("/agent_ids/agent-1/jwt_id_tokens")
        assert dict(tokens.kwargs["params"]) == {"page": "2", "page_size": "10"}
        assert identities.kwargs["url"].endswith("/users/alice/agent_ids")
        assert dict(identities.kwargs["params"]) == {"status": "paused", "page": "3", "page_size": "20"}

    @pytest.mark.parametrize(("page", "page_size"), [(0, 20), (1, 0), (1, 51)])
    def test_paging_is_validated(self, client, page, page_size):
        with pytest.raises(InvalidParameter):
            client.list_agent_token_records("agent-1", page=page, page_size=page_size)

    def test_update_and_public_jwk_are_strict(self, client, public_jwk):
        with pytest.raises(InvalidParameter, match="at least one"):
            client.update_agent_identity("agent-1", {})
        with pytest.raises(InvalidParameter, match="private"):
            client.create_agent_identity({"agent_name": "builder", "public_key": {**public_jwk, "d": "not-allowed"}})
        with pytest.raises(InvalidParameter, match="status"):
            client.list_user_agent_identities("alice", status="retired")

    def test_public_operations_never_attach_ambient_credentials(self, client):
        with patch.object(
            client._session,
            "request",
            side_effect=[_response({"access_token": "jwt"}), _response({}), _response({"keys": []})],
        ) as request:
            client.issue_agent_token(
                {
                    "agent_id": "agent-1",
                    "kid": "key-1",
                    "audience": "hub",
                    "timestamp": 1,
                    "signature": "signature",
                }
            )
            client.get_agent_id_configuration()
            client.get_agent_id_jwks()
        for call in request.call_args_list:
            assert "Authorization" not in call.kwargs["headers"]
            assert call.kwargs["cookies"] == {}
        assert request.call_args_list[0].kwargs["method"] == "POST"
        assert request.call_args_list[0].kwargs["url"].endswith("/agent_id/token")
        assert request.call_args_list[1].kwargs["url"].endswith("/agent_id/.well-known/agentid-configuration")
        assert request.call_args_list[2].kwargs["url"].endswith("/agent_id/.well-known/agentid-jwks")


class TestAgentIdpFacade:
    def test_facade_converts_identity_and_pagination(self):
        api = HubApi(token="test-token")
        api._openapi = MagicMock()
        api._openapi.create_agent_identity.return_value = {"agent_id": "agent-1", "agent_name": "builder"}
        api._openapi.list_user_agent_identities.return_value = {
            "agent_identities": [{"agent_id": "agent-1", "agent_name": "builder"}],
            "total_count": 1,
            "page_number": 1,
            "page_size": 20,
        }
        created = api.create_agent_identity({"agent_name": "builder", "public_key": {}})
        page = api.list_user_agent_identities("alice")
        assert isinstance(created, AgentIdentity)
        assert created.agent_id == "agent-1"
        assert page.total_count == 1
        assert page.items[0].agent_name == "builder"

    def test_facade_converts_issued_token(self):
        api = HubApi(token="test-token")
        api._openapi = MagicMock()
        api._openapi.issue_agent_token.return_value = {"access_token": "jwt", "token_type": "Bearer", "expire_at": 10}
        token = api.issue_agent_token(
            {"agent_id": "agent-1", "kid": "key-1", "audience": "hub", "timestamp": 1, "signature": "sig"}
        )
        assert isinstance(token, AgentToken)
        assert token.access_token == "jwt"
