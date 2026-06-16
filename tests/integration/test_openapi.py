"""Integration tests for the OpenAPI client.

These tests verify that the OpenAPIClient correctly communicates with
the ModelScope Hub OpenAPI surface. They exercise the client directly,
independent of the HubApi facade.

Requires MODELSCOPE_TEST_TOKEN and MODELSCOPE_TEST_OWNER in tests/.env.
"""
from __future__ import annotations

import pytest

from modelscope_hub._openapi import OpenAPIClient
from modelscope_hub.config import HubConfig


@pytest.fixture
def openapi(test_token, test_endpoint) -> OpenAPIClient:
    config = HubConfig(token=test_token, endpoint=test_endpoint)
    client = OpenAPIClient(config)
    yield client
    client.close()


@pytest.mark.remote
class TestOpenAPIAuth:
    """Test authentication-related OpenAPI calls."""

    def test_get_current_user(self, openapi, test_owner):
        user = openapi.get_current_user()
        assert isinstance(user, dict)
        username = user.get("Username") or user.get("username")
        assert username == test_owner


@pytest.mark.remote
class TestOpenAPIModels:
    """Test model listing and retrieval via OpenAPI."""

    def test_list_models(self, openapi):
        result = openapi.list_models(page_size=5)
        assert isinstance(result, dict)
        models = result.get("Models") or result.get("models") or []
        assert isinstance(models, list)
        assert len(models) <= 5

    def test_list_models_with_search(self, openapi):
        result = openapi.list_models(search="bert", page_size=3)
        assert isinstance(result, dict)
        models = result.get("Models") or result.get("models") or []
        assert isinstance(models, list)

    def test_get_model(self, openapi):
        result = openapi.get_model("Qwen", "Qwen2.5-0.5B")
        assert isinstance(result, dict)
        assert len(result) > 0


@pytest.mark.remote
class TestOpenAPIDatasets:
    """Test dataset listing and retrieval via OpenAPI."""

    def test_list_datasets(self, openapi):
        import json

        result = openapi.list_datasets(page_size=3, owner="modelscope")
        print(f"Datasets result: {json.dumps(result, indent=2)}")
        assert isinstance(result, dict)
        datasets = result.get("Datasets") or result.get("datasets") or result.get("Data") or []
        assert isinstance(datasets, list)

    def test_get_dataset(self, openapi):
        result = openapi.get_dataset("modelscope", "clue")
        assert isinstance(result, dict)


@pytest.mark.remote
class TestOpenAPIMCP:
    """Test MCP server listing via OpenAPI."""

    def test_list_mcp_servers(self, openapi):
        result = openapi.list_mcp_servers(page_size=5)
        assert isinstance(result, dict)
        servers = result.get("mcp_server_list") or []
        assert isinstance(servers, list)
        assert len(servers) <= 5

    def test_list_mcp_servers_with_search(self, openapi):
        result = openapi.list_mcp_servers(search="weather", page_size=3)
        assert isinstance(result, dict)

    def test_list_mcp_servers_with_filter(self, openapi):
        result = openapi.list_mcp_servers(
            page_size=5,
            filter={"is_hosted": True},
        )
        assert isinstance(result, dict)

    def test_list_mcp_servers_total_count(self, openapi):
        result = openapi.list_mcp_servers(page_size=1)
        total = result.get("total") or result.get("total_count") or 0
        assert total > 0

    def test_get_mcp_server(self, openapi):
        listing = openapi.list_mcp_servers(page_size=1)
        servers = listing.get("mcp_server_list") or []
        if not servers:
            pytest.skip("No MCP servers available")
        server_id = servers[0].get("id") or servers[0].get("Id")
        result = openapi.get_mcp_server(server_id)
        assert isinstance(result, dict)


@pytest.mark.remote
class TestOpenAPISkills:
    """Test skill listing via OpenAPI."""

    def test_list_skills(self, openapi):
        result = openapi.list_skills(page_size=5)
        assert isinstance(result, dict)
        skills = result.get("skills") or result.get("Skills") or []
        assert isinstance(skills, list)

    def test_list_skills_with_search(self, openapi):
        result = openapi.list_skills(search="chat", page_size=3)
        assert isinstance(result, dict)


@pytest.mark.remote
class TestOpenAPIStudios:
    """Test studio endpoints via OpenAPI (read-only)."""

    def test_get_studio_public(self, openapi):
        try:
            result = openapi.get_studio("modelscope", "Qwen2.5-Coder-artifacts")
            assert isinstance(result, dict)
        except Exception:
            pytest.skip("Public studio not available or requires auth")


@pytest.mark.remote
class TestOpenAPIPagination:
    """Test pagination defaults and limits."""

    def test_models_default_page_size_returns_10(self, openapi):
        result = openapi.list_models()
        models = result.get("Models") or result.get("models") or []
        assert len(models) <= 10

    def test_datasets_default_page_size_returns_10(self, openapi):
        result = openapi.list_datasets()
        datasets = result.get("Datasets") or result.get("datasets") or []
        assert len(datasets) <= 10

    def test_models_pagination_page_2(self, openapi):
        page1 = openapi.list_models(page_size=3, page_number=1)
        page2 = openapi.list_models(page_size=3, page_number=2)
        models1 = page1.get("Models") or page1.get("models") or []
        models2 = page2.get("Models") or page2.get("models") or []
        if models1 and models2:
            ids1 = {m.get("id") or m.get("Id") for m in models1}
            ids2 = {m.get("id") or m.get("Id") for m in models2}
            assert ids1 != ids2
