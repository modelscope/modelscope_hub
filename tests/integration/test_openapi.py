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
        result = openapi.list_datasets(page_size=3)
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

    def test_list_mcp_servers_with_search(self, openapi):
        result = openapi.list_mcp_servers(search="weather", page_size=3)
        assert isinstance(result, dict)


@pytest.mark.remote
class TestOpenAPISkills:
    """Test skill listing via OpenAPI."""

    def test_list_skills(self, openapi):
        result = openapi.list_skills(page_size=5)
        assert isinstance(result, dict)
