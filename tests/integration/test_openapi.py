"""Integration tests for the OpenAPI client.

These tests verify that the OpenAPIClient correctly communicates with
the ModelScope Hub OpenAPI surface. They exercise the client directly,
independent of the HubApi facade.

Requires MODELSCOPE_TEST_TOKEN and MODELSCOPE_TEST_OWNER in tests/.env.
"""

from __future__ import annotations

import contextlib
import os
import uuid

import pytest

from modelscope_hub._openapi import OpenAPIClient
from modelscope_hub.api import HubApi
from modelscope_hub.config import HubConfig
from modelscope_hub.errors import (
    AlreadyExistsError,
    AuthenticationError,
    InvalidParameter,
    PermissionDeniedError,
)


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
        print(f"Datasets result:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
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

    def test_list_studios(self, openapi):
        result = openapi.list_studios(page_size=5)
        assert isinstance(result, dict)
        studios = result.get("studios") or []
        assert isinstance(studios, list)
        assert len(studios) <= 5

    def test_list_studios_with_search(self, openapi):
        result = openapi.list_studios(search="chat", page_size=3)
        assert isinstance(result, dict)

    def test_list_studios_by_owner(self, openapi, test_owner):
        result = openapi.list_studios(owner=test_owner, page_size=5)
        for studio in result.get("studios") or []:
            assert (studio.get("owner") or "").lower() == test_owner.lower()

    def test_list_studios_rejects_unknown_sort(self, openapi):
        with pytest.raises(InvalidParameter):
            openapi.list_studios(sort="downloads")


@pytest.mark.remote
class TestOpenAPIStudioResources:
    """Resource-discovery endpoints backing --hardware / --base-image / --sdk-version."""

    def test_list_hardware(self, openapi):
        result = openapi.list_studio_hardware()
        hardware = (result or {}).get("hardware")
        assert isinstance(hardware, list)
        if hardware:
            assert "name" in hardware[0]

    def test_list_hardware_filtered_by_sdk_type(self, openapi):
        result = openapi.list_studio_hardware(sdk_type="gradio")
        assert isinstance((result or {}).get("hardware"), list)

    def test_list_base_images(self, openapi):
        result = openapi.list_studio_base_images()
        images = (result or {}).get("base_images")
        assert isinstance(images, list)
        if images:
            assert "name" in images[0]

    def test_gradio_publishes_sdk_versions(self, openapi):
        result = openapi.list_studio_sdk_versions(sdk_type="gradio")
        versions = (result or {}).get("sdk_versions")
        assert isinstance(versions, list)

    def test_non_gradio_sdk_has_no_versions(self, openapi):
        """The specification states only gradio publishes a version list."""
        result = openapi.list_studio_sdk_versions(sdk_type="docker")
        assert not ((result or {}).get("sdk_versions") or [])


@pytest.mark.remote
class TestOpenAPIStudioVariables:
    """Full add -> list -> update -> delete cycle against a real Studio.

    Requires MODELSCOPE_TEST_STUDIO (``owner/repo_name``) in tests/.env, because
    the endpoints operate on an existing space. The key is randomised so parallel
    runs cannot collide, and cleanup runs even when an assertion fails.
    """

    @pytest.fixture
    def studio_id(self) -> str:
        studio = os.environ.get("MODELSCOPE_TEST_STUDIO")
        if not studio:
            pytest.skip("Set MODELSCOPE_TEST_STUDIO=owner/repo_name to exercise Studio variables")
        return studio

    @pytest.fixture
    def variable_key(self, openapi, studio_id):
        owner, repo_name = studio_id.split("/", 1)
        key = f"MSHUB_TEST_{uuid.uuid4().hex[:8].upper()}"
        yield key
        with contextlib.suppress(Exception):
            openapi.delete_studio_variable(owner, repo_name, key)

    def test_variable_lifecycle(self, openapi, studio_id, variable_key):
        owner, repo_name = studio_id.split("/", 1)

        openapi.add_studio_variable(owner, repo_name, variable_key, "first")
        listed = openapi.list_studio_variables(owner, repo_name)
        entries = {v["key"]: v.get("value") for v in (listed or {}).get("variables") or []}
        assert entries.get(variable_key) == "first"

        openapi.update_studio_variable(owner, repo_name, variable_key, "second")
        listed = openapi.list_studio_variables(owner, repo_name)
        entries = {v["key"]: v.get("value") for v in (listed or {}).get("variables") or []}
        assert entries.get(variable_key) == "second"

        openapi.delete_studio_variable(owner, repo_name, variable_key)
        listed = openapi.list_studio_variables(owner, repo_name)
        remaining = {v["key"] for v in (listed or {}).get("variables") or []}
        assert variable_key not in remaining

    def test_adding_a_duplicate_reports_already_exists(self, openapi, studio_id, variable_key):
        """The endpoint answers 409, which must surface as AlreadyExistsError."""
        owner, repo_name = studio_id.split("/", 1)
        openapi.add_studio_variable(owner, repo_name, variable_key, "first")
        with pytest.raises(AlreadyExistsError):
            openapi.add_studio_variable(owner, repo_name, variable_key, "again")


@pytest.mark.remote
class TestOpenAPIMcp:
    """MCP discovery and the caller's own hosted servers."""

    def test_list_mcp_servers(self, openapi):
        result = openapi.list_mcp_servers(page_size=5)
        assert isinstance(result, dict)
        assert isinstance(result.get("mcp_server_list"), list)

    def test_list_mcp_servers_rejects_excessive_offset(self, openapi):
        with pytest.raises(InvalidParameter):
            openapi.list_mcp_servers(page_number=11, page_size=10)

    def test_list_operational_mcp_servers(self, openapi):
        result = openapi.list_operational_mcp_servers()
        assert isinstance(result, dict)
        assert isinstance(result.get("mcp_server_list"), list)


@pytest.mark.remote
class TestOpenAPIReadOnlyToken:
    """Behaviour under a read-scoped token.

    Set MODELSCOPE_TEST_READONLY_TOKEN in tests/.env to a token issued with read
    permission only. This is the only way to verify the permission-tier handling
    end to end: the spec models no scopes, so it can only be observed at runtime.
    """

    @pytest.fixture
    def readonly_config(self, test_endpoint):
        token = os.environ.get("MODELSCOPE_TEST_READONLY_TOKEN")
        if not token:
            pytest.skip("Set MODELSCOPE_TEST_READONLY_TOKEN to exercise read-scoped behaviour")
        return HubConfig(token=token, endpoint=test_endpoint)

    def test_read_operations_succeed(self, readonly_config):
        client = OpenAPIClient(readonly_config)
        try:
            result = client.list_studios(page_size=3)
            assert isinstance(result, dict)
        finally:
            client.close()

    def test_login_succeeds_with_reduced_capability(self, readonly_config, tmp_path, monkeypatch):
        monkeypatch.setenv("MODELSCOPE_HOME", str(tmp_path))
        api = HubApi(config=HubConfig(config_dir=tmp_path, endpoint=readonly_config.endpoint))
        user = api.login(readonly_config.token)
        assert user.username

    def test_write_operation_names_the_missing_permission(self, readonly_config, test_owner):
        client = OpenAPIClient(readonly_config)
        try:
            with pytest.raises((PermissionDeniedError, AuthenticationError)) as excinfo:
                client.create_studio({"owner": test_owner, "repo_name": f"mshub-perm-{uuid.uuid4().hex[:8]}"})
        finally:
            client.close()
        if isinstance(excinfo.value, PermissionDeniedError):
            assert "write" in excinfo.value.suggestion


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
