"""Wire-level tests for the Studio endpoints added in this cycle.

Every assertion pins what actually reaches the transport -- method, URL, query
string and body -- because the endpoints silently ignore an unrecognised field or
enum value, so a wrong spelling produces plausible-looking but wrong results
rather than an error.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from modelscope_hub._openapi import OpenAPIClient
from modelscope_hub.api import HubApi
from modelscope_hub.config import HubConfig
from modelscope_hub.constants import RepoType
from modelscope_hub.errors import InvalidParameter, NotSupportedError, PermissionDeniedError

_BASE = "https://modelscope.cn/openapi/v1"


@pytest.fixture
def client() -> OpenAPIClient:
    return OpenAPIClient(HubConfig(token="test-token", endpoint="https://modelscope.cn"))


def _response(payload=None, status_code=200):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.headers = {}
    resp.content = b"x"
    resp.json.return_value = payload if payload is not None else {"success": True, "data": {}}
    resp.request = MagicMock(method="GET", path_url="/x", url=_BASE)
    resp.url = _BASE
    return resp


# ---------------------------------------------------------------------------
# GET /studios
# ---------------------------------------------------------------------------
class TestListStudios:
    def test_url_and_defaults(self, client):
        resp = _response({"studios": [], "total_count": 0, "page_number": 1, "page_size": 10})
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.list_studios()
        kwargs = mock_req.call_args.kwargs
        assert kwargs["method"] == "GET"
        assert kwargs["url"] == f"{_BASE}/studios"
        assert dict(kwargs["params"]) == {"page_number": "1", "page_size": "10"}

    def test_all_filters_are_serialised(self, client):
        resp = _response({"studios": [], "total_count": 0})
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.list_studios(
                search="chat",
                owner="alice",
                sort="likes",
                page_number=2,
                page_size=20,
                status="all",
                mcp_support=True,
                hardware_type="xgpu",
            )
        params = dict(mock_req.call_args.kwargs["params"])
        assert params == {
            "search": "chat",
            "owner": "alice",
            "sort": "likes",
            "page_number": "2",
            "page_size": "20",
            "status": "all",
            "mcp_support": "true",
            "hardware_type": "xgpu",
        }

    def test_false_flag_is_sent_not_dropped(self, client):
        """``False`` is a real filter value; dropping it would silently widen the query."""
        resp = _response({"studios": []})
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.list_studios(mcp_support=False)
        assert dict(mock_req.call_args.kwargs["params"])["mcp_support"] == "false"

    @pytest.mark.parametrize(
        ("kwargs", "field"),
        [
            ({"sort": "downloads"}, "sort"),
            ({"status": "stopped"}, "status"),
            ({"hardware_type": "nvidia"}, "hardware_type"),
        ],
    )
    def test_unknown_enum_value_is_rejected(self, client, kwargs, field):
        with pytest.raises(InvalidParameter, match=field):
            client.list_studios(**kwargs)

    def test_offset_limit_is_enforced(self, client):
        with pytest.raises(InvalidParameter, match="<= 3000"):
            client.list_studios(page_number=61, page_size=50)

    def test_anonymous_when_no_token(self):
        config = HubConfig(token="placeholder", endpoint="https://modelscope.cn")
        config.token = None
        client = OpenAPIClient(config)
        with patch.object(HubConfig, "load_token", return_value=None):
            with patch.object(client._session, "request", return_value=_response({"studios": []})) as mock_req:
                client.list_studios()
        assert "Authorization" not in mock_req.call_args.kwargs["headers"]


# ---------------------------------------------------------------------------
# Resource discovery
# ---------------------------------------------------------------------------
class TestStudioResourceDiscovery:
    def test_hardware_url_and_params(self, client):
        resp = _response({"success": True, "data": {"hardware": []}})
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.list_studio_hardware(sdk_type="gradio", studio="alice/demo")
        kwargs = mock_req.call_args.kwargs
        assert kwargs["method"] == "GET"
        assert kwargs["url"] == f"{_BASE}/studios/hardware"
        assert dict(kwargs["params"]) == {"sdk_type": "gradio", "studio": "alice/demo"}

    def test_hardware_rejects_unknown_sdk_type(self, client):
        with pytest.raises(InvalidParameter, match="sdk_type"):
            client.list_studio_hardware(sdk_type="flask")

    def test_base_images_url(self, client):
        resp = _response({"success": True, "data": {"base_images": []}})
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.list_studio_base_images()
        kwargs = mock_req.call_args.kwargs
        assert kwargs["method"] == "GET"
        assert kwargs["url"] == f"{_BASE}/studios/base-images"

    def test_sdk_versions_url_and_params(self, client):
        resp = _response({"success": True, "data": {"sdk_versions": []}})
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.list_studio_sdk_versions(sdk_type="gradio")
        kwargs = mock_req.call_args.kwargs
        assert kwargs["url"] == f"{_BASE}/studios/sdk-versions"
        assert dict(kwargs["params"]) == {"sdk_type": "gradio"}

    @pytest.mark.parametrize(
        "method_name",
        ["list_studio_hardware", "list_studio_base_images", "list_studio_sdk_versions"],
    )
    def test_discovery_works_without_a_token(self, method_name):
        config = HubConfig(token="placeholder", endpoint="https://modelscope.cn")
        config.token = None
        client = OpenAPIClient(config)
        with patch.object(HubConfig, "load_token", return_value=None):
            with patch.object(client._session, "request", return_value=_response()) as mock_req:
                getattr(client, method_name)()
        assert "Authorization" not in mock_req.call_args.kwargs["headers"]


# ---------------------------------------------------------------------------
# Plaintext variables
# ---------------------------------------------------------------------------
class TestStudioVariables:
    def test_list_url(self, client):
        resp = _response({"success": True, "data": {"variables": []}})
        with patch.object(client._session, "request", return_value=resp) as mock_req:
            client.list_studio_variables("alice", "demo")
        kwargs = mock_req.call_args.kwargs
        assert kwargs["method"] == "GET"
        assert kwargs["url"] == f"{_BASE}/studios/alice/demo/variables"

    @pytest.mark.parametrize(
        ("method_name", "verb"),
        [("add_studio_variable", "POST"), ("update_studio_variable", "PUT")],
    )
    def test_write_body(self, client, method_name, verb):
        with patch.object(client._session, "request", return_value=_response()) as mock_req:
            getattr(client, method_name)("alice", "demo", "MODEL", "Qwen")
        kwargs = mock_req.call_args.kwargs
        assert kwargs["method"] == verb
        assert kwargs["url"] == f"{_BASE}/studios/alice/demo/variables"
        assert kwargs["json"] == {"key": "MODEL", "value": "Qwen"}

    def test_delete_body_carries_key_only(self, client):
        with patch.object(client._session, "request", return_value=_response()) as mock_req:
            client.delete_studio_variable("alice", "demo", "MODEL")
        kwargs = mock_req.call_args.kwargs
        assert kwargs["method"] == "DELETE"
        assert kwargs["json"] == {"key": "MODEL"}

    def test_variables_never_degrade_to_anonymous(self, client):
        """Account-private data: an anonymous retry would report a false 'empty'."""
        denied = _response({"message": "denied"}, status_code=403)
        with patch.object(client._session, "request", return_value=denied) as mock_req:
            with pytest.raises(PermissionDeniedError):
                client.list_studio_variables("alice", "demo")
        assert mock_req.call_count == 1

    def test_variables_mirror_the_secrets_routes(self, client):
        """The two blocks must stay symmetrical apart from the path segment."""
        captured = []
        with patch.object(client._session, "request", return_value=_response()) as mock_req:
            client.add_studio_variable("a", "d", "K", "V")
            captured.append(mock_req.call_args.kwargs)
            client.add_studio_secret("a", "d", "K", "V")
            captured.append(mock_req.call_args.kwargs)
        variable, secret = captured
        assert variable["json"] == secret["json"]
        assert variable["method"] == secret["method"]
        assert variable["url"].replace("/variables", "") == secret["url"].replace("/secrets", "")


# ---------------------------------------------------------------------------
# Studio logs
# ---------------------------------------------------------------------------
class TestStudioLogs:
    def test_page_size_cap_is_enforced(self, client):
        with pytest.raises(InvalidParameter, match="<= 500"):
            client.get_studio_logs("alice", "demo", "run", page_size=501)

    def test_page_size_at_boundary_is_accepted(self, client):
        with patch.object(client._session, "request", return_value=_response()) as mock_req:
            client.get_studio_logs("alice", "demo", "run", page_size=500)
        assert dict(mock_req.call_args.kwargs["params"])["page_size"] == "500"

    def test_unknown_log_type_is_rejected(self, client):
        with pytest.raises(InvalidParameter, match="log_type"):
            client.get_studio_logs("alice", "demo", "trace")


# ---------------------------------------------------------------------------
# HubApi facade
# ---------------------------------------------------------------------------
class TestFacade:
    @pytest.fixture
    def api(self):
        api = HubApi(token="test-token")
        api._openapi = MagicMock()
        return api

    def test_list_repos_studio_is_supported(self, api):
        api._openapi.list_studios.return_value = {
            "studios": [{"id": "alice/demo", "sdk_type": "gradio", "hardware": "cpu"}],
            "total_count": 1,
            "page_number": 1,
            "page_size": 10,
        }
        page = api.list_repos(RepoType.STUDIO)
        assert page.total_count == 1
        assert page.items[0].repo_id == "alice/demo"
        assert page.items[0].sdk_type == "gradio"
        assert page.collection_key == "studios"

    def test_list_repos_studio_parses_unenveloped_payload(self, api):
        """The endpoint omits the {success, data} envelope other lists use."""
        api._openapi.list_studios.return_value = {"studios": [{"id": "a/b"}], "total_count": 7}
        page = api.list_repos("studio")
        assert [r.repo_id for r in page.items] == ["a/b"]
        assert page.total_count == 7

    def test_list_repos_studio_forwards_filters(self, api):
        api._openapi.list_studios.return_value = {"studios": []}
        api.list_repos("studio", owner="alice", status="all", mcp_support=True)
        kwargs = api._openapi.list_studios.call_args.kwargs
        assert kwargs["owner"] == "alice"
        assert kwargs["status"] == "all"
        assert kwargs["mcp_support"] is True

    def test_variables_unwrap_the_payload(self, api):
        api._openapi.list_studio_variables.return_value = {"variables": [{"key": "K", "value": "V"}]}
        assert api.list_variables("alice/demo") == [{"key": "K", "value": "V"}]

    def test_variables_tolerate_a_bare_list(self, api):
        api._openapi.list_studio_variables.return_value = [{"key": "K", "value": "V"}]
        assert api.list_variables("alice/demo") == [{"key": "K", "value": "V"}]

    def test_variables_tolerate_an_empty_payload(self, api):
        api._openapi.list_studio_variables.return_value = None
        assert api.list_variables("alice/demo") == []

    @pytest.mark.parametrize(
        ("method_name", "args"),
        [
            ("list_variables", ()),
            ("add_variable", ("K", "V")),
            ("update_variable", ("K", "V")),
            ("delete_variable", ("K",)),
        ],
    )
    def test_variables_are_studio_only(self, api, method_name, args):
        with pytest.raises(NotSupportedError):
            getattr(api, method_name)("alice/demo", *args, repo_type=RepoType.MODEL)

    def test_hardware_unwraps_the_payload(self, api):
        api._openapi.list_studio_hardware.return_value = {"hardware": [{"name": "cpu"}]}
        assert api.list_studio_hardware() == [{"name": "cpu"}]

    def test_hardware_forwards_repo_id_as_studio(self, api):
        api._openapi.list_studio_hardware.return_value = {"hardware": []}
        api.list_studio_hardware(sdk_type="gradio", repo_id="alice/demo")
        api._openapi.list_studio_hardware.assert_called_once_with(sdk_type="gradio", studio="alice/demo")

    def test_base_images_unwraps_the_payload(self, api):
        api._openapi.list_studio_base_images.return_value = {"base_images": [{"name": "ubuntu"}]}
        assert api.list_studio_base_images() == [{"name": "ubuntu"}]

    def test_sdk_versions_unwraps_the_payload(self, api):
        api._openapi.list_studio_sdk_versions.return_value = {"sdk_versions": [{"version": "4.44.1"}]}
        assert api.list_studio_sdk_versions(sdk_type="gradio") == [{"version": "4.44.1"}]

    def test_operational_mcp_servers_matches_list_mcp_servers_shape(self, api):
        api._openapi.list_operational_mcp_servers.return_value = {
            "mcp_server_list": [{"id": "alice/weather", "operational_urls": [{"url": "https://x/sse"}]}],
            "total_count": 1,
        }
        page = api.list_operational_mcp_servers()
        assert page.total_count == 1
        assert page.items[0]["id"] == "alice/weather"


# ---------------------------------------------------------------------------
# Studio field normalisation must not leak onto other repo types
# ---------------------------------------------------------------------------
class TestStudioFieldNormalisation:
    @pytest.fixture
    def api(self):
        api = HubApi(token="test-token")
        api._openapi = MagicMock()
        return api

    def test_camelcase_cover_image_is_normalised_for_studios(self, api):
        api.create_repo("alice/demo", RepoType.STUDIO, coverImage="https://img")
        payload = api._openapi.create_studio.call_args[0][0]
        assert payload["cover_image"] == "https://img"
        assert "coverImage" not in payload

    def test_studio_visibility_expands_to_the_private_companion(self, api):
        api.create_repo("alice/demo", RepoType.STUDIO, visibility="protected")
        payload = api._openapi.create_studio.call_args[0][0]
        assert payload["visibility"] == "protected"
        assert payload["private"] is False

    def test_skill_payload_is_left_alone(self, api):
        """A Skill has no cover image and no visibility field, so the Studio
        normaliser must not invent them on a Skill payload."""
        api.create_repo("alice/my-skill", RepoType.SKILL, coverImage="https://img")
        payload = api._openapi.create_skill.call_args[0][0]
        assert payload["coverImage"] == "https://img"
        assert "cover_image" not in payload

    def test_skill_visibility_does_not_gain_a_studio_field(self, api):
        api.create_repo("alice/my-skill", RepoType.SKILL, visibility="private")
        payload = api._openapi.create_skill.call_args[0][0]
        assert "visibility" not in payload
        assert payload["private"] is True

    def test_settings_normalisation_applies_to_studios_only(self, api):
        api.update_repo_settings("alice/demo", RepoType.STUDIO, coverImage="https://img")
        assert api._openapi.update_studio_settings.call_args[0][2] == {"cover_image": "https://img"}
        api.update_repo_settings("alice/my-skill", RepoType.SKILL, logo_url="https://logo")
        assert api._openapi.update_skill_settings.call_args[0][2] == {"logo_url": "https://logo"}
