"""Regression tests for the Studio methods on the legacy compat surface.

The old ``modelscope.hub.api.HubApi`` signature let callers append ``token=`` /
``endpoint=`` to any method, and the umbrella SDK's Studio CLI does exactly
that. Forwarding those kwargs wholesale caused two concrete defects that these
tests pin down:

* ``update_studio_settings`` forwards its kwargs as the settings payload, so the
  caller's API token was serialised into the ``PATCH`` request body.
* ``get_studio_logs`` forwards into a keyword-only signature, so passing
  ``token=`` raised ``TypeError`` and the call could never succeed.

Network-free: the OpenAPI client is mocked, so assertions run against the exact
arguments that would have gone on the wire.
"""

from __future__ import annotations

from unittest import mock

import pytest

from modelscope_hub.api import HubApi
from modelscope_hub.compat import LegacyHubApi
from modelscope_hub.compat.hub_api import _split_control_kwargs


def _compat(token: str = "ms-ambient-token") -> tuple[LegacyHubApi, mock.MagicMock]:
    """Build a compat wrapper whose OpenAPI transport is mocked out."""
    api = HubApi(token=token)
    openapi = mock.MagicMock()
    api._openapi = openapi
    legacy = LegacyHubApi.__new__(LegacyHubApi)
    legacy._api = api
    legacy._endpoint = None
    return legacy, openapi


# ---------------------------------------------------------------------------
# The helper itself
# ---------------------------------------------------------------------------
class TestSplitControlKwargs:
    def test_control_args_are_separated(self):
        control, passthrough = _split_control_kwargs({"token": "ms-x", "endpoint": "https://e", "display_name": "X"})
        assert control == {"token": "ms-x", "endpoint": "https://e"}
        assert passthrough == {"display_name": "X"}

    def test_all_control_keys_recognised(self):
        control, passthrough = _split_control_kwargs(
            {
                "token": "t",
                "endpoint": "e",
                "cookies": "c",
                "headers": {},
                "timeout": 1,
                "max_retries": 2,
            }
        )
        assert passthrough == {}
        assert set(control) == {"token", "endpoint", "cookies", "headers", "timeout", "max_retries"}

    def test_business_only_kwargs_pass_through_untouched(self):
        control, passthrough = _split_control_kwargs({"key": "K", "value": "V"})
        assert control == {}
        assert passthrough == {"key": "K", "value": "V"}


# ---------------------------------------------------------------------------
# Defect 1: the token must never reach a request body
# ---------------------------------------------------------------------------
class TestSettingsDoesNotLeakCredentials:
    def test_token_and_endpoint_absent_from_patch_body(self):
        legacy, openapi = _compat()
        legacy.update_studio_settings(
            "owner/demo",
            token="ms-ambient-token",
            endpoint="https://modelscope.cn",
            display_name="Demo",
        )
        settings = openapi.update_studio_settings.call_args[0][2]
        assert "token" not in settings
        assert "endpoint" not in settings
        assert settings == {"display_name": "Demo"}

    def test_every_control_key_is_stripped(self):
        legacy, openapi = _compat()
        legacy.update_studio_settings(
            "owner/demo",
            token="ms-ambient-token",
            endpoint="https://modelscope.cn",
            cookies="session=1",
            headers={"X": "1"},
            timeout=5,
            max_retries=1,
            description="d",
        )
        assert openapi.update_studio_settings.call_args[0][2] == {"description": "d"}

    def test_business_fields_still_reach_the_wire(self):
        legacy, openapi = _compat()
        legacy.update_studio_settings(
            "owner/demo",
            token="ms-ambient-token",
            display_name="Demo",
            description="d",
            license="apache-2.0",
            private=True,
        )
        assert openapi.update_studio_settings.call_args[0][2] == {
            "display_name": "Demo",
            "description": "d",
            "license": "apache-2.0",
            "private": True,
        }


# ---------------------------------------------------------------------------
# Defect 2: ``studio logs`` used to raise TypeError outright
# ---------------------------------------------------------------------------
class TestLogsAcceptsControlKwargs:
    def test_token_kwarg_does_not_raise_type_error(self):
        legacy, openapi = _compat()
        legacy.get_studio_logs(
            "owner/demo",
            token="ms-ambient-token",
            endpoint="https://modelscope.cn",
            log_type="run",
            page_size=5,
        )
        args, kwargs = openapi.get_studio_logs.call_args
        assert args == ("owner", "demo", "run")
        assert kwargs["page_size"] == 5

    def test_log_options_are_forwarded(self):
        legacy, openapi = _compat()
        legacy.get_studio_logs(
            "owner/demo",
            token="ms-ambient-token",
            log_type="build",
            page_num=3,
            page_size=50,
            keyword="ERROR",
            start_timestamp=1,
            end_timestamp=2,
        )
        args, kwargs = openapi.get_studio_logs.call_args
        assert args == ("owner", "demo", "build")
        assert kwargs == {
            "page_num": 3,
            "page_size": 50,
            "keyword": "ERROR",
            "start_timestamp": 1,
            "end_timestamp": 2,
        }


# ---------------------------------------------------------------------------
# Defect 3: a per-call token used to be silently ignored
# ---------------------------------------------------------------------------
class TestPerCallTokenOverride:
    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("deploy_studio", ("owner/demo",)),
            ("stop_studio", ("owner/demo",)),
            ("list_studio_secrets", ("owner/demo",)),
            ("add_studio_secret", ("owner/demo", "K", "V")),
            ("update_studio_secret", ("owner/demo", "K", "V")),
            ("delete_studio_secret", ("owner/demo", "K")),
            ("update_studio_settings", ("owner/demo",)),
        ],
    )
    def test_distinct_token_builds_a_dedicated_client(self, method, args):
        legacy, _ = _compat(token="ms-ambient-token")
        with mock.patch("modelscope_hub.compat.hub_api.HubApi") as hub_cls:
            getattr(legacy, method)(*args, token="ms-per-call")
        hub_cls.assert_called_once()
        assert hub_cls.call_args.kwargs["token"] == "ms-per-call"

    def test_matching_token_reuses_the_ambient_client(self):
        legacy, openapi = _compat(token="ms-ambient-token")
        with mock.patch("modelscope_hub.compat.hub_api.HubApi") as hub_cls:
            legacy.stop_studio("owner/demo", token="ms-ambient-token")
        hub_cls.assert_not_called()
        openapi.stop_studio.assert_called_once_with("owner", "demo")

    def test_no_token_reuses_the_ambient_client(self):
        legacy, openapi = _compat()
        with mock.patch("modelscope_hub.compat.hub_api.HubApi") as hub_cls:
            legacy.deploy_studio("owner/demo")
        hub_cls.assert_not_called()
        openapi.deploy_studio.assert_called_once()

    def test_distinct_endpoint_builds_a_dedicated_client(self):
        legacy, _ = _compat()
        with mock.patch("modelscope_hub.compat.hub_api.HubApi") as hub_cls:
            legacy.stop_studio("owner/demo", endpoint="https://modelscope.ai")
        hub_cls.assert_called_once()
        assert hub_cls.call_args.kwargs["endpoint"] == "https://modelscope.ai"


# ---------------------------------------------------------------------------
# Untouched delegation behaviour
# ---------------------------------------------------------------------------
class TestStudioDelegation:
    def test_deploy_forwards_payload(self):
        legacy, openapi = _compat()
        legacy.deploy_studio("owner/demo", payload={"a": 1}, token="ms-ambient-token")
        assert openapi.deploy_studio.call_args[0] == ("owner", "demo", {"a": 1})

    def test_secret_writes_return_none(self):
        legacy, _ = _compat()
        assert legacy.add_studio_secret("owner/demo", "K", "V") is None
        assert legacy.update_studio_secret("owner/demo", "K", "V") is None
        assert legacy.delete_studio_secret("owner/demo", "K") is None


# ---------------------------------------------------------------------------
# Newly delegated Studio operations
# ---------------------------------------------------------------------------
class TestNewStudioDelegation:
    def test_list_studios_returns_a_paginated_dict(self):
        legacy, openapi = _compat()
        openapi.list_studios.return_value = {
            "studios": [{"id": "owner/demo"}],
            "total_count": 1,
            "page_number": 1,
            "page_size": 10,
        }
        result = legacy.list_studios(token="ms-ambient-token")
        assert result["total_count"] == 1
        assert result["studios"][0]["id"] == "owner/demo"

    def test_list_studios_forwards_filters(self):
        legacy, openapi = _compat()
        openapi.list_studios.return_value = {"studios": []}
        legacy.list_studios(owner="alice", page_size=20, token="ms-ambient-token")
        kwargs = openapi.list_studios.call_args.kwargs
        assert kwargs["owner"] == "alice"
        assert kwargs["page_size"] == 20

    def test_list_variables_unwraps_the_payload(self):
        legacy, openapi = _compat()
        openapi.list_studio_variables.return_value = {"variables": [{"key": "K", "value": "V"}]}
        assert legacy.list_studio_variables("owner/demo") == [{"key": "K", "value": "V"}]
        assert openapi.list_studio_variables.call_args[0] == ("owner", "demo")

    @pytest.mark.parametrize(
        ("method", "args", "target"),
        [
            ("add_studio_variable", ("owner/demo", "K", "V"), "add_studio_variable"),
            ("update_studio_variable", ("owner/demo", "K", "V"), "update_studio_variable"),
            ("delete_studio_variable", ("owner/demo", "K"), "delete_studio_variable"),
        ],
    )
    def test_variable_writes_return_none_and_reach_the_endpoint(self, method, args, target):
        legacy, openapi = _compat()
        assert getattr(legacy, method)(*args) is None
        assert getattr(openapi, target).call_args[0][:2] == ("owner", "demo")

    def test_hardware_unwraps_and_forwards_options(self):
        legacy, openapi = _compat()
        openapi.list_studio_hardware.return_value = {"hardware": [{"name": "cpu"}]}
        assert legacy.list_studio_hardware(sdk_type="gradio") == [{"name": "cpu"}]
        assert openapi.list_studio_hardware.call_args.kwargs["sdk_type"] == "gradio"

    def test_base_images_unwraps_the_payload(self):
        legacy, openapi = _compat()
        openapi.list_studio_base_images.return_value = {"base_images": [{"name": "ubuntu"}]}
        assert legacy.list_studio_base_images() == [{"name": "ubuntu"}]

    def test_sdk_versions_unwraps_and_forwards_options(self):
        legacy, openapi = _compat()
        openapi.list_studio_sdk_versions.return_value = {"sdk_versions": [{"version": "4.44.1"}]}
        assert legacy.list_studio_sdk_versions(sdk_type="gradio") == [{"version": "4.44.1"}]
        assert openapi.list_studio_sdk_versions.call_args.kwargs["sdk_type"] == "gradio"

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("list_studios", ()),
            ("list_studio_variables", ("owner/demo",)),
            ("add_studio_variable", ("owner/demo", "K", "V")),
            ("update_studio_variable", ("owner/demo", "K", "V")),
            ("delete_studio_variable", ("owner/demo", "K")),
            ("list_studio_hardware", ()),
            ("list_studio_base_images", ()),
            ("list_studio_sdk_versions", ()),
        ],
    )
    def test_control_kwargs_never_become_business_arguments(self, method, args):
        """Every new shim must strip token/endpoint like the existing ones do."""
        legacy, openapi = _compat()
        openapi.list_studios.return_value = {"studios": []}
        with mock.patch("modelscope_hub.compat.hub_api.HubApi") as hub_cls:
            getattr(legacy, method)(*args, token="ms-per-call", endpoint="https://modelscope.ai")
        hub_cls.assert_called_once()
        for call in hub_cls.return_value.mock_calls:
            assert "token" not in call.kwargs
            assert "endpoint" not in call.kwargs
