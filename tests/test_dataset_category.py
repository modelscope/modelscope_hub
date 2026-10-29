"""The dataset zone endpoints, and the settings fields that are not writable.

Mirrors the model zone rules with two differences worth pinning: the general
dataset category is ``base`` (not ``basic``, as earlier drafts had it), and
``login_required`` appears in a dataset's detail payload but no request schema
accepts it, so it cannot be set through the SDK at all.
"""

from __future__ import annotations

from unittest import mock

import pytest

from modelscope_hub import HubApi
from modelscope_hub import api as api_module
from modelscope_hub.errors import InvalidParameter, NotSupportedError


@pytest.fixture
def api() -> HubApi:
    return HubApi(endpoint="https://modelscope.cn", token="ms-test")


@pytest.fixture
def request_mock(api: HubApi):
    with mock.patch.object(api.openapi, "_request", return_value={"id": "alice/d"}) as m:
        yield m


# ---------------------------------------------------------------------------
# PUT /category
# ---------------------------------------------------------------------------
def test_setting_a_zone_uses_put(api, request_mock):
    api.set_dataset_category(
        "alice/d",
        dataset_category="science",
        primary_subject_area="life-sciences",
        secondary_specialization="dna",
    )

    assert request_mock.call_args.args == ("PUT", "/datasets/alice/d/category")
    assert request_mock.call_args.kwargs["json_body"] == {
        "dataset_category": "science",
        "primary_subject_area": "life-sciences",
        "secondary_specialization": "dna",
    }


def test_the_general_category_is_base_not_basic(api, request_mock):
    api.set_dataset_category("alice/d", dataset_category="base")

    assert request_mock.call_args.kwargs["json_body"] == {"dataset_category": "base"}


def test_basic_is_rejected_and_the_valid_values_are_named(api, request_mock):
    with pytest.raises(InvalidParameter, match="'base', 'science'"):
        api.set_dataset_category("alice/d", dataset_category="basic")
    request_mock.assert_not_called()


# ---------------------------------------------------------------------------
# PATCH /category
# ---------------------------------------------------------------------------
def test_updating_a_zone_uses_patch(api, request_mock):
    api.update_dataset_category("alice/d", secondary_specialization="medicine")

    assert request_mock.call_args.args == ("PATCH", "/datasets/alice/d/category")
    assert request_mock.call_args.kwargs["json_body"] == {"secondary_specialization": "medicine"}


def test_the_category_discriminator_is_dropped_with_a_warning(api, request_mock, caplog, monkeypatch):
    monkeypatch.setattr(api_module.logger, "handlers", [*api_module.logger.handlers, caplog.handler])
    with caplog.at_level("WARNING", logger="modelscope_hub.api"):
        api.update_dataset_category("alice/d", dataset_category="science", cover_image="https://x/y.png")

    assert "dataset_category" not in request_mock.call_args.kwargs["json_body"]
    assert "set_dataset_category" in caplog.text


def test_an_explicit_none_is_forwarded_to_clear_a_field(api, request_mock):
    """Omitting a field leaves it unchanged; ``None`` clears it."""
    api.update_dataset_category("alice/d", cover_image=None)

    assert request_mock.call_args.kwargs["json_body"] == {"cover_image": None}


# ---------------------------------------------------------------------------
# PATCH /settings
# ---------------------------------------------------------------------------
def test_dataset_settings_reach_the_settings_endpoint(api, request_mock):
    api.update_repo_settings("alice/d", "dataset", description="new", display_name="D")

    assert request_mock.call_args.args == ("PATCH", "/datasets/alice/d/settings")
    assert request_mock.call_args.kwargs["json_body"] == {"description": "new", "display_name": "D"}


def test_model_settings_reach_the_settings_endpoint(api, request_mock):
    api.update_repo_settings("alice/m", "model", task="text-generation")

    assert request_mock.call_args.args == ("PATCH", "/models/alice/m/settings")
    assert request_mock.call_args.kwargs["json_body"] == {"task": "text-generation"}


@pytest.mark.parametrize("repo_type", ["model", "dataset"])
@pytest.mark.parametrize("visibility", [None, "private", 1, "1"])
@pytest.mark.parametrize("switch", [True, False])
def test_settings_use_historical_switch_and_visibility(api, request_mock, repo_type, visibility, switch):
    api.update_repo_settings("alice/x", repo_type, visibility=visibility, gated_mode=switch)
    expected = {"visibility": "gated", "gated_mode": "auto"} if switch else {"visibility": "private"}
    assert request_mock.call_args.kwargs["json_body"] == expected


@pytest.mark.parametrize("mode", ["auto", "manual", ""])
def test_native_settings_payload_still_belongs_to_openapi(api, request_mock, mode):
    payload = {"visibility": "gated", "gated_mode": mode}
    api.openapi.update_dataset_settings("alice", "d", payload)
    assert request_mock.call_args.kwargs["json_body"] == payload


@pytest.mark.parametrize("bad", ["auto", "manual", "", 0, []])
def test_settings_reject_non_boolean_switches(api, request_mock, bad):
    with pytest.raises(InvalidParameter, match="gated_mode must be"):
        api.update_repo_settings("alice/d", "dataset", gated_mode=bad)
    request_mock.assert_not_called()


@pytest.mark.parametrize("name", ["approval_mode", "gated"])
@pytest.mark.parametrize("value", [None, False, "manual"])
@pytest.mark.parametrize("repo_type", ["model", "dataset", "studio", "skill"])
def test_removed_settings_parameters_are_rejected_before_sending(api, request_mock, repo_type, name, value):
    with pytest.raises(TypeError, match=name):
        api.update_repo_settings("alice/d", repo_type, **{name: value})
    request_mock.assert_not_called()


def test_settings_none_is_not_a_new_gating_policy(api, request_mock):
    api.update_repo_settings("alice/d", "dataset", gated_mode=None, visibility=None, description="same")
    assert request_mock.call_args.kwargs["json_body"] == {"description": "same"}


def test_settings_do_not_accept_native_gated_visibility(api, request_mock):
    with pytest.raises(ValueError, match="Unknown visibility"):
        api.update_repo_settings("alice/d", "dataset", visibility="gated")
    request_mock.assert_not_called()


def test_integer_visibility_is_translated_for_settings(api, request_mock):
    api.update_repo_settings("alice/d", "dataset", visibility=1)

    assert request_mock.call_args.kwargs["json_body"] == {"visibility": "private"}


def test_internal_visibility_is_refused_for_settings(api, request_mock):
    """Unlike create, there is no legacy settings endpoint to fall back to."""
    with pytest.raises(InvalidParameter, match="not supported"):
        api.update_repo_settings("alice/d", "dataset", visibility="internal")
    request_mock.assert_not_called()


def test_login_required_is_reported_as_unsupported(api, request_mock):
    """No request schema accepts it, so silently dropping it would mislead."""
    with pytest.raises(NotSupportedError, match="login_required"):
        api.update_repo_settings("alice/d", "dataset", login_required=True)
    request_mock.assert_not_called()


def test_unsupported_repo_types_still_raise(api, request_mock):
    with pytest.raises(NotSupportedError):
        api.update_repo_settings("alice/x", "mcp", description="x")


# ---------------------------------------------------------------------------
# High-level historical switch, not an approval-policy selector
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("repo_type", ["model", "dataset"])
def test_false_switch_changes_to_private_without_native_boolean(api, request_mock, repo_type):
    api.update_repo_settings("alice/x", repo_type, gated_mode=False)
    assert request_mock.call_args.kwargs["json_body"] == {"visibility": "private"}


def test_public_visibility_wins_over_only_boolean_switch(api, request_mock, caplog, monkeypatch):
    monkeypatch.setattr(api_module.logger, "handlers", [*api_module.logger.handlers, caplog.handler])
    api.update_repo_settings("alice/x", "model", visibility="public", gated_mode=True)
    assert request_mock.call_args.kwargs["json_body"] == {"visibility": "public"}
    assert "ignored" in caplog.text


# ---------------------------------------------------------------------------
# Repo type guards
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("repo_type", ["model", "studio", "skill"])
def test_the_dataset_methods_reject_other_repo_types(api, request_mock, repo_type):
    with pytest.raises(NotSupportedError):
        api.set_dataset_category("alice/d", repo_type, dataset_category="base")
    with pytest.raises(NotSupportedError):
        api.update_dataset_category("alice/d", repo_type, cover_image=None)
    request_mock.assert_not_called()
