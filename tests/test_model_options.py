"""The enum option endpoints: anonymous, cached, and ``{type, id, label}``.

These four endpoints publish product configuration rather than account data, so
they take no credentials and must stay reachable without a token. Their values
change on a release cadence, so repeated reads are served from a short-lived
cache -- otherwise setting a category would re-fetch the whole enum tree on
every call.
"""

from __future__ import annotations

import importlib
from unittest import mock

import pytest

from modelscope_hub import HubApi

_AIGC_OPTIONS = {
    "aigc_types": [{"type": "aigc_type", "id": "LoRA", "label": "LoRA"}],
    "base_architectures": [
        {
            "type": "base_architecture",
            "id": "Z_IMAGE_TURBO",
            "label": "Z-Image Turbo",
            "subtypes": [{"type": "base_architecture_subtype", "id": "Z_IMAGE_TURBO", "label": "Z-Image Turbo"}],
        }
    ],
    "content_styles": [{"type": "content_style", "id": "ILLUSTRATION", "label": "插画"}],
    "content_functions": [{"type": "content_function", "id": "TEXT_TO_IMAGE", "label": "文生图"}],
    "content_elements": [{"type": "content_element", "id": "PERSON", "label": "人物"}],
}

_SCIENCE_OPTIONS = {
    "subject_areas": [
        {
            "type": "subject_area",
            "id": "life-sciences",
            "label": "生命科学",
            "specializations": [{"type": "specialization", "id": "dna", "label": "基因组学"}],
        }
    ]
}

_MODEL_OPTIONS = {
    "tasks": [{"type": "task", "id": "text-generation", "label": "Text Generation"}],
    "license": [{"type": "license", "id": "apache-2.0", "label": "Apache License 2.0"}],
}

# (HubApi method, request path, payload)
_ENDPOINTS = [
    ("list_aigc_model_options", "/model/aigc/options", _AIGC_OPTIONS),
    ("list_science_model_options", "/model/science/options", _SCIENCE_OPTIONS),
    ("list_science_dataset_options", "/dataset/science/options", _SCIENCE_OPTIONS),
    ("list_model_options", "/models/options", _MODEL_OPTIONS),
]


@pytest.fixture
def api() -> HubApi:
    return HubApi(endpoint="https://modelscope.cn", token="ms-test")


# ---------------------------------------------------------------------------
# Routing and anonymity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(("method", "path", "payload"), _ENDPOINTS)
def test_each_endpoint_is_reached_without_requiring_a_token(api, method, path, payload):
    with mock.patch.object(api.openapi, "_request", return_value=payload) as request_mock:
        result = getattr(api, method)()

    assert request_mock.call_args.args == ("GET", path)
    assert request_mock.call_args.kwargs["require_token"] is False
    assert result == payload


@pytest.mark.parametrize(("method", "path", "payload"), _ENDPOINTS)
def test_the_endpoints_work_with_no_credentials_at_all(method, path, payload):
    anonymous = HubApi(endpoint="https://modelscope.cn", token=None)

    with mock.patch.object(anonymous.openapi, "_request", return_value=payload) as request_mock:
        getattr(anonymous, method)()

    assert request_mock.call_args.kwargs["require_token"] is False


@pytest.mark.parametrize(("method", "_path", "_payload"), _ENDPOINTS)
def test_a_non_mapping_payload_degrades_to_an_empty_dict(api, method, _path, _payload):
    with mock.patch.object(api.openapi, "_request", return_value=None):
        assert getattr(api, method)() == {}


# ---------------------------------------------------------------------------
# Option item shape
# ---------------------------------------------------------------------------
def test_options_are_submitted_by_id_and_displayed_by_label(api):
    with mock.patch.object(api.openapi, "_request", return_value=_AIGC_OPTIONS):
        options = api.list_aigc_model_options()

    architecture = options["base_architectures"][0]
    for key in ("type", "id", "label"):
        assert key in architecture
    assert "value" not in architecture  # an earlier draft's spelling
    assert architecture["subtypes"][0]["id"] == "Z_IMAGE_TURBO"


def test_the_science_tree_cascades_from_subject_area_to_specialization(api):
    with mock.patch.object(api.openapi, "_request", return_value=_SCIENCE_OPTIONS):
        tree = api.list_science_model_options()

    area = tree["subject_areas"][0]
    assert area["id"] == "life-sciences"
    assert [s["id"] for s in area["specializations"]] == ["dna"]


def test_model_options_publish_tasks_and_licenses(api):
    with mock.patch.object(api.openapi, "_request", return_value=_MODEL_OPTIONS):
        options = api.list_model_options()

    assert [t["id"] for t in options["tasks"]] == ["text-generation"]
    assert [lic["id"] for lic in options["license"]] == ["apache-2.0"]


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def test_a_repeated_read_is_served_from_the_cache(api):
    with mock.patch.object(api.openapi, "_request", return_value=_AIGC_OPTIONS) as request_mock:
        first = api.list_aigc_model_options()
        second = api.list_aigc_model_options()

    request_mock.assert_called_once()
    assert first == second


def test_each_endpoint_is_cached_separately(api):
    with mock.patch.object(api.openapi, "_request", side_effect=[_AIGC_OPTIONS, _SCIENCE_OPTIONS]) as request_mock:
        api.list_aigc_model_options()
        api.list_science_model_options()

    assert [call.args[1] for call in request_mock.call_args_list] == [
        "/model/aigc/options",
        "/model/science/options",
    ]


def test_clearing_the_cache_forces_a_refetch(api):
    with mock.patch.object(api.openapi, "_request", return_value=_AIGC_OPTIONS) as request_mock:
        api.list_aigc_model_options()
        api.openapi.clear_options_cache()
        api.list_aigc_model_options()

    assert request_mock.call_count == 2


def test_two_clients_do_not_share_a_cache(api):
    other = HubApi(endpoint="https://modelscope.cn", token="ms-test")

    with (
        mock.patch.object(api.openapi, "_request", return_value=_AIGC_OPTIONS) as first_mock,
        mock.patch.object(other.openapi, "_request", return_value=_AIGC_OPTIONS) as second_mock,
    ):
        api.list_aigc_model_options()
        other.list_aigc_model_options()

    first_mock.assert_called_once()
    second_mock.assert_called_once()


def test_a_zero_ttl_disables_caching(monkeypatch):
    """``MODELSCOPE_OPTIONS_CACHE_TTL=0`` is the documented opt-out."""
    monkeypatch.setenv("MODELSCOPE_OPTIONS_CACHE_TTL", "0")
    import modelscope_hub.constants as constants_module

    importlib.reload(constants_module)
    assert constants_module.OPTIONS_CACHE_TTL_SECONDS == 0

    import modelscope_hub._openapi as openapi_module

    importlib.reload(openapi_module)
    try:
        client = openapi_module.OpenAPIClient()
        with mock.patch.object(client, "_request", return_value=_AIGC_OPTIONS) as request_mock:
            client.list_aigc_model_options()
            client.list_aigc_model_options()
        assert request_mock.call_count == 2
    finally:
        # Restore the real modules so later tests see the default TTL.
        monkeypatch.delenv("MODELSCOPE_OPTIONS_CACHE_TTL", raising=False)
        importlib.reload(constants_module)
        importlib.reload(openapi_module)
        importlib.reload(importlib.import_module("modelscope_hub.api"))
        importlib.reload(importlib.import_module("modelscope_hub"))


def test_an_expired_entry_is_refetched(api, monkeypatch):
    import modelscope_hub._openapi as openapi_module

    monkeypatch.setattr(openapi_module, "OPTIONS_CACHE_TTL_SECONDS", 300)
    clock = iter([0.0, 0.0, 1000.0, 1000.0])
    monkeypatch.setattr(openapi_module.time, "monotonic", lambda: next(clock))

    with mock.patch.object(api.openapi, "_request", return_value=_AIGC_OPTIONS) as request_mock:
        api.list_aigc_model_options()
        api.list_aigc_model_options()

    assert request_mock.call_count == 2
