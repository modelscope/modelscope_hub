"""The model zone endpoints: ``PUT`` to change zone, ``PATCH`` to edit in place.

The two verbs are not interchangeable, and their payload rules are close to
inverses of each other:

* ``PUT`` sets the zone. Required fields follow from the *target* zone, and
  fields belonging to another zone are rejected.
* ``PATCH`` edits the current zone. The server reads the zone itself and
  rejects the discriminator, addresses AIGC versions by numeric id, and refuses
  to touch an existing version's weights.

Checks the SDK can make locally are made locally, so a request certain to fail
names the fix instead of costing a round trip for a generic 400.
"""

from __future__ import annotations

from unittest import mock

import pytest

from modelscope_hub import HubApi
from modelscope_hub import api as api_module
from modelscope_hub.constants import WEIGHT_FILE_SUFFIXES
from modelscope_hub.errors import InvalidParameter, NotSupportedError

_AIGC = {
    "aigc_type": "LoRA",
    "base_model_architecture_type": "Z_IMAGE_TURBO",
    "base_model_architecture_subtype": "Z_IMAGE_TURBO",
    "version_name": "v1.0",
    "weight_files": [{"path_in_repo": "model.safetensors"}],
    "sample_images": [{"url": "https://example.com/a.png", "prompt": "a cat"}],
}


@pytest.fixture
def api() -> HubApi:
    return HubApi(endpoint="https://modelscope.cn", token="ms-test")


@pytest.fixture
def request_mock(api: HubApi):
    with mock.patch.object(api.openapi, "_request", return_value={"id": "alice/m"}) as m:
        yield m


# ---------------------------------------------------------------------------
# PUT: verb, path, and the per-zone required fields
# ---------------------------------------------------------------------------
def test_setting_a_zone_uses_put(api, request_mock):
    api.set_model_category("alice/m", model_category="base")

    assert request_mock.call_args.args == ("PUT", "/models/alice/m/category")
    assert request_mock.call_args.kwargs["json_body"] == {"model_category": "base"}


def test_moving_to_aigc_sends_every_required_field(api, request_mock):
    api.set_model_category("alice/m", model_category="aigc", **_AIGC)

    body = request_mock.call_args.kwargs["json_body"]
    assert body["model_category"] == "aigc"
    assert body["version_name"] == "v1.0"
    assert body["weight_files"] == [{"path_in_repo": "model.safetensors"}]
    assert body["sample_images"] == [{"url": "https://example.com/a.png", "prompt": "a cat"}]


def test_moving_to_science_needs_only_the_subject_areas(api, request_mock):
    api.set_model_category(
        "alice/m",
        model_category="science",
        primary_subject_area="life-sciences",
        secondary_specialization="dna",
    )

    assert request_mock.call_args.kwargs["json_body"] == {
        "model_category": "science",
        "primary_subject_area": "life-sciences",
        "secondary_specialization": "dna",
    }


@pytest.mark.parametrize("missing", sorted(_AIGC))
def test_a_missing_aigc_field_is_reported_before_the_request(api, request_mock, missing):
    payload = {k: v for k, v in _AIGC.items() if k != missing}

    with pytest.raises(InvalidParameter, match=missing):
        api.set_model_category("alice/m", model_category="aigc", **payload)
    request_mock.assert_not_called()


@pytest.mark.parametrize("empty", [[], None])
def test_empty_weight_files_and_sample_images_are_rejected(api, request_mock, empty):
    for field in ("weight_files", "sample_images"):
        with pytest.raises(InvalidParameter, match=field):
            api.set_model_category("alice/m", model_category="aigc", **{**_AIGC, field: empty})
    request_mock.assert_not_called()


def test_an_unknown_category_names_the_valid_values(api, request_mock):
    with pytest.raises(InvalidParameter, match="'base', 'aigc', 'science'"):
        api.set_model_category("alice/m", model_category="basic")
    request_mock.assert_not_called()


def test_an_unknown_aigc_type_names_the_valid_values(api, request_mock):
    with pytest.raises(InvalidParameter, match="Checkpoint"):
        api.set_model_category("alice/m", model_category="aigc", **{**_AIGC, "aigc_type": "lora"})
    request_mock.assert_not_called()


@pytest.mark.parametrize("suffix", [".safetensors", ".CKPT", ".Bin", ".onnx"])
def test_recognised_weight_suffixes_are_accepted_case_insensitively(api, request_mock, suffix):
    api.set_model_category(
        "alice/m",
        model_category="aigc",
        **{**_AIGC, "weight_files": [{"path_in_repo": f"sub/dir/model{suffix}"}]},
    )

    assert request_mock.called


def test_an_unrecognised_weight_suffix_lists_the_supported_ones(api, request_mock):
    with pytest.raises(InvalidParameter) as excinfo:
        api.set_model_category(
            "alice/m",
            model_category="aigc",
            **{**_AIGC, "weight_files": [{"path_in_repo": "notes.txt"}]},
        )

    message = str(excinfo.value)
    assert "notes.txt" in message
    for suffix in WEIGHT_FILE_SUFFIXES:
        assert suffix in message
    request_mock.assert_not_called()


def test_a_weight_file_without_a_path_is_rejected(api, request_mock):
    with pytest.raises(InvalidParameter, match="path_in_repo"):
        api.set_model_category("alice/m", model_category="aigc", **{**_AIGC, "weight_files": [{}]})


def test_a_sample_image_without_a_url_is_rejected(api, request_mock):
    with pytest.raises(InvalidParameter, match="url"):
        api.set_model_category(
            "alice/m",
            model_category="aigc",
            **{**_AIGC, "sample_images": [{"prompt": "no url"}]},
        )


def test_the_retired_sample_image_text_key_is_rejected(api, request_mock):
    """The schema takes ``prompt``/``negative_prompt``; ``text`` never existed."""
    with pytest.raises(InvalidParameter, match="prompt"):
        api.set_model_category(
            "alice/m",
            model_category="aigc",
            **{**_AIGC, "sample_images": [{"url": "https://example.com/a.png", "text": "cat"}]},
        )


def test_an_overlong_version_name_is_rejected(api, request_mock):
    with pytest.raises(InvalidParameter, match="version_name"):
        api.set_model_category("alice/m", model_category="aigc", **{**_AIGC, "version_name": "v" * 21})


# ---------------------------------------------------------------------------
# PATCH: verb, path, and the inverted payload rules
# ---------------------------------------------------------------------------
def test_updating_a_zone_uses_patch(api, request_mock):
    api.update_model_category("alice/m", primary_subject_area="life-sciences")

    assert request_mock.call_args.args == ("PATCH", "/models/alice/m/category")
    assert request_mock.call_args.kwargs["json_body"] == {"primary_subject_area": "life-sciences"}


def test_the_category_discriminator_is_dropped_with_a_warning(api, request_mock, caplog, monkeypatch):
    """Earlier drafts required the field; the endpoint now 400s if it is present."""
    monkeypatch.setattr(api_module.logger, "handlers", [*api_module.logger.handlers, caplog.handler])
    with caplog.at_level("WARNING", logger="modelscope_hub.api"):
        api.update_model_category("alice/m", model_category="aigc", trigger_words=["x"])

    assert "model_category" not in request_mock.call_args.kwargs["json_body"]
    assert "set_model_category" in caplog.text


@pytest.mark.parametrize("field", ["weight_files", "base_model_id"])
def test_fields_only_put_can_change_are_refused(api, request_mock, field):
    with pytest.raises(InvalidParameter, match="set_model_category"):
        api.update_model_category("alice/m", **{field: "whatever"})
    request_mock.assert_not_called()


def test_version_name_is_a_resolution_hint_not_a_wire_field(api):
    """The endpoint rejects ``version_name``; the SDK turns it into a lookup."""
    detail = {"versions": [{"version_name": "v1.0", "version_id": 42}]}

    with mock.patch.object(api.openapi, "_request", side_effect=[detail, {"id": "alice/m"}]) as request_mock:
        api.update_model_category("alice/m", version_name="v1.0", trigger_words=["x"])

    body = request_mock.call_args_list[1].kwargs["json_body"]
    assert "version_name" not in body
    assert body["version_id"] == 42


def test_an_explicit_version_id_is_forwarded(api, request_mock):
    api.update_model_category("alice/m", version_id=1070760, trigger_words=["anime"])

    assert request_mock.call_args.kwargs["json_body"]["version_id"] == 1070760


def test_a_science_update_sends_no_version_id(api, request_mock):
    """The endpoint rejects ``version_id`` outside the AIGC zone."""
    api.update_model_category("alice/m", cover_image=None)

    assert "version_id" not in request_mock.call_args.kwargs["json_body"]


# ---------------------------------------------------------------------------
# version_name -> version_id resolution
# ---------------------------------------------------------------------------
def test_a_version_name_is_resolved_to_its_numeric_id(api):
    detail = {"model_category": "aigc", "versions": [{"version_name": "v1.0", "version_id": 42}]}

    with mock.patch.object(api.openapi, "_request", side_effect=[detail, {"id": "alice/m"}]) as request_mock:
        api.update_model_category("alice/m", version_name="v1.0", trigger_words=["anime"])

    detail_call, update_call = request_mock.call_args_list
    assert detail_call.args == ("GET", "/models/alice/m")
    body = update_call.kwargs["json_body"]
    assert body["version_id"] == 42
    assert "version_name" not in body


def test_an_unknown_version_name_lists_what_is_available(api):
    detail = {"versions": [{"version_name": "v1.0", "version_id": 42}]}

    with mock.patch.object(api.openapi, "_request", return_value=detail):
        with pytest.raises(InvalidParameter) as excinfo:
            api.update_model_category("alice/m", version_name="v9.9", trigger_words=["x"])

    assert "'v1.0'" in str(excinfo.value)


def test_an_explicit_version_id_skips_the_lookup(api, request_mock):
    api.update_model_category("alice/m", version_id=7, version_name="v1.0", trigger_words=["x"])

    request_mock.assert_called_once()
    assert request_mock.call_args.kwargs["json_body"]["version_id"] == 7


# ---------------------------------------------------------------------------
# Repo type guards
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("repo_type", ["dataset", "studio", "skill"])
def test_the_model_methods_reject_other_repo_types(api, request_mock, repo_type):
    with pytest.raises(NotSupportedError):
        api.set_model_category("alice/m", repo_type, model_category="base")
    with pytest.raises(NotSupportedError):
        api.update_model_category("alice/m", repo_type, trigger_words=["x"])
    request_mock.assert_not_called()
