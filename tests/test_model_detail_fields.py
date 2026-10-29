"""``get_model`` / ``get_dataset`` return the detail payload, not a ``RepoInfo``.

The zone fields -- AIGC architecture and ``versions``, or the science subject
areas -- have no place on the ``RepoInfo`` dataclass, so ``get_repo`` drops
them. These methods hand back the server's payload as-is instead.

Which keys are present depends on the category: the fields of the zones a
repository is *not* in are absent rather than null, so callers must use
``.get()`` and must not assume a fixed shape.
"""

from __future__ import annotations

from unittest import mock

import pytest

from modelscope_hub import HubApi
from modelscope_hub.compat import LegacyHubApi
from modelscope_hub.constants import RepoType

_BASE_MODEL = {
    "id": "alice/plain",
    "owner": "alice",
    "repo_name": "plain",
    "model_category": "base",
    "visibility": "public",
    "downloads": 10,
    "likes": 2,
}

_AIGC_MODEL = {
    "id": "alice/lora",
    "owner": "alice",
    "repo_name": "lora",
    "model_category": "aigc",
    "visibility": "public",
    "aigc_type": "LoRA",
    "base_model_architecture_type": "Z_IMAGE_TURBO",
    "base_model_architecture_subtype": "Z_IMAGE_TURBO",
    "content_categories": {"style": "ILLUSTRATION"},
    "versions": [
        {
            "version_id": 1070760,
            "version_name": "v1.0",
            "weight_files": [{"path_in_repo": "model.safetensors", "size": 42, "sha": "abc"}],
            "sample_images": [{"url": "https://example.com/a.png", "prompt": "a cat"}],
            "trigger_words": ["anime"],
            "created_at": "2026-06-22T10:00:00Z",
        }
    ],
}

_SCIENCE_MODEL = {
    "id": "alice/rna",
    "owner": "alice",
    "repo_name": "rna",
    "model_category": "science",
    "primary_subject_area": "life-sciences",
    "secondary_specialization": "dna",
    "cover_image": "https://example.com/cover.png",
}

_SCIENCE_DATASET = {
    "id": "alice/rna-seq",
    "owner": "alice",
    "repo_name": "rna-seq",
    "dataset_category": "science",
    "login_required": True,
    "primary_subject_area": "life-sciences",
    "secondary_specialization": "dna",
}


@pytest.fixture
def api() -> HubApi:
    return HubApi(endpoint="https://modelscope.cn", token="ms-test")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def test_get_model_reads_the_model_detail_endpoint(api):
    with mock.patch.object(api.openapi, "_request", return_value=_AIGC_MODEL) as request_mock:
        detail = api.get_model("alice/lora")

    assert request_mock.call_args.args == ("GET", "/models/alice/lora")
    assert detail == _AIGC_MODEL


def test_get_dataset_reads_the_dataset_detail_endpoint(api):
    with mock.patch.object(api.openapi, "_request", return_value=_SCIENCE_DATASET) as request_mock:
        detail = api.get_dataset("alice/rna-seq")

    assert request_mock.call_args.args == ("GET", "/datasets/alice/rna-seq")
    assert detail == _SCIENCE_DATASET


@pytest.mark.parametrize(("alias", "canonical"), [("model_info", "get_model"), ("dataset_info", "get_dataset")])
def test_the_info_aliases_match_their_canonical_methods(api, alias, canonical):
    with mock.patch.object(api.openapi, "_request", return_value=_BASE_MODEL) as request_mock:
        from_alias = getattr(api, alias)("alice/plain")
        from_canonical = getattr(api, canonical)("alice/plain")

    assert from_alias == from_canonical
    assert request_mock.call_args_list[0].args == request_mock.call_args_list[1].args


# ---------------------------------------------------------------------------
# Zone fields survive, and are category-dependent
# ---------------------------------------------------------------------------
def test_aigc_versions_survive_with_their_numeric_ids(api):
    """``update_model_category`` needs ``version_id``, so it must not be lost."""
    with mock.patch.object(api.openapi, "_request", return_value=_AIGC_MODEL):
        detail = api.get_model("alice/lora")

    version = detail["versions"][0]
    assert version["version_id"] == 1070760
    assert version["weight_files"][0]["sha"] == "abc"
    assert version["sample_images"][0]["prompt"] == "a cat"


def test_a_base_model_carries_no_zone_fields(api):
    with mock.patch.object(api.openapi, "_request", return_value=_BASE_MODEL):
        detail = api.get_model("alice/plain")

    assert detail["model_category"] == "base"
    for absent in ("aigc_type", "versions", "primary_subject_area", "cover_image"):
        assert absent not in detail


def test_a_science_model_carries_subject_areas_and_no_versions(api):
    with mock.patch.object(api.openapi, "_request", return_value=_SCIENCE_MODEL):
        detail = api.get_model("alice/rna")

    assert detail["primary_subject_area"] == "life-sciences"
    assert detail["cover_image"] == "https://example.com/cover.png"
    assert "versions" not in detail
    assert "aigc_type" not in detail


def test_an_aigc_model_carries_no_cover_image(api):
    """Cover images belong to the science zone only."""
    with mock.patch.object(api.openapi, "_request", return_value=_AIGC_MODEL):
        detail = api.get_model("alice/lora")

    assert "cover_image" not in detail


def test_login_required_is_readable_even_though_it_is_not_writable(api):
    with mock.patch.object(api.openapi, "_request", return_value=_SCIENCE_DATASET):
        detail = api.get_dataset("alice/rna-seq")

    assert detail["login_required"] is True


def test_a_non_mapping_payload_degrades_to_an_empty_dict(api):
    with mock.patch.object(api.openapi, "_request", return_value=None):
        assert api.get_model("alice/plain") == {}
        assert api.get_dataset("alice/plain") == {}


# ---------------------------------------------------------------------------
# RepoInfo keeps the category, without gaining the zone payload
# ---------------------------------------------------------------------------
def test_get_repo_reports_the_category_but_not_the_zone_payload(api):
    with mock.patch.object(api.openapi, "_request", return_value=_AIGC_MODEL):
        info = api.get_repo("alice/lora", "model")

    assert info.model_category == "aigc"
    assert info.repo_id == "alice/lora"
    # ``versions`` has no dataclass field: use get_model() for it.
    assert not hasattr(info, "versions")


def test_a_dataset_repo_info_reports_its_category(api):
    with mock.patch.object(api.openapi, "_request", return_value=_SCIENCE_DATASET):
        info = api.get_repo("alice/rna-seq", "dataset")

    assert info.dataset_category == "science"
    assert info.login_required is True


def test_the_new_fields_are_omitted_from_to_dict_when_absent(api):
    with mock.patch.object(api.openapi, "_request", return_value={"id": "alice/x", "private": False}):
        info = api.get_repo("alice/x", "studio")

    as_dict = info.to_dict()
    for optional in ("model_category", "dataset_category", "gated_mode"):
        assert optional not in as_dict


# ---------------------------------------------------------------------------
# Compat surface keeps both shapes available
# ---------------------------------------------------------------------------
def test_compat_get_model_keeps_its_legacy_pascalcase_shape(api):
    compat = LegacyHubApi(endpoint="https://modelscope.cn", token="ms-test")

    with mock.patch.object(compat._api.openapi, "_request", return_value=_AIGC_MODEL):
        out = compat.get_model("alice/lora")

    # Historical contract: a flattened RepoInfo with PascalCase keys.
    assert out["Id"] == "alice/lora"
    assert "model_category" in out  # carried through as a RepoInfo field


def test_compat_get_model_detail_returns_the_raw_payload(api):
    compat = LegacyHubApi(endpoint="https://modelscope.cn", token="ms-test")

    with mock.patch.object(compat._api.openapi, "_request", return_value=_AIGC_MODEL):
        out = compat.get_model_detail("alice/lora")

    assert out == _AIGC_MODEL
    assert out["versions"][0]["version_id"] == 1070760


def test_compat_get_dataset_detail_returns_the_raw_payload(api):
    compat = LegacyHubApi(endpoint="https://modelscope.cn", token="ms-test")

    with mock.patch.object(compat._api.openapi, "_request", return_value=_SCIENCE_DATASET):
        out = compat.get_dataset_detail("alice/rna-seq")

    assert out == _SCIENCE_DATASET


def test_compat_get_dataset_still_returns_a_repo_info():
    """Deprecated but unchanged: it hands back the dataclass, not a dict."""
    compat = LegacyHubApi(endpoint="https://modelscope.cn", token="ms-test")

    with mock.patch.object(compat._api.openapi, "_request", return_value=_SCIENCE_DATASET):
        with pytest.deprecated_call():
            out = compat.get_dataset("alice/rna-seq")

    assert out.repo_id == "alice/rna-seq"
    assert out.repo_type is RepoType.DATASET
