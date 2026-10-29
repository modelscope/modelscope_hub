"""Unit tests for selected legacy-compatible ``LegacyHubApi`` methods.

Network-free: the bottom-level HTTP transport is mocked where full compat
call-chain coverage matters.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from modelscope_hub.compat import LegacyHubApi
from modelscope_hub.errors import InvalidParameter, RequestTimeoutError


class _FakeAigcModel:
    tag = "v1.0"
    cover_images = ["data:image/png;base64,AAAA"]
    aigc_type = "LoRA"
    description = "AIGC compatibility test"
    base_model_type = "SD_XL"
    base_model_id = "owner/base-model"
    weight_filename = "model.safetensors"
    weight_sha256 = "abc123"
    weight_size = 42
    model_path = "/tmp/model.safetensors"
    trigger_words = ["trigger"]
    model_source = "USER_UPLOAD"
    base_model_sub_type = "SD_XL"
    official_tags = ["photography"]
    readme_content = None

    def __init__(self):
        self.preupload_weights = mock.MagicMock()
        self.upload_to_repo = mock.MagicMock(return_value=True)


def _response(data=None):
    response = mock.MagicMock()
    response.json.return_value = {"Data": data or {}}
    return response


def _fake_files():
    return [
        SimpleNamespace(path="config.json", size=10),
        SimpleNamespace(path="model.safetensors", size=100),
        SimpleNamespace(path="subdir/extra.bin", size=5),
    ]


@pytest.mark.parametrize("name", ["approval_mode", "gated"])
@pytest.mark.parametrize("value", [None, "manual"])
def test_legacy_aigc_rejects_removed_parameters_before_upload(name, value):
    lha = LegacyHubApi(token="ms-test")
    aigc = _FakeAigcModel()
    with pytest.raises(TypeError, match=name):
        lha.create_model("owner/repo", aigc_model=aigc, visibility=1, gated_mode=True, **{name: value})
    aigc.preupload_weights.assert_not_called()
    aigc.upload_to_repo.assert_not_called()


@pytest.mark.parametrize("method", ["create_repo", "create_model", "update_model", "update_dataset"])
@pytest.mark.parametrize("name", ["approval_mode", "gated"])
def test_plain_compat_wrappers_reject_removed_request_parameters(method, name):
    lha = LegacyHubApi(token="ms-test")
    with (
        mock.patch.object(lha._api.legacy, "create_repo") as legacy,
        mock.patch.object(lha._api.openapi, "_request") as native,
    ):
        with pytest.raises(TypeError, match=name):
            getattr(lha, method)("owner/repo", **{name: None})
    legacy.assert_not_called()
    native.assert_not_called()


class TestGetModelFilesLegacyCompat:
    def test_revision_is_accepted_and_forwarded(self):
        lha = LegacyHubApi()
        with mock.patch.object(lha._api, "list_repo_files", return_value=_fake_files()) as m:
            out = lha.get_model_files("Qwen/Qwen2.5-1.5B-Instruct", revision="v2")

        assert [f["Path"] for f in out] == [
            "config.json",
            "model.safetensors",
            "subdir/extra.bin",
        ]
        _, kwargs = m.call_args
        assert kwargs["revision"] == "v2"

    def test_root_restricts_to_subpath(self):
        lha = LegacyHubApi()
        with mock.patch.object(lha._api, "list_repo_files", return_value=_fake_files()):
            out = lha.get_model_files("owner/name", root="subdir")

        assert [f["Path"] for f in out] == ["subdir/extra.bin"]

    def test_tolerates_legacy_transport_kwargs(self):
        lha = LegacyHubApi()
        with mock.patch.object(lha._api, "list_repo_files", return_value=_fake_files()):
            # Historical kwargs must not raise "unexpected keyword argument".
            out = lha.get_model_files("owner/name", revision="master", use_cookies=True, headers={})

        assert len(out) == 3

    def test_default_revision_none_forwarded(self):
        lha = LegacyHubApi()
        with mock.patch.object(lha._api, "list_repo_files", return_value=_fake_files()) as m:
            lha.get_model_files("owner/name")

        _, kwargs = m.call_args
        assert kwargs["revision"] is None
        assert kwargs["recursive"] is True


class TestCreateModelLegacyCompat:
    def test_aigc_model_uses_dedicated_endpoint_and_payload(self):
        api = LegacyHubApi(endpoint="https://modelscope.cn", token="ms-test")
        aigc_model = _FakeAigcModel()

        with mock.patch.object(api._api.legacy, "_request", return_value=_response()) as request:
            url = api.create_model(
                "owner/aigc-model",
                visibility=1,
                license="Apache License 2.0",
                chinese_name="AIGC 模型",
                original_model_id="owner/original-model",
                aigc_model=aigc_model,
                gated_mode=False,
            )

        request.assert_called_once()
        method, path = request.call_args.args
        body = request.call_args.kwargs["json_body"]
        assert (method, path) == ("POST", "models/aigc")
        assert body == {
            "Path": "owner",
            "Name": "aigc-model",
            "ChineseName": "AIGC 模型",
            "Visibility": 1,
            "License": "Apache License 2.0",
            "OriginalModelId": "owner/original-model",
            "TrainId": "",
            "TagShowName": "v1.0",
            "CoverImages": ["data:image/png;base64,AAAA"],
            "AigcType": "LoRA",
            "TagDescription": "AIGC compatibility test",
            "VisionFoundation": "SD_XL",
            "BaseModel": "owner/base-model",
            "WeightsName": "model.safetensors",
            "WeightsSha256": "abc123",
            "WeightsSize": 42,
            "ModelPath": "/tmp/model.safetensors",
            "TriggerWords": ["trigger"],
            "ModelSource": "USER_UPLOAD",
            "SubVisionFoundation": "SD_XL",
            "OfficialTags": ["photography"],
            "ProtectedMode": 2,
        }
        assert url == "https://modelscope.cn/models/owner/aigc-model"
        aigc_model.preupload_weights.assert_called_once()
        preupload_kwargs = aigc_model.preupload_weights.call_args.kwargs
        assert preupload_kwargs["cookies"]["m_session_id"] == "ms-test"
        assert preupload_kwargs["endpoint"] == "https://modelscope.cn"
        aigc_model.upload_to_repo.assert_called_once()
        upload_api, model_id, token = aigc_model.upload_to_repo.call_args.args
        assert upload_api._api is api._api
        assert model_id == "owner/aigc-model"
        assert token is None

    def test_aigc_model_uploads_optional_readme_after_creation(self):
        api = LegacyHubApi(endpoint="https://modelscope.cn", token="ms-test")
        aigc_model = _FakeAigcModel()

        with (
            mock.patch.object(api._api.legacy, "_request", return_value=_response()) as legacy_request,
            mock.patch.object(api._api.openapi, "_request", return_value={}) as openapi_request,
        ):
            api.create_model(
                "owner/aigc-model",
                aigc_model=aigc_model,
                readme_content="# Initial AIGC README\n",
            )

        # The AIGC repository itself still goes through the dedicated legacy
        # endpoint: no OpenAPI route accepts its payload.
        assert legacy_request.call_count == 2
        create_call, readme_commit_call = legacy_request.call_args_list
        assert create_call.args == ("POST", "models/aigc")
        assert readme_commit_call.args == (
            "POST",
            "repos/models/owner/aigc-model/commit/master",
        )
        # Only the uploader's "ensure the repo exists" step moved to OpenAPI.
        assert [call.args for call in openapi_request.call_args_list] == [("POST", "/models")]
        commit_body = readme_commit_call.kwargs["json_body"]
        assert commit_body["commit_message"] == "Update README.md for AIGC version v1.0"
        operation = commit_body["actions"][0]
        assert operation["path"] == "README.md"
        assert operation["content"] == "IyBJbml0aWFsIEFJR0MgUkVBRE1FCg=="

    def test_plain_model_uses_the_openapi_create_endpoint(self):
        api = LegacyHubApi(endpoint="https://modelscope.cn", token="ms-test")

        with (
            mock.patch.object(
                api._api.openapi,
                "_request",
                return_value={"id": "owner/plain-model", "owner": "owner", "repo_name": "plain-model"},
            ) as openapi_request,
            mock.patch.object(api._api.legacy, "_request", return_value=_response()) as legacy_request,
        ):
            url = api.create_model(
                "owner/plain-model",
                visibility=1,
                license="Apache License 2.0",
                chinese_name="普通模型",
                aigc_model=None,
            )

        legacy_request.assert_not_called()
        openapi_request.assert_called_once()
        assert openapi_request.call_args.args == ("POST", "/models")
        body = openapi_request.call_args.kwargs["json_body"]
        assert body == {
            "owner": "owner",
            "repo_name": "plain-model",
            "visibility": "private",
            "display_name": "普通模型",
            "license": "apache-2.0",
        }
        assert "TagShowName" not in body
        assert "aigc_model" not in body
        assert url == "https://modelscope.cn/models/owner/plain-model"

    def test_plain_model_falls_back_to_legacy_for_an_unlistable_license(self):
        """``POST /models`` rejects licenses outside its enum; legacy accepts them."""
        api = LegacyHubApi(endpoint="https://modelscope.cn", token="ms-test")

        with (
            mock.patch.object(api._api.openapi, "_request") as openapi_request,
            mock.patch.object(api._api.legacy, "_request", return_value=_response()) as legacy_request,
        ):
            api.create_model("owner/plain-model", license="Unlicense", aigc_model=None)

        openapi_request.assert_not_called()
        legacy_request.assert_called_once()
        assert legacy_request.call_args.args == ("POST", "models")
        assert legacy_request.call_args.kwargs["json_body"]["License"] == "unlicense"

    def test_aigc_model_tag_uses_dedicated_endpoint(self):
        api = LegacyHubApi(endpoint="https://modelscope.cn", token="ms-test")
        aigc_model = _FakeAigcModel()

        with mock.patch.object(api._api.legacy, "_request", return_value=_response()) as request:
            url = api.create_model_tag(
                "owner/aigc-model",
                "v1.1",
                aigc_model=aigc_model,
            )

        request.assert_called_once()
        method, path = request.call_args.args
        assert (method, path) == ("POST", "models/aigc/repo/tag")
        assert request.call_args.kwargs["json_body"] == {
            "CoverImages": ["data:image/png;base64,AAAA"],
            "Name": "aigc-model",
            "Path": "owner",
            "TagShowName": "v1.1",
            "WeightsName": "model.safetensors",
            "WeightsSha256": "abc123",
            "WeightsSize": 42,
            "TriggerWords": ["trigger"],
            "AigcType": "LoRA",
            "VisionFoundation": "SD_XL",
        }
        aigc_model.preupload_weights.assert_called_once()
        assert url == "https://modelscope.cn/models/owner/aigc-model/tags/v1.1"

    def test_aigc_model_tag_reconciles_timeout_with_remote_state(self):
        api = LegacyHubApi(endpoint="https://modelscope.cn", token="ms-test")
        aigc_model = _FakeAigcModel()
        revisions = _response(
            {
                "RevisionMap": {
                    "Branches": [{"Revision": "master", "ShowName": ""}],
                    "Tags": [{"Revision": "20260829162320", "ShowName": "v1.3"}],
                }
            }
        )

        with mock.patch.object(
            api._api.legacy,
            "_request",
            side_effect=[RequestTimeoutError("timed out"), revisions],
        ) as request:
            url = api.create_model_tag(
                "owner/aigc-model",
                "v1.3",
                aigc_model=aigc_model,
            )

        assert request.call_count == 2
        assert request.call_args_list[0].args == ("POST", "models/aigc/repo/tag")
        assert request.call_args_list[1].args == (
            "GET",
            "models/owner/aigc-model/revisions",
        )
        assert url == "https://modelscope.cn/models/owner/aigc-model/tags/v1.3"

    def test_aigc_model_tag_uploads_readme_before_snapshot(self):
        api = LegacyHubApi(endpoint="https://modelscope.cn", token="ms-test")
        aigc_model = _FakeAigcModel()
        aigc_model.readme_content = "# Version v1.3\n\nCustom README content.\n"

        with (
            mock.patch.object(api._api.legacy, "_request", return_value=_response()) as legacy_request,
            mock.patch.object(api._api.openapi, "_request", return_value={}),
        ):
            api.create_model_tag(
                "owner/aigc-model",
                "v1.3",
                aigc_model=aigc_model,
            )

        # The README must be committed before the immutable tag snapshot.
        assert legacy_request.call_count == 2
        readme_commit_call, tag_call = legacy_request.call_args_list
        assert readme_commit_call.args == (
            "POST",
            "repos/models/owner/aigc-model/commit/master",
        )
        commit_body = readme_commit_call.kwargs["json_body"]
        assert commit_body["commit_message"] == "Update README.md for AIGC version v1.3"
        operation = commit_body["actions"][0]
        assert operation["path"] == "README.md"
        assert operation["type"] == "normal"
        assert operation["encoding"] == "base64"
        assert operation["content"] == "IyBWZXJzaW9uIHYxLjMKCkN1c3RvbSBSRUFETUUgY29udGVudC4K"
        assert tag_call.args == ("POST", "models/aigc/repo/tag")
        assert tag_call.kwargs["json_body"]["TagShowName"] == "v1.3"

    def test_explicit_readme_content_overrides_aigc_model_value(self):
        api = LegacyHubApi(endpoint="https://modelscope.cn", token="ms-test")
        aigc_model = _FakeAigcModel()
        aigc_model.readme_content = "model value"

        with (
            mock.patch.object(api._api.legacy, "_request", return_value=_response()) as legacy_request,
            mock.patch.object(api._api.openapi, "_request", return_value={}),
        ):
            api.create_model_tag(
                "owner/aigc-model",
                "v1.3",
                aigc_model=aigc_model,
                readme_content="explicit value",
            )

        commit_body = legacy_request.call_args_list[0].kwargs["json_body"]
        operation = commit_body["actions"][0]
        assert operation["content"] == "ZXhwbGljaXQgdmFsdWU="

    def test_plain_model_tag_keeps_generic_endpoint(self):
        api = LegacyHubApi(endpoint="https://modelscope.cn", token="ms-test")

        with mock.patch.object(api._api.legacy, "_request", return_value=_response()) as request:
            url = api.create_model_tag("owner/plain-model", "v1.1")

        request.assert_called_once()
        method, path = request.call_args.args
        assert (method, path) == ("POST", "models/owner/plain-model/repo/tag")
        assert request.call_args.kwargs["json_body"] == {
            "TagName": "v1.1",
            "Ref": "master",
        }
        assert url == "https://modelscope.cn/models/owner/plain-model/tags/v1.1"

    def test_plain_model_tag_rejects_aigc_readme_content(self):
        api = LegacyHubApi(endpoint="https://modelscope.cn", token="ms-test")

        with mock.patch.object(api._api.legacy, "_request") as request:
            with pytest.raises(
                InvalidParameter,
                match="readme_content is only supported for AIGC model tags",
            ):
                api.create_model_tag(
                    "owner/plain-model",
                    "v1.1",
                    readme_content="# AIGC only",
                )
        request.assert_not_called()
