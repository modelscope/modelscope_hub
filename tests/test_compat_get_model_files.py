"""Unit tests for selected legacy-compatible ``LegacyHubApi`` methods.

Network-free: the bottom-level HTTP transport is mocked where full compat
call-chain coverage matters.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from modelscope_hub.compat import LegacyHubApi


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

    def test_plain_model_keeps_generic_create_repo_path(self):
        api = LegacyHubApi(endpoint="https://modelscope.cn", token="ms-test")

        with mock.patch.object(api._api.legacy, "_request", return_value=_response()) as request:
            url = api.create_model(
                "owner/plain-model",
                visibility=1,
                license="Apache License 2.0",
                chinese_name="普通模型",
                aigc_model=None,
            )

        request.assert_called_once()
        method, path = request.call_args.args
        body = request.call_args.kwargs["json_body"]
        assert (method, path) == ("POST", "models")
        assert body["Path"] == "owner"
        assert body["Name"] == "plain-model"
        assert "TagShowName" not in body
        assert "aigc_model" not in body
        assert url == "https://modelscope.cn/models/owner/plain-model"

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
