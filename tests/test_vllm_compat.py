"""Test argument forwarding, cache lookup, and opt-in vLLM adapter behavior."""

import functools
import importlib.machinery
import pickle
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from modelscope_hub import _vllm_compat as compat
from modelscope_hub import config
from modelscope_hub.errors import CacheNotFound


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "true")
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path))
    monkeypatch.setattr(config, "_default_config", None)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", SimpleNamespace(HF_HUB_OFFLINE=True))
    path = tmp_path / "models/org--model/snapshots/master"
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    return path


def original_convert(model_repo):
    return "legacy:" + model_repo


def test_converter_is_local_and_no_download(cache):
    convert = compat._wrap_convert(original_convert)
    with patch.object(compat, "snapshot_download", side_effect=AssertionError("纯转换不得下载")):
        assert convert("org/model") == str(cache)
        assert convert(str(cache)) == str(cache)
        assert convert("org/missing") == "org/missing"
        assert convert("org/model", revision="main") == "org/model"


@pytest.mark.parametrize("flag", ["", "false", "0"])
def test_disabled_converter_is_unchanged(cache, monkeypatch, flag):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", flag)
    assert compat._wrap_convert(original_convert)("org/model") == "legacy:org/model"


@pytest.mark.parametrize("revision", [None, "master"])
def test_offline_loader_preserves_revision_and_kwargs(cache, revision):
    def original(processor_name, *args, revision=None, **kwargs):
        return processor_name, args, revision, kwargs

    load = compat._wrap_loader(original, download=True)
    with patch.object(compat, "snapshot_download", side_effect=AssertionError("离线不得下载")):
        assert load("org/model", 7, revision=revision, token="", local_files_only=True) == (
            str(cache),
            (7,),
            revision,
            {"token": "", "local_files_only": True},
        )


@pytest.mark.parametrize("revision", ["main", "unknown"])
def test_offline_loader_rejects_missing_revision(cache, revision):
    def original(processor_name, *, revision=None, **kwargs):
        raise AssertionError("缺失快照不得交给 HF")

    with pytest.raises(CacheNotFound):
        compat._wrap_loader(original, download=True)("org/model", revision=revision)


def test_loader_preserves_custom_cache(cache, tmp_path):
    other = tmp_path / "custom"
    snapshot = other / "models/org--model/snapshots/master"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")

    def original(processor_name, **kwargs):
        return processor_name, kwargs

    assert compat._wrap_loader(original, download=True)("org/model", cache_dir=other)[0] == str(snapshot)


def test_online_loader_downloads_only_non_weights(cache, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", SimpleNamespace(HF_HUB_OFFLINE=False))

    def original(processor_name, *, revision=None, **kwargs):
        return processor_name, revision, kwargs

    with patch.object(compat, "snapshot_download", return_value=str(cache)) as download:
        value = compat._wrap_loader(original, download=True)(
            "org/model",
            revision="v1",
            token=False,
            user_agent="custom/1",
            trust_remote_code=True,
        )
    assert value[0] == str(cache) and value[1] == "v1"
    assert value[2]["trust_remote_code"] is True
    assert download.call_args.kwargs["token"] == ""
    assert "*.safetensors" in download.call_args.kwargs["ignore_patterns"]
    assert "invoked_by/vllm" in download.call_args.kwargs["user_agent"]
    assert "custom/1" in download.call_args.kwargs["user_agent"]


def test_offline_config_is_resolved_before_hf_lookup(cache):
    def original(model, trust_remote_code, revision=None, **kwargs):
        return model, trust_remote_code, revision

    assert compat._wrap_config(original)("org/model", False, "master") == (str(cache), False, "master")
    with pytest.raises(CacheNotFound):
        compat._wrap_config(original)("org/model", False, "main")


def test_adapt_rebuilds_lru_aliases_once(cache):
    module = ModuleType("vllm.transformers_utils.processor")

    def get_processor(processor_name, *, revision=None, **kwargs):
        return processor_name

    module.get_processor = get_processor
    module.cached_get_processor = functools.lru_cache(maxsize=3)(get_processor)
    compat._adapt(module)
    wrapped = module.get_processor
    assert module.cached_get_processor("org/model") == str(cache)
    assert module.cached_get_processor.cache_parameters()["maxsize"] == 3
    compat._adapt(module)
    assert module.get_processor is wrapped
    assert module.cached_get_processor.cache_info().hits == 0


def test_converted_path_resolves_later_explicit_revision(cache):
    path = compat._wrap_convert(original_convert)("org/model")
    alternate = cache.with_name("v1")
    alternate.mkdir()
    (alternate / "config.json").write_text("{}")

    def original(processor_name, *, revision=None, **kwargs):
        return processor_name

    load = compat._wrap_loader(original, download=False)
    assert isinstance(path, str)
    restored = pickle.loads(pickle.dumps(path))
    assert restored.repo_id == path.repo_id
    assert load(restored, revision="v1") == str(alternate)
    assert load(path, revision="v1") == str(alternate)
    with pytest.raises(CacheNotFound):
        load(path, revision="missing")
    assert load(str(path), revision="missing") == str(cache)


def test_converter_retains_explicit_revision_for_later_lookup(cache):
    alternate = cache.with_name("v1")
    alternate.mkdir()
    (alternate / "config.json").write_text("{}")
    path = compat._wrap_convert(original_convert)("org/model", revision="v1")

    def original(processor_name, *, revision=None):
        return processor_name

    assert compat._wrap_loader(original, download=False)(path) == str(alternate)


@pytest.mark.parametrize("filename", ["flax_model.msgpack", "model.onnx_data", "model.safetensors"])
def test_all_weight_formats_excluded(cache, monkeypatch, filename):
    from modelscope_hub._download import _matches_patterns

    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", SimpleNamespace(HF_HUB_OFFLINE=False))

    def original(processor_name, **kwargs):
        return processor_name

    with patch.object(compat, "snapshot_download", return_value=str(cache)) as download:
        compat._wrap_loader(original, download=True)("org/model")
    assert _matches_patterns(filename, download.call_args.kwargs["ignore_patterns"])


def test_finder_ignores_unrelated_and_disabled_imports(cache, monkeypatch):
    finder = compat._Finder()
    assert finder.find_spec("transformers") is None
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "false")
    assert finder.find_spec("vllm.transformers_utils.processor") is None


def test_finder_delegates_to_custom_importer(cache, monkeypatch):
    finder = compat._Finder()
    original = SimpleNamespace(create_module=lambda spec: None, exec_module=lambda module: None)
    spec = importlib.machinery.ModuleSpec("vllm.transformers_utils.utils", original)
    custom = SimpleNamespace(find_spec=lambda *args: spec)
    monkeypatch.setattr(sys, "meta_path", [finder, custom])
    result = finder.find_spec(spec.name)
    assert isinstance(result.loader, compat._Loader)
    assert result.loader.original is original
