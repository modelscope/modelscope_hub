"""Test vLLM adapter boundaries; regression scripts exercise real upstream loaders."""

from __future__ import annotations

import functools
import inspect
import sys
from types import ModuleType

import pytest
import requests

from modelscope_hub import _vllm_compat as adapter
from modelscope_hub import config
from modelscope_hub.errors import CacheNotFound, InvalidParameter


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    adapter.uninstall()
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "true")
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MODELSCOPE_HOME", str(tmp_path / "auth"))
    monkeypatch.setattr(config, "_default_config", None)
    monkeypatch.setattr(requests.Session, "send", lambda *a, **k: pytest.fail("单元测试不能联网"))
    yield
    adapter.uninstall()


def cached(tmp_path, revision="master"):
    path = tmp_path / "cache/models/org--model/snapshots" / revision
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    return path


@pytest.mark.parametrize("flag", ["true", "1", "TRUE", " true "])
def test_conversion_uses_actual_snapshot(tmp_path, monkeypatch, flag):
    path = cached(tmp_path)
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", flag)
    wrapped = adapter._wrap_convert(lambda name: "legacy/" + name)
    before = sorted(tmp_path.rglob("*"))
    assert wrapped("org/model") == str(path)
    assert sorted(tmp_path.rglob("*")) == before
    assert not (tmp_path / "auth").exists()


@pytest.mark.parametrize("flag", ["", "0", "false"])
def test_disabled_conversion_preserves_original(flag, monkeypatch):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", flag)

    def original(name):
        return "original/" + name

    assert adapter._wrap_convert(original)("org/model") == "original/org/model"
    adapter.install()
    finder = next(finder for finder in sys.meta_path if isinstance(finder, adapter._Finder))
    assert finder.find_spec("vllm.transformers_utils.utils") is None


def test_conversion_preserves_local_paths(tmp_path):
    wrapped = adapter._wrap_convert(lambda name: "wrong")
    assert wrapped(str(tmp_path)) == str(tmp_path)


def test_cache_miss_keeps_repo_id_without_creating_directories(tmp_path):
    assert adapter._resolve("org/missing") == "org/missing"
    assert not (tmp_path / "cache").exists()


def test_missing_revision_cannot_reuse_master(tmp_path):
    cached(tmp_path)
    assert adapter._resolve("org/model", "absent") == "org/model"


def test_default_and_explicit_revisions_are_distinct(tmp_path):
    master = cached(tmp_path)
    main = cached(tmp_path, "main")
    release = cached(tmp_path, "release/v1")
    assert adapter._resolve("org/model") == str(master)
    assert adapter._resolve("org/model", "main") == str(main)
    assert adapter._resolve("org/model", "release/v1") == str(release)


def test_invalid_repo_id_is_not_masked():
    with pytest.raises(InvalidParameter):
        adapter._resolve("../outside")


@pytest.mark.parametrize("token,expected", [(False, ""), (True, None), ("test-token", "test-token")])
@pytest.mark.parametrize("offline", [False, True])
def test_loader_forwards_revision_cache_auth_and_offline(tmp_path, monkeypatch, token, expected, offline):
    hf_constants = pytest.importorskip("huggingface_hub.constants")

    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", offline)
    path = cached(tmp_path, "v2")
    calls = []

    def download(model, **kwargs):
        calls.append((model, kwargs))
        return str(path)

    def original(processor_name, *, revision=None, trust_remote_code=False, **kwargs):
        return processor_name, revision, trust_remote_code, kwargs

    monkeypatch.setattr(adapter, "snapshot_download", download)
    wrapped = adapter._wrap_loader(original)
    result = wrapped(
        "org/model",
        revision="v2",
        cache_dir=str(tmp_path / "cache"),
        token=token,
        trust_remote_code=True,
        custom_option=5,
    )
    assert result == (str(path), "v2", True, {"cache_dir": str(tmp_path / "cache"), "token": token, "custom_option": 5})
    model, kwargs = calls[0]
    assert model == "org/model"
    assert kwargs["revision"] == "v2"
    assert kwargs["cache_dir"] == str(tmp_path / "cache")
    assert kwargs["local_files_only"] is offline
    assert kwargs["token"] == expected
    assert "*.safetensors" in kwargs["ignore_patterns"]
    assert inspect.signature(wrapped) == inspect.signature(original)


def test_loader_preserves_local_and_disabled_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "snapshot_download", lambda *a, **k: pytest.fail("不应下载"))

    def original(processor_name, *, revision=None, **kwargs):
        return processor_name, revision, kwargs

    wrapped = adapter._wrap_loader(original)
    assert wrapped(str(tmp_path), revision="v1") == (str(tmp_path), "v1", {})
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "false")
    assert wrapped("org/model", revision="v1") == ("org/model", "v1", {})


def test_loader_preserves_explicit_offline_and_custom_user_agent(tmp_path, monkeypatch):
    hf_constants = pytest.importorskip("huggingface_hub.constants")
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", False)
    path = cached(tmp_path)
    calls = []

    def download(*args, **kwargs):
        calls.append(kwargs)
        return str(path)

    def original(processor_name, **kwargs):
        return processor_name

    monkeypatch.setattr(adapter, "snapshot_download", download)
    assert adapter._wrap_loader(original)("org/model", local_files_only=True, user_agent="application/example") == str(
        path
    )
    assert calls[0]["local_files_only"] is True
    assert calls[0]["user_agent"] == "invoked_by/vllm; application/example"


def test_metadata_resolver_does_not_download(tmp_path, monkeypatch):
    path = cached(tmp_path, "v2")
    monkeypatch.setattr(adapter, "snapshot_download", lambda *a, **k: pytest.fail("只读解析不应下载"))

    def original(processor_name, revision=None):
        return processor_name, revision

    wrapped = adapter._wrap_loader(original, download=False)
    assert wrapped("org/model", "v2") == (str(path), "v2")
    with pytest.raises(CacheNotFound):
        wrapped("org/model", "absent")


def test_install_is_lazy_and_unrelated_modules_are_untouched():
    before = set(sys.modules)
    adapter.install()
    adapter.install()
    assert sum(isinstance(finder, adapter._Finder) for finder in sys.meta_path) == 1
    assert not {name for name in set(sys.modules) - before if name == "vllm" or name.startswith("vllm.")}
    assert adapter._Finder().find_spec("unrelated_package") is None


def test_already_imported_functions_cached_aliases_and_uninstall(tmp_path, monkeypatch):
    path = cached(tmp_path)
    utils = ModuleType("vllm.transformers_utils.utils")
    processor = ModuleType("vllm.transformers_utils.processor")

    def convert(model_repo):
        return "legacy/" + model_repo

    def load(processor_name, *, revision=None):
        return processor_name, revision

    utils.convert_model_repo_to_path = convert
    processor.convert_model_repo_to_path = convert
    processor.get_processor = load
    processor.cached_get_processor = functools.lru_cache(maxsize=8)(load)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setitem(sys.modules, processor.__name__, processor)
    monkeypatch.setattr(adapter, "snapshot_download", lambda *a, **k: str(path))
    adapter.install()
    first = processor.get_processor
    cached_first = processor.cached_get_processor
    adapter.install()
    assert processor.get_processor is first
    assert processor.cached_get_processor is cached_first
    assert processor.convert_model_repo_to_path is utils.convert_model_repo_to_path
    assert processor.cached_get_processor("org/model", revision="v2") == (str(path), "v2")
    assert processor.cached_get_processor.cache_parameters()["maxsize"] == 8
    adapter.uninstall()
    assert utils.convert_model_repo_to_path is convert
    assert processor.get_processor is load
    assert processor.convert_model_repo_to_path is convert
    assert processor.cached_get_processor.__wrapped__ is load


@pytest.mark.parametrize("offline", [False, True])
def test_config_loader_only_resolves_locally_when_offline(tmp_path, monkeypatch, offline):
    hf_constants = pytest.importorskip("huggingface_hub.constants")
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", offline)
    path = cached(tmp_path, "v2")

    def original(model, trust_remote_code, *, revision=None, hf_token=None):
        return model, trust_remote_code, revision, hf_token

    wrapped = adapter._wrap_config(original)
    assert wrapped("org/model", True, revision="v2", hf_token="test-token") == (
        str(path) if offline else "org/model",
        True,
        "v2",
        "test-token",
    )
    assert inspect.signature(wrapped) == inspect.signature(original)
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "false")
    assert wrapped("org/model", False) == ("org/model", False, None, None)


def test_loader_executes_original_module_before_adapting():
    module = ModuleType("vllm.transformers_utils.utils")

    class Loader:
        def create_module(self, spec):
            return None

        def exec_module(self, target):
            target.convert_model_repo_to_path = lambda name: name

    adapter._Loader(Loader()).exec_module(module)
    assert module.convert_model_repo_to_path._modelscope_hub_adapter
