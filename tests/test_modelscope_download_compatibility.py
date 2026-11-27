"""Verify download counting and legacy API compatibility through the real modelscope shims; mock only HTTP."""

from __future__ import annotations

import importlib
import inspect
from functools import partial
from pathlib import Path

import pytest
import requests
import responses

from modelscope_hub import _download, config
from modelscope_hub.errors import CacheNotFound
from tests.test_download_integrity import REPO
from tests.test_download_integrity import hub as hub


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELSCOPE_HOME", str(tmp_path / "sdk-home"))
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "")
    monkeypatch.setattr(config, "_default_config", None)


@pytest.fixture
def sdk():
    pytest.importorskip("modelscope")
    return (
        importlib.import_module("modelscope.hub.snapshot_download"),
        importlib.import_module("modelscope.hub.file_download"),
    )


def _requests(hub):
    return [r for r in hub.requests if r.kind == "api"]


@pytest.mark.parametrize("repo_type", ["model", "dataset"])
@pytest.mark.parametrize("legacy_filters", [False, True])
def test_snapshot_positional_arguments_filters_callbacks_and_cache(sdk, hub, tmp_path, repo_type, legacy_filters):
    snapshots, _ = sdk
    from modelscope.hub.callback import ProgressCallback

    ended = []

    class Callback(ProgressCallback):
        def end(self):
            ended.append(self.filename)

    hub.add("selected/a.bin", b"content-a")
    hub.add("selected/b.txt", b"content-b", mode="direct")
    hub.add("selected/skip.bin", b"skip")
    hub.add("other/c.bin", b"other")
    include = "allow_file_pattern" if legacy_filters else "allow_patterns"
    exclude = "ignore_file_pattern" if legacy_filters else "ignore_patterns"
    kwargs = {
        include: "selected/*",
        exclude: "selected/skip.bin",
        "repo_type": repo_type,
        "endpoint": hub.url,
        "token": "",
        "progress_callbacks": [Callback],
        "max_workers": 2,
    }
    cache = tmp_path / "cache"
    first = snapshots.snapshot_download(REPO, "v-compat", cache, **kwargs)
    assert isinstance(first, str)
    assert Path(first).parts[-1] == "v-compat"
    assert (Path(first) / "selected/a.bin").read_bytes() == b"content-a"
    assert sorted(ended) == ["selected/a.bin", "selected/b.txt"]
    sent = _requests(hub)
    assert {r.path for r in sent} == {"selected/a.bin", "selected/b.txt"}
    identifiers = {r.snapshot for r in sent}
    assert len(identifiers) == 1 and None not in identifiers
    assert all("invoked_by/compat" in r.user_agent for r in sent)
    hub.requests.clear()
    assert snapshots.snapshot_download(REPO, "v-compat", cache, **kwargs) == first
    assert not _requests(hub)


def test_dataset_snapshot_positional_local_dir_and_repo_alias(sdk, hub, tmp_path):
    snapshots, _ = sdk
    hub.add("a.bin", b"dataset")
    output = tmp_path / "local"
    result = snapshots.dataset_snapshot_download(
        None, "master", tmp_path / "cache", str(output), repo_id=REPO, token="", endpoint=hub.url
    )
    assert result == str(output)
    assert (output / "a.bin").read_bytes() == b"dataset"
    assert len({r.snapshot for r in _requests(hub)}) == 1


@pytest.mark.parametrize("name", ["model_file_download", "dataset_file_download"])
@pytest.mark.parametrize("revision", ["master", ""])
def test_single_file_signature_and_identifier_absence(sdk, hub, tmp_path, name, revision):
    _, files = sdk
    hub.add("a.bin", b"single")
    result = getattr(files, name)(
        REPO, "a.bin", revision=revision, local_dir=str(tmp_path / "out"), endpoint=hub.url, token=""
    )
    assert isinstance(result, str) and Path(result).read_bytes() == b"single"
    assert {r.snapshot for r in _requests(hub)} == {None}
    assert all("invoked_by/compat" in r.user_agent for r in _requests(hub))


def test_modelscope_hubapi_download_model_delegates_to_task_context(sdk, hub, tmp_path):
    from modelscope.hub.api import HubApi

    hub.add("a.bin", b"model")
    hub.add("config.json", b"{}", mode="direct")
    api = HubApi(endpoint=hub.url, token="")
    output = api.download_model(REPO, revision="master", local_dir=str(tmp_path / "out"))
    assert isinstance(output, str)
    assert (Path(output) / "a.bin").read_bytes() == b"model"
    assert len({r.snapshot for r in _requests(hub)}) == 1
    assert all(r.snapshot for r in _requests(hub))


def test_dataset_snapshot_retries_share_one_identifier(sdk, hub, tmp_path, monkeypatch):
    snapshots, _ = sdk
    monkeypatch.setattr(_download, "DOWNLOAD_CHUNK_SIZE", 8192)
    content = b"data" * 50000
    hub.add("a.bin", content, truncate_first=70000)
    output = snapshots.dataset_snapshot_download(REPO, local_dir=str(tmp_path / "out"), endpoint=hub.url, token="")
    assert (Path(output) / "a.bin").read_bytes() == content
    sent = _requests(hub)
    assert len(sent) == 2
    assert sent[1].range == "bytes=70000-"
    assert len({r.snapshot for r in sent}) == 1


@pytest.mark.parametrize("entry", ["model_snapshot", "dataset_snapshot", "model_file", "dataset_file"])
@pytest.mark.parametrize("revision", [None, "", "master", "missing-revision"])
def test_offline_shims_do_not_attempt_network_credentials_or_writes(sdk, tmp_path, monkeypatch, entry, revision):
    snapshots, files = sdk
    repo_type = "model" if entry.startswith("model") else "dataset"
    cache = tmp_path / "cache"
    output = cache / f"{repo_type}s" / "owner--data" / "snapshots/master"
    output.mkdir(parents=True)
    (output / "a.bin").write_bytes(b"offline")
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("offline entry points must not touch the network, credentials, or the filesystem")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", forbidden)
    monkeypatch.setattr(config.HubConfig, "load_token", forbidden)
    monkeypatch.setattr(config.HubConfig, "get_session_id", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    kwargs = dict(revision=revision, cache_dir=str(cache), local_files_only=True)
    if entry.endswith("snapshot"):
        call = partial(snapshots.snapshot_download, REPO, repo_type=repo_type, **kwargs)
        expected = str(output)
    else:
        function = files.model_file_download if repo_type == "model" else files.dataset_file_download
        call = partial(function, REPO, "a.bin", **kwargs)
        expected = str(output / "a.bin")
    if revision == "missing-revision":
        with pytest.raises(CacheNotFound):
            call()
    else:
        assert call() == expected
    assert not attempts


@pytest.mark.parametrize("repo_type", ["model", "dataset"])
@pytest.mark.parametrize("layout", ["flat", "hub"])
def test_sdk_reuses_legacy_offline_snapshot_without_network(sdk, tmp_path, monkeypatch, repo_type, layout):
    snapshots, _ = sdk
    cache = tmp_path / "cache"
    legacy = cache / f"{repo_type}s" / "owner/data"
    if layout == "hub":
        legacy = cache / "hub" / f"{repo_type}s" / "owner/data"
    legacy.mkdir(parents=True)
    (legacy / "a.bin").write_bytes(b"legacy")
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("legacy cache resolution must not touch the network, credentials, or the filesystem")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", forbidden)
    monkeypatch.setattr(config.HubConfig, "load_token", forbidden)
    monkeypatch.setattr(config.HubConfig, "get_session_id", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    result = snapshots.snapshot_download(REPO, "master", cache, repo_type=repo_type, local_files_only=True)
    assert result == str(legacy)
    assert not attempts


@pytest.mark.parametrize("name", ["model_file_download", "dataset_file_download"])
@responses.activate
def test_missing_file_exception_is_preserved_at_the_sdk_boundary(sdk, name, tmp_path, monkeypatch):
    _, files = sdk
    monkeypatch.setenv("MODELSCOPE_DOWNLOAD_INTRA_CLOUD", "false")
    segment = "models" if name == "model_file_download" else "datasets"
    url = f"https://modelscope.cn/api/v1/{segment}/{REPO}/repo"
    responses.add(responses.GET, url, status=404, json={"Code": 404, "Message": "not found"})
    with pytest.raises(requests.HTTPError):
        getattr(files, name)(
            REPO,
            "missing.bin",
            revision="master",
            endpoint="https://modelscope.cn",
            token="",
            local_dir=str(tmp_path / "out"),
        )
    assert len(responses.calls) == 1
    assert "snapshot-identifier" not in responses.calls[0].request.headers


def test_report_options_are_not_silently_injected_into_unsupported_shims(sdk):
    snapshots, _ = sdk
    if "continue_on_error" in inspect.signature(snapshots.snapshot_download).parameters:
        pytest.skip("the shim now accepts report options; extend this test to cover the supported path")
    with pytest.raises(TypeError):
        snapshots.snapshot_download(REPO, continue_on_error=True)


def test_snapshot_module_keeps_importable_invalid_parameter(sdk):
    snapshots, _ = sdk
    from modelscope.hub.errors import InvalidParameter

    assert snapshots.InvalidParameter is InvalidParameter
