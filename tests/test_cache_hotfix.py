"""Test cache layout compatibility and cleanup boundaries in pytest sandboxes."""

from pathlib import Path
from unittest.mock import patch

import pytest

from modelscope_hub import HubApi, HubConfig, config, get_cache_root, resolve_cached_snapshot
from modelscope_hub._cache_manager import _cache_clear_targets, _safe_rmtree, clear_cache
from modelscope_hub.errors import CacheError, CacheNotFound


def populate(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    return path


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv("MODELSCOPE_CACHE", str(root))
    monkeypatch.setattr(config, "_default_config", None)
    return root


def test_root_is_read_only_and_config_aware(cache, tmp_path, monkeypatch):
    with patch.object(HubConfig, "__post_init__", side_effect=AssertionError("不得初始化认证")):
        assert get_cache_root() == cache
    assert not cache.exists()
    cfg = HubConfig(cache_dir=tmp_path / "override", config_dir=tmp_path / "auth", token="")
    monkeypatch.setattr(config, "_default_config", cfg)
    assert get_cache_root() == tmp_path / "override"
    assert get_cache_root(cache) == cache


@pytest.mark.parametrize("repo_type", ["model", "dataset"])
@pytest.mark.parametrize("local_dir", [False, True])
def test_offline_snapshot_has_no_transport_or_auth_writes(cache, tmp_path, repo_type, local_dir):
    from modelscope_hub.compat.snapshot_download import dataset_snapshot_download, snapshot_download

    path = populate(cache / f"{repo_type}s/org--model/snapshots/master")
    auth = tmp_path / "auth"
    download = snapshot_download if repo_type == "model" else dataset_snapshot_download
    kwargs = {"local_dir": path} if local_dir else {"cache_dir": cache}
    with (
        patch.object(HubConfig, "__post_init__", side_effect=AssertionError("no auth initialization")),
        patch("requests.Session.send", side_effect=AssertionError("no offline network")),
    ):
        assert download("org/model", local_files_only=True, **kwargs) == str(path)
    assert not auth.exists()
    assert config._default_config is None


def test_config_normalizes_paths(tmp_path):
    cfg = HubConfig(cache_dir=str(tmp_path / "cache"), config_dir=str(tmp_path / "auth"), token="")
    assert isinstance(cfg.cache_dir, Path)
    assert isinstance(cfg.config_dir, Path)
    cfg.ensure_dirs()
    assert cfg.cache_dir.is_dir()


def test_legacy_alias_cache_root(tmp_path, monkeypatch):
    monkeypatch.delenv("MODELSCOPE_CACHE", raising=False)
    monkeypatch.setenv("MS_CACHE_HOME", str(tmp_path / "alias"))
    monkeypatch.setattr(config, "_default_config", None)
    assert get_cache_root() == tmp_path / "alias"
    assert HubConfig(token="").cache_dir == get_cache_root()


@pytest.mark.parametrize("revision", [None, "master", "v1", "feature/test"])
def test_snapshot_uses_real_layout(cache, revision):
    path = populate(cache / "models/org--model.name/snapshots" / (revision or "master"))
    with patch("requests.Session.send", side_effect=AssertionError("不得联网")):
        assert resolve_cached_snapshot("org/model.name", revision=revision) == path


@pytest.mark.parametrize("prefix", ["models", "hub/models"])
def test_old_masked_cache_is_reused(cache, prefix):
    path = populate(cache / prefix / "org/model___name")
    assert resolve_cached_snapshot("org/model.name", revision="v1") == path


def test_missing_revision_is_not_replaced(cache):
    populate(cache / "models/org--model/snapshots/master")
    with pytest.raises(CacheNotFound):
        resolve_cached_snapshot("org/model", revision="main")


def test_default_revision_fallback_and_ambiguity(cache):
    path = populate(cache / "models/org--model/snapshots/v1")
    assert resolve_cached_snapshot("org/model") == path
    populate(cache / "models/org--model/snapshots/v2")
    with pytest.raises(CacheNotFound):
        resolve_cached_snapshot("org/model")


@pytest.mark.parametrize("repo_id", ["", ".", "..", "../..", "/tmp/x", "org/..", "../org", "a/b/c", "a\\b", "a/\0"])
def test_invalid_ids_never_delete(cache, repo_id):
    path = populate(cache / "models/org--model/snapshots/master")
    with patch("modelscope_hub._cache_manager._safe_rmtree") as remove:
        with pytest.raises(CacheError):
            clear_cache(cache, "model", repo_id)
        remove.assert_not_called()
    assert path.is_dir()


@pytest.mark.parametrize("repo_type", ["", "../..", "models", "unknown"])
def test_invalid_types_never_delete(cache, repo_type):
    populate(cache / "models/org--model/snapshots/master")
    with pytest.raises(CacheError):
        clear_cache(cache, repo_type)


def test_empty_root_is_not_default(cache):
    populate(cache / "models/org--model/snapshots/master")
    with pytest.raises(CacheError):
        clear_cache("")


def test_specific_repo_requires_type(cache):
    with pytest.raises(CacheError):
        clear_cache(cache, repo_id="org/model")


def test_clear_by_type_preserves_other_types_and_sdk_data(cache):
    modern = populate(cache / "models/org--model/snapshots/master")
    legacy = populate(cache / "hub/models/org/old___model")
    dataset = populate(cache / "hub/datasets/org/data")
    sdk_data = populate(cache / "hub/ast_indexer")
    flat = populate(cache / "org/ambiguous")
    assert clear_cache(cache, "model") == 4
    assert not modern.exists() and not legacy.exists()
    assert dataset.is_dir() and sdk_data.is_dir() and flat.is_dir()
    assert (cache / "models").is_dir() and (cache / "hub").is_dir()


def test_clear_exact_repo_and_dot_alias(cache):
    path = populate(cache / "hub/models/org/model___name")
    alias = path.with_name("model.name")
    alias.symlink_to(path, target_is_directory=True)
    other = populate(cache / "hub/models/org/other")
    assert clear_cache(cache, "model", "org/model.name") == 2
    assert not path.exists() and not alias.is_symlink()
    assert other.is_dir()


def test_all_targets_validated_before_any_delete(cache, tmp_path):
    valid = populate(cache / "models/org--model/snapshots/master")
    outside = populate(tmp_path / "outside")
    link = cache / "hub/models/org/model"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(CacheError):
        clear_cache(cache, "model", "org/model")
    assert valid.is_dir() and outside.is_dir() and link.is_symlink()


def test_symlinked_parent_rejects_entire_batch(cache, tmp_path):
    populate(cache / "models/org--model/snapshots/master")
    outside = populate(tmp_path / "outside/org/model")
    (cache / "hub").mkdir()
    (cache / "hub/models").symlink_to(outside.parent.parent, target_is_directory=True)
    with pytest.raises(CacheError):
        clear_cache(cache, "model", "org/model")
    assert outside.is_dir()


def test_nested_links_are_unlinked_not_followed(cache, tmp_path):
    path = populate(cache / "models/org--model/snapshots/master")
    outside = populate(tmp_path / "outside")
    (path / "external").symlink_to(outside, target_is_directory=True)
    clear_cache(cache, "model", "org/model")
    assert outside.is_dir() and not path.exists()


def test_dry_run_targets_are_read_only(cache):
    path = populate(cache / "models/org--model/snapshots/master")
    root, targets = _cache_clear_targets(cache, "model", "org/model")
    assert root == cache and targets == [path.parent.parent]
    assert path.is_dir()


@pytest.mark.parametrize("container", ["hub", "models", "models/org", "hub/models", "hub/models/org"])
def test_containers_are_protected(cache, container):
    target = cache / container
    populate(target)
    with pytest.raises(CacheError):
        _safe_rmtree(target, cache)
    assert target.is_dir()


def test_home_and_ancestors_are_protected(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    path = populate(home / "models/org--model/snapshots/master")
    for root in (home, tmp_path):
        with pytest.raises(CacheError):
            clear_cache(root, "model", "org/model") if root == home else _safe_rmtree(home, root)
    assert path.is_dir()


def test_api_uses_its_config_and_preserves_empty_argument(cache, tmp_path):
    cfg = HubConfig(cache_dir=tmp_path / "custom", config_dir=tmp_path / "auth", token="")
    path = populate(cfg.cache_dir / "models/org--model/snapshots/master")
    api = HubApi(config=cfg)
    api.clear_cache(repo_type="model", repo_id="org/model")
    assert not path.exists()
    with pytest.raises(CacheError):
        api.clear_cache(cache_dir="", repo_type="model")


@pytest.mark.parametrize("repo_type", ["model", "dataset"])
@pytest.mark.parametrize("revision", [None, "", "master"])
def test_compat_empty_revision_preserves_default(cache, repo_type, revision):
    from modelscope_hub.compat.snapshot_download import dataset_snapshot_download, snapshot_download

    path = populate(cache / f"{repo_type}s/org--model/snapshots/master")
    download = snapshot_download if repo_type == "model" else dataset_snapshot_download
    with patch.object(HubConfig, "__post_init__", side_effect=AssertionError("不得初始化认证")):
        assert download("org/model", revision=revision, local_files_only=True) == str(path)
    with pytest.raises(CacheError):
        resolve_cached_snapshot("org/model", repo_type, revision="")


@pytest.mark.parametrize("repo_type", ["model", "dataset"])
@pytest.mark.parametrize("local_dir", [False, True])
@pytest.mark.parametrize("revision", [None, "", "master", "feature/test"])
def test_offline_file_has_no_transport_or_auth_writes(cache, repo_type, local_dir, revision):
    from modelscope_hub.compat.file_download import dataset_file_download, model_file_download

    path = populate(cache / f"{repo_type}s/org--model/snapshots" / (revision or "master"))
    download = model_file_download if repo_type == "model" else dataset_file_download
    kwargs = {"local_dir": str(path)} if local_dir else {"cache_dir": str(cache)}
    with (
        patch.object(HubConfig, "__post_init__", side_effect=AssertionError("不得初始化认证")),
        patch("requests.Session.send", side_effect=AssertionError("不得联网")),
    ):
        assert download("org/model", "config.json", revision=revision, local_files_only=True, **kwargs) == str(
            path / "config.json"
        )
    assert config._default_config is None


@pytest.mark.parametrize("prefix", ["models", "hub/models"])
def test_offline_file_reuses_legacy_cache(cache, prefix):
    from modelscope_hub.compat.file_download import model_file_download

    path = populate(cache / prefix / "org/model")
    with patch.object(HubConfig, "__post_init__", side_effect=AssertionError("不得初始化认证")):
        assert model_file_download("org/model", "config.json", local_files_only=True) == str(path / "config.json")


def test_offline_file_missing_revision_is_not_replaced(cache):
    from modelscope_hub.compat.file_download import model_file_download

    populate(cache / "models/org--model/snapshots/master")
    with pytest.raises(CacheNotFound):
        model_file_download("org/model", "config.json", revision="main", local_files_only=True)


@pytest.mark.parametrize("repo_type", ["model", "dataset"])
@pytest.mark.parametrize("kwargs", [{"continue_on_error": True}, {"tracker_path": "x"}, {"retry_failed": True}])
def test_offline_report_conflicts_use_invalid_parameter(cache, repo_type, kwargs):
    from modelscope_hub.compat.snapshot_download import dataset_snapshot_download, snapshot_download
    from modelscope_hub.errors import InvalidParameter

    download = snapshot_download if repo_type == "model" else dataset_snapshot_download
    with pytest.raises(InvalidParameter):
        download("org/model", local_files_only=True, **kwargs)


@pytest.mark.parametrize("prefix", ["models", "hub/models"])
def test_scan_detects_masked_legacy_without_counting_alias(cache, prefix):
    from modelscope_hub._cache_manager import scan_cache

    path = populate(cache / prefix / "org/model___name")
    path.with_name("model.name").symlink_to(path, target_is_directory=True)
    with patch.object(HubConfig, "__post_init__", side_effect=AssertionError("不得初始化认证")):
        info = scan_cache(cache)
    assert info.total_repos == 1
    assert info.repos[0].repo_id == "org/model.name"
    assert info.total_size == 2
