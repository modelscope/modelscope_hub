"""Network-free tests for the huggingface_hub monkey-patcher in ``modelscope_hub.compat``.

The patch functions are exercised with mocked ``HubApi``/``LegacyHubApi``/
``compat`` downloads; the tests run only when the optional ``huggingface_hub``
package is installed.
"""

from __future__ import annotations

from unittest import mock

import pytest

from modelscope_hub import FileInfo
from modelscope_hub.compat import hf_hub_patcher
from modelscope_hub.compat.hf_hub_patcher import (
    HfCommitInfo,
    HfRepoUrl,
    _create_commit,
    _create_repo,
    _file_exists,
    _hf_hub_download,
    _upload_file,
    _upload_folder,
    _whoami,
    hf_hub_patch_scope,
    patch_hf_hub,
    unpatch_hf_hub,
)

hf_api_mod = pytest.importorskip("huggingface_hub.hf_api")
hf_top = pytest.importorskip("huggingface_hub")
repocard = pytest.importorskip("huggingface_hub.repocard")


@pytest.fixture
def patched_hub():
    patch_hf_hub()
    yield
    unpatch_hf_hub()


# ---------------------------------------------------------------------------
# _hf_hub_download
# ---------------------------------------------------------------------------
def test_hf_hub_download_routes_model_file_download():
    with mock.patch("modelscope_hub.compat.model_file_download", return_value="/cache/x.json") as dl:
        result = _hf_hub_download(
            "foo/bar",
            "config.json",
            revision="main",
            cache_dir="/tmp/cache",
            local_dir="/tmp/out",
            token="ms-token",
            local_files_only=False,
        )

    assert result == "/cache/x.json"
    assert dl.call_args.args == ("foo/bar", "config.json")
    kwargs = dl.call_args.kwargs
    # `main` is normalized to `master`; string tokens pass through.
    assert kwargs["revision"] == "master"
    assert kwargs["cache_dir"] == "/tmp/cache"
    assert kwargs["local_dir"] == "/tmp/out"
    assert kwargs["token"] == "ms-token"
    assert kwargs["local_files_only"] is False


def test_hf_hub_download_dataset_routes_dataset_download():
    with mock.patch("modelscope_hub.compat.dataset_file_download", return_value="/cache/x.bin") as dl:
        result = _hf_hub_download("foo/data", "train.bin", repo_type="dataset", revision="v2")

    assert result == "/cache/x.bin"
    assert dl.call_args.args == ("foo/data", "train.bin")
    assert dl.call_args.kwargs["revision"] == "v2"


def test_hf_hub_download_rejects_unsupported_repo_type():
    with pytest.raises(ValueError, match="repo_type=space"):
        _hf_hub_download("foo/bar", "config.json", repo_type="space")


def test_hf_hub_download_bool_token_is_dropped():
    with mock.patch("modelscope_hub.compat.model_file_download", return_value="/cache/x") as dl:
        _hf_hub_download("foo/bar", "config.json", token=True)

    assert dl.call_args.kwargs["token"] is None


def test_hf_hub_download_subfolder_is_joined():
    with mock.patch("modelscope_hub.compat.model_file_download", return_value="/cache/x") as dl:
        _hf_hub_download("foo/bar", "nested.txt", subfolder="configs")

    assert dl.call_args.args == ("foo/bar", "configs/nested.txt")


# ---------------------------------------------------------------------------
# _file_exists / _whoami
# ---------------------------------------------------------------------------
def test_file_exists_uses_listing():
    fake = mock.MagicMock()
    fake.list_repo_files.return_value = [
        FileInfo(path="README.md", size=1),
        FileInfo(path="config.json", size=2),
    ]
    with mock.patch("modelscope_hub.HubApi", return_value=fake):
        assert _file_exists("foo/bar", "config.json", revision="main")
        assert not _file_exists("foo/bar", "missing.txt", revision="main")

    # `main` is normalized to `master`.
    call = fake.list_repo_files.call_args
    assert call.args == ("foo/bar", "model")
    assert call.kwargs["revision"] == "master"


def test_whoami_returns_name_dict():
    fake = mock.MagicMock()
    fake.whoami.return_value.username = "alice"
    with mock.patch("modelscope_hub.HubApi", return_value=fake):
        assert _whoami(token="ms-token") == {"name": "alice"}

    fake.whoami.assert_called_once()


# ---------------------------------------------------------------------------
# _create_repo / _upload_file / _upload_folder / _create_commit
# ---------------------------------------------------------------------------
def test_create_repo_returns_hf_repo_url():
    with mock.patch("modelscope_hub.compat.LegacyHubApi") as legacy:
        result = _create_repo("foo/bar", token="ms-token", private=True)

    legacy.assert_called_once_with(token="ms-token")
    legacy.return_value.create_repo.assert_called_once_with("foo/bar", visibility="private")

    # Legacy duck-typing shape: a str URL plus repo_id / repo_type attributes
    # (hf's own RepoUrl rejects non-HF hosts on recent releases).
    assert isinstance(result, HfRepoUrl)
    assert "foo/bar" in result
    assert result.repo_id == "foo/bar"
    assert result.repo_type == "model"


def test_upload_file_forwards_to_hub_upload():
    fake = mock.MagicMock()
    with mock.patch("modelscope_hub.HubApi", return_value=fake):
        _upload_file(
            path_or_fileobj="/tmp/w.safetensors",
            path_in_repo="w.safetensors",
            repo_id="foo/bar",
            token="ms-token",
            revision="main",
            commit_message="add weights",
            commit_description="tuned",
        )

    fake.upload_file.assert_called_once_with(
        "foo/bar",
        "model",
        "/tmp/w.safetensors",
        "w.safetensors",
        commit_message="add weights",
        commit_description="tuned",
        revision="master",
    )


def test_upload_folder_returns_legacy_commit_info_shape():
    fake = mock.MagicMock()
    with mock.patch("modelscope_hub.HubApi", return_value=fake):
        result = _upload_folder(
            repo_id="foo/bar",
            folder_path="/tmp/out",
            commit_message="sync",
            ignore_patterns=["*.tmp"],
            revision="v2",
        )

    fake.upload_folder.assert_called_once_with(
        "foo/bar",
        "model",
        "/tmp/out",
        path_in_repo="",
        commit_message="sync",
        commit_description=None,
        revision="v2",
        ignore_patterns=["*.tmp"],
        allow_patterns=None,
    )
    assert isinstance(result, HfCommitInfo)
    assert result.commit_url.endswith("/models/foo/bar/files")
    assert result.commit_message == "sync"
    assert result.oid is None


@pytest.mark.parametrize("patch_fn", [_upload_file, _upload_folder])
def test_upload_rejects_non_model_repo_type(patch_fn):
    kwargs = (
        dict(path_or_fileobj="/tmp/a", path_in_repo="a", repo_id="foo/data")
        if patch_fn is _upload_file
        else dict(repo_id="foo/data", folder_path="/tmp/out")
    )
    with mock.patch("modelscope_hub.HubApi"):
        with pytest.raises(ValueError, match="repo_type=dataset"):
            patch_fn(repo_type="dataset", **kwargs)


def test_create_commit_rejects_non_add_operations():
    from huggingface_hub import CommitOperationDelete

    fake = mock.MagicMock()
    with mock.patch("modelscope_hub.HubApi", return_value=fake):
        with pytest.raises(ValueError, match="only supports Add"):
            _create_commit("foo/bar", [CommitOperationDelete("old.txt", False)], commit_message="x")

    fake.upload_file.assert_not_called()


def test_create_commit_maps_each_add_operation(tmp_path):
    from huggingface_hub import CommitOperationAdd

    # hf >= 1.x verifies that str/Path sources exist on the local filesystem.
    a = tmp_path / "a.txt"
    a.write_text("a", encoding="utf-8")
    b = tmp_path / "b.txt"
    b.write_text("b", encoding="utf-8")

    fake = mock.MagicMock()
    with mock.patch("modelscope_hub.HubApi", return_value=fake):
        _create_commit(
            "foo/bar",
            [
                CommitOperationAdd(path_in_repo="a.txt", path_or_fileobj=str(a)),
                CommitOperationAdd(path_in_repo="b.txt", path_or_fileobj=b"bytes"),
            ],
            commit_message="batch",
            revision="main",
        )

    assert fake.upload_file.call_count == 2
    first, second = fake.upload_file.call_args_list
    assert first.args == ("foo/bar", "model", str(a), "a.txt")
    assert second.args == ("foo/bar", "model", b"bytes", "b.txt")
    assert first.kwargs["commit_message"] == "batch"
    assert first.kwargs["revision"] == "master"


def test_create_commit_rejects_non_model_repo_type(tmp_path):
    from huggingface_hub import CommitOperationAdd

    a = tmp_path / "a.txt"
    a.write_text("a", encoding="utf-8")

    with mock.patch("modelscope_hub.HubApi"):
        with pytest.raises(ValueError, match="repo_type=dataset"):
            _create_commit(
                "foo/data",
                [CommitOperationAdd(path_in_repo="a.txt", path_or_fileobj=str(a))],
                commit_message="x",
                repo_type="dataset",
            )


# ---------------------------------------------------------------------------
# repocard
# ---------------------------------------------------------------------------
def test_repocard_load_routes_to_compat_download(tmp_path):
    from huggingface_hub.repocard import RepoCard

    with mock.patch(
        "modelscope_hub.compat.model_file_download",
        return_value=str(tmp_path / "README.md"),
    ) as dl:
        (tmp_path / "README.md").write_text("---\nlicense: apache-2.0\n---\nhello", encoding="utf-8")
        card = hf_hub_patcher._repocard_load(RepoCard, "foo/bar")

    assert dl.call_args.args == ("foo/bar", "README.md")
    assert card.data.tags == []


# ---------------------------------------------------------------------------
# Patch round-trip
# ---------------------------------------------------------------------------
def test_patch_roundtrip_restores_all_symbols():
    original = hf_top.hf_hub_download
    original_file_download = hf_top.file_download.hf_hub_download

    patch_hf_hub()
    try:
        assert hf_top.hf_hub_download is hf_hub_patcher._hf_hub_download
        assert hf_top.file_exists is hf_hub_patcher._file_exists
        assert hf_top.whoami is hf_hub_patcher._whoami
        assert hf_top.create_repo is hf_hub_patcher._create_repo
        assert hf_top.upload_file is hf_hub_patcher._upload_file
        assert hf_top.upload_folder is hf_hub_patcher._upload_folder
        assert hf_top.create_commit is hf_hub_patcher._create_commit
        # repocard class patches are active while patched.
        assert repocard.RepoCard.__dict__["load"].__func__ is hf_hub_patcher._repocard_load
    finally:
        unpatch_hf_hub()

    assert hf_top.hf_hub_download is original
    assert hf_top.file_download.hf_hub_download is original_file_download
    assert not hasattr(hf_api_mod, "_hf_hub_download_origin")
    assert not hasattr(hf_api_mod, "_file_exists_origin")
    assert not hasattr(hf_api_mod, "_whoami_origin")
    assert not hasattr(hf_api_mod, "_create_repo_origin")
    assert not hasattr(hf_api_mod, "_upload_folder_origin")
    assert not hasattr(hf_api_mod, "_upload_file_origin")
    assert not hasattr(hf_api_mod, "_create_commit_origin")
    assert not hasattr(repocard.RepoCard, "_validate_origin")
    assert not hasattr(repocard.RepoCard, "_load_origin")


def test_patch_is_idempotent(patched_hub):
    first = hf_top.hf_hub_download
    patch_hf_hub()
    assert hf_top.hf_hub_download is first


def test_patch_respects_legacy_markers():
    """A pre-existing legacy patch (same marker names) must not be overwritten."""
    fake_legacy = mock.MagicMock(name="legacy_download")
    with mock.patch.object(hf_top, "hf_hub_download", fake_legacy):
        setattr(hf_api_mod, "_hf_hub_download_origin", fake_legacy)
        try:
            patch_hf_hub()
            assert hf_top.hf_hub_download is fake_legacy
            # The already-patched function stays; markers unchanged.
            assert hf_api_mod._hf_hub_download_origin is fake_legacy
        finally:
            delattr(hf_api_mod, "_hf_hub_download_origin")


def test_patch_scope_restores_on_exit():
    original = hf_top.hf_hub_download

    with hf_hub_patch_scope():
        assert hf_top.hf_hub_download is hf_hub_patcher._hf_hub_download
        with hf_hub_patch_scope():
            assert hf_top.hf_hub_download is hf_hub_patcher._hf_hub_download
        # The inner scope must not undo the outer patch.
        assert hf_top.hf_hub_download is hf_hub_patcher._hf_hub_download

    assert hf_top.hf_hub_download is original


def test_module_has_no_transformers_import():
    """The hub-side patcher must stay importable without transformers."""
    import inspect

    source = inspect.getsource(hf_hub_patcher)
    assert "import transformers" not in source
    assert "from transformers" not in source


# ---------------------------------------------------------------------------
# huggingface_hub is not a declared dependency: fail at call time
# ---------------------------------------------------------------------------
def test_patch_raises_actionable_error_without_hf_hub(monkeypatch):
    """No `hf` extra exists: patch_hf_hub fails with install instructions."""
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match="pip install huggingface_hub"):
        patch_hf_hub()


def test_unpatch_is_noop_without_hf_hub(monkeypatch):
    """Nothing can have been patched, so unpatch stays a silent no-op."""
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    unpatch_hf_hub()  # must not raise


def test_patch_scope_raises_actionable_error_without_hf_hub(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match="pip install huggingface_hub"):
        with hf_hub_patch_scope():
            pass
