"""Network-free tests for the ``modelscope_hub.kernels`` ModelScope-backed shim.

``MsKernelApi`` is exercised with an injected fake ``HubApi``; the patch
round-trip tests run only when the optional ``kernels`` package (and its
``huggingface_hub`` dependency) is installed.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from modelscope_hub import FileInfo
from modelscope_hub.errors import CacheNotFound, NotExistError
from modelscope_hub.kernels import get_kernel as kernels_get_kernel
from modelscope_hub.kernels._hub import MsKernelApi, _ms_repo_type, _ms_revision
from modelscope_hub.kernels._patch import _ms_api, patch_kernels_hub, patch_scope, unpatch_kernels_hub


# ---------------------------------------------------------------------------
# Revision / repo-type mapping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("revision", "expected"),
    [(None, "master"), ("main", "master"), ("master", "master"), ("v1", "v1"), ("deadbeef", "deadbeef")],
)
def test_ms_revision_maps_hf_defaults(revision, expected):
    assert _ms_revision(revision) == expected


@pytest.mark.parametrize(
    ("repo_type", "expected"),
    [(None, "model"), ("kernel", "model"), ("model", "model"), ("dataset", "dataset")],
)
def test_ms_repo_type_maps_kernel_to_model(repo_type, expected):
    assert _ms_repo_type(repo_type) == expected


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------
def test_snapshot_download_forwards_to_download_repo():
    fake = mock.MagicMock()
    fake.download_repo.return_value = Path("/cache/snapshots/abc")
    api = MsKernelApi(hub_api=fake)

    result = api.snapshot_download(
        "kernels-community/activation",
        repo_type="kernel",
        allow_patterns=["build/torch-cpu/*"],
        ignore_patterns=["*.pyc"],
        cache_dir="/tmp/kernels-cache",
        revision="v1",
        local_files_only=False,
    )

    assert result == Path("/cache/snapshots/abc")
    fake.download_repo.assert_called_once_with(
        "kernels-community/activation",
        "model",
        revision="v1",
        cache_dir=Path("/tmp/kernels-cache"),
        allow_patterns=["build/torch-cpu/*"],
        ignore_patterns=["*.pyc"],
        local_files_only=False,
        user_agent=mock.ANY,
    )


def test_snapshot_download_cache_miss_raises_local_entry_not_found():
    hf_errors = pytest.importorskip("huggingface_hub.errors")

    fake = mock.MagicMock()
    fake.download_repo.side_effect = CacheNotFound("no snapshot")
    api = MsKernelApi(hub_api=fake)

    with pytest.raises(hf_errors.LocalEntryNotFoundError):
        api.snapshot_download("kernels-community/activation", repo_type="kernel", revision="v1")


def test_hf_hub_download_forwards_to_download_file():
    fake = mock.MagicMock()
    fake.download_file.return_value = Path("/cache/build/torch-cpu/metadata.json")
    api = MsKernelApi(hub_api=fake)

    result = api.hf_hub_download(
        "kernels-community/activation",
        repo_type="kernel",
        filename="build/torch-cpu/metadata.json",
        cache_dir="/tmp/kernels-cache",
        revision="v1",
        local_files_only=False,
    )

    assert result == Path("/cache/build/torch-cpu/metadata.json")
    fake.download_file.assert_called_once_with(
        "kernels-community/activation",
        "model",
        file_path="build/torch-cpu/metadata.json",
        revision="v1",
        cache_dir=Path("/tmp/kernels-cache"),
        local_files_only=False,
        user_agent=mock.ANY,
    )


def test_hf_hub_download_missing_file_raises_entry_not_found():
    hf_errors = pytest.importorskip("huggingface_hub.errors")

    fake = mock.MagicMock()
    fake.download_file.side_effect = NotExistError("no such file")
    api = MsKernelApi(hub_api=fake)

    with pytest.raises(hf_errors.EntryNotFoundError):
        api.hf_hub_download(
            "kernels-community/activation",
            repo_type="kernel",
            filename="kernel-status.toml",
            revision="v1",
        )


def test_hf_hub_download_cache_miss_raises_local_entry_not_found():
    hf_errors = pytest.importorskip("huggingface_hub.errors")

    fake = mock.MagicMock()
    fake.download_file.side_effect = CacheNotFound("no cached file")
    api = MsKernelApi(hub_api=fake)

    with pytest.raises(hf_errors.LocalEntryNotFoundError):
        api.hf_hub_download(
            "kernels-community/activation",
            repo_type="kernel",
            filename="kernel-status.toml",
            revision="v1",
            local_files_only=True,
        )


# ---------------------------------------------------------------------------
# Revision listing
# ---------------------------------------------------------------------------
def test_list_repo_refs_builds_hf_ref_objects():
    hf_api = pytest.importorskip("huggingface_hub.hf_api")

    fake = mock.MagicMock()
    fake.legacy.list_revisions_detail.return_value = (
        [{"Revision": "v1"}, {"Revision": "v2"}, {"Revision": "master"}],
        [{"Revision": "1.0"}],
    )
    api = MsKernelApi(hub_api=fake)

    refs = api.list_repo_refs("kernels-community/activation", repo_type="kernel")

    fake.legacy.list_revisions_detail.assert_called_once_with("kernels-community/activation", "model")
    assert [b.name for b in refs.branches] == ["v1", "v2", "master"]
    assert [t.name for t in refs.tags] == ["1.0"]
    assert all(isinstance(b, hf_api.GitRefInfo) for b in refs.branches)
    # target_commit is a synthetic hex OID (kernels 0.18 validates it with
    # Oid.from_str); the download entry points map it back to the ref name.
    import re as _re

    assert all(_re.fullmatch(r"[0-9a-f]{40}", b.target_commit) for b in refs.branches)
    assert refs.converts == []


def test_list_repo_refs_synthesizes_v1_when_no_version_refs():
    """A plain ModelScope mirror with only `master` exposes it as kernel version v1."""
    pytest.importorskip("huggingface_hub.hf_api")
    import re as _re

    fake = mock.MagicMock()
    fake.legacy.list_revisions_detail.return_value = ([{"Revision": "master"}], [])
    api = MsKernelApi(hub_api=fake)

    refs = api.list_repo_refs("kernels-community/activation", repo_type="kernel")

    v1 = [b for b in refs.branches if b.name == "v1"]
    assert len(v1) == 1
    # The synthetic v1 ref carries a hex OID that resolves downloads back
    # onto `master` (see test_synthetic_oid_resolved_back_to_branch_on_downloads).
    assert _re.fullmatch(r"[0-9a-f]{40}", v1[0].target_commit)


# ---------------------------------------------------------------------------
# Tree listing
# ---------------------------------------------------------------------------
def test_list_repo_tree_derives_direct_folders_from_blob_paths():
    hf_api = pytest.importorskip("huggingface_hub.hf_api")

    fake = mock.MagicMock()
    fake.list_repo_files.return_value = [
        FileInfo(path="README.md", size=10),
        FileInfo(path="build/torch-cpu/kernel.py", size=20),
        FileInfo(path="build/torch-cpu/sub/deep.py", size=5),
        FileInfo(path="build/torch215-cuda124-aarch64-linux/kernel.py", size=30),
    ]
    api = MsKernelApi(hub_api=fake)

    tree = api.list_repo_tree("kernels-community/activation", path_in_repo="build", repo_type="kernel", revision="v1")

    fake.list_repo_files.assert_called_once_with("kernels-community/activation", "model", revision="v1", recursive=True)
    assert [item.path for item in tree] == [
        "build/torch-cpu",
        "build/torch215-cuda124-aarch64-linux",
    ]
    assert all(isinstance(item, hf_api.RepoFolder) for item in tree)


def test_list_repo_tree_uses_explicit_tree_entries():
    pytest.importorskip("huggingface_hub.hf_api")

    fake = mock.MagicMock()
    fake.list_repo_files.return_value = [
        FileInfo(path="build/torch-cpu", type="tree"),
        FileInfo(path="build/torch-cpu/kernel.py", size=20),
    ]
    api = MsKernelApi(hub_api=fake)

    tree = api.list_repo_tree("kernels-community/activation", path_in_repo="build", revision="v1")

    assert [item.path for item in tree] == ["build/torch-cpu"]


def test_file_exists_checks_listing():
    fake = mock.MagicMock()
    fake.list_repo_files.return_value = [
        FileInfo(path="README.md", size=10),
        FileInfo(path="build/torch-cpu/kernel.py", size=20),
    ]
    api = MsKernelApi(hub_api=fake)

    assert api.file_exists("kernels-community/activation", "build/torch-cpu/kernel.py", revision="v1")
    assert not api.file_exists("kernels-community/activation", "missing.py", revision="v1")


# ---------------------------------------------------------------------------
# Publisher trust
# ---------------------------------------------------------------------------
def test_trusted_publisher_allowlist():
    api = MsKernelApi(hub_api=mock.MagicMock())

    assert api.get_organization_overview("kernels-community").trustedKernelPublisher is True
    assert api.get_organization_overview("someone-else").trustedKernelPublisher is False


def test_trusted_publisher_allowlist_env_extension(monkeypatch):
    monkeypatch.setenv("MS_KERNELS_TRUSTED_PUBLISHERS", "my-org, other-org")
    api = MsKernelApi(hub_api=mock.MagicMock())

    assert api.get_organization_overview("my-org").trustedKernelPublisher is True
    assert api.get_organization_overview("other-org").trustedKernelPublisher is True
    assert api.get_organization_overview("kernels-community").trustedKernelPublisher is True


# ---------------------------------------------------------------------------
# Patch round-trip (requires the optional `kernels` package)
# ---------------------------------------------------------------------------
@pytest.fixture
def kernels_modules():
    kernels = pytest.importorskip("kernels")
    kernels_hf_hub = pytest.importorskip("kernels.hf_hub")
    kernels_load = pytest.importorskip("kernels.load")

    return kernels, kernels_hf_hub, kernels_load


def test_patch_kernels_hub_roundtrip(kernels_modules):
    _, kernels_hf_hub, kernels_load = kernels_modules
    original = kernels_hf_hub._get_hf_api
    original_load = kernels_load._get_hf_api

    patch_kernels_hub()
    try:
        assert kernels_hf_hub._get_hf_api is _ms_api
        assert kernels_load._get_hf_api is _ms_api
        assert isinstance(kernels_hf_hub._get_hf_api(), MsKernelApi)
    finally:
        unpatch_kernels_hub()

    assert kernels_hf_hub._get_hf_api is original
    assert kernels_load._get_hf_api is original_load


def test_patch_kernels_hub_is_idempotent(kernels_modules):
    kernels_hf_hub = kernels_modules[1]
    original = kernels_hf_hub._get_hf_api

    patch_kernels_hub()
    try:
        first = kernels_hf_hub._get_hf_api
        patch_kernels_hub()
        assert kernels_hf_hub._get_hf_api is first
    finally:
        unpatch_kernels_hub()

    assert kernels_hf_hub._get_hf_api is original


def test_patch_scope_restores_on_exit_and_nests(kernels_modules):
    kernels_hf_hub = kernels_modules[1]
    original = kernels_hf_hub._get_hf_api

    with patch_scope():
        assert kernels_hf_hub._get_hf_api is _ms_api
        with patch_scope():
            assert kernels_hf_hub._get_hf_api is _ms_api
        # The inner scope must not undo the outer patch.
        assert kernels_hf_hub._get_hf_api is _ms_api

    assert kernels_hf_hub._get_hf_api is original


def test_get_kernel_wrapper_scopes_the_patch(kernels_modules, monkeypatch):
    kernels, kernels_hf_hub, _ = kernels_modules
    original = kernels_hf_hub._get_hf_api
    recorded = {}

    def fake_get_kernel(repo_id, **kwargs):
        recorded["repo_id"] = repo_id
        recorded["kwargs"] = kwargs
        recorded["patched"] = kernels_hf_hub._get_hf_api is _ms_api
        return "kernel-module"

    monkeypatch.setattr(kernels, "get_kernel", fake_get_kernel)

    result = kernels_get_kernel(
        "kernels-community/activation",
        version=1,
        trust_remote_code=True,
    )

    assert result == "kernel-module"
    assert recorded["repo_id"] == "kernels-community/activation"
    assert recorded["kwargs"] == {
        "revision": None,
        "version": 1,
        "backend": None,
        "user_agent": None,
        "trust_remote_code": True,
        "check_arch": True,
    }
    # The patch is active for the duration of the call only.
    assert recorded["patched"] is True
    assert kernels_hf_hub._get_hf_api is original


def test_has_kernel_wrapper_scopes_the_patch(kernels_modules, monkeypatch):
    from modelscope_hub.kernels import has_kernel

    kernels, kernels_hf_hub, _ = kernels_modules
    original = kernels_hf_hub._get_hf_api
    calls = {}

    def fake_has_kernel(repo_id, **kwargs):
        calls["patched"] = kernels_hf_hub._get_hf_api is _ms_api
        return True

    monkeypatch.setattr(kernels, "has_kernel", fake_has_kernel)

    assert has_kernel("kernels-community/activation", version=1) is True
    assert calls["patched"] is True
    assert kernels_hf_hub._get_hf_api is original


def test_scoped_call_inside_global_patch_keeps_global_patch(kernels_modules, monkeypatch):
    """Dual entry points: a scoped wrapper call inside a global patch must not undo it."""
    kernels, kernels_hf_hub, _ = kernels_modules
    original = kernels_hf_hub._get_hf_api
    recorded = {}

    def fake_get_kernel(repo_id, **kwargs):
        recorded["patched"] = kernels_hf_hub._get_hf_api is _ms_api
        return "kernel-module"

    monkeypatch.setattr(kernels, "get_kernel", fake_get_kernel)

    patch_kernels_hub()
    try:
        assert kernels_hf_hub._get_hf_api is _ms_api
        result = kernels_get_kernel("kernels-community/activation", version=1)
        assert result == "kernel-module"
        # The scoped call saw the (already active) patch...
        assert recorded["patched"] is True
        # ...and did not undo the global patch on exit.
        assert kernels_hf_hub._get_hf_api is _ms_api
    finally:
        unpatch_kernels_hub()

    assert kernels_hf_hub._get_hf_api is original


# ---------------------------------------------------------------------------
# New scoped wrappers: install / lock entries
# ---------------------------------------------------------------------------
def test_install_kernel_wrapper_scopes_the_patch(kernels_modules, monkeypatch):
    from modelscope_hub.kernels import install_kernel

    kernels, kernels_hf_hub, _ = kernels_modules
    original = kernels_hf_hub._get_hf_api
    calls = {}

    def fake_install(repo_id, **kwargs):
        calls["patched"] = kernels_hf_hub._get_hf_api is _ms_api
        calls["repo_id"] = repo_id
        calls["kwargs"] = kwargs
        return Path("/cache/variants/x")

    monkeypatch.setattr(kernels, "install_kernel", fake_install)

    result = install_kernel(
        "kernels-community/activation",
        version=1,
        local_files_only=False,
        trust_remote_code=True,
    )
    assert result == Path("/cache/variants/x")
    assert calls["patched"] is True
    assert calls["repo_id"] == "kernels-community/activation"
    assert calls["kwargs"]["version"] == 1
    assert calls["kwargs"]["trust_remote_code"] is True
    assert kernels_hf_hub._get_hf_api is original


def test_install_kernel_all_variants_wrapper_scopes_the_patch(kernels_modules, monkeypatch):
    from modelscope_hub.kernels import install_kernel_all_variants

    kernels, kernels_hf_hub, _ = kernels_modules
    original = kernels_hf_hub._get_hf_api
    calls = {}

    def fake_install_all(repo_id, **kwargs):
        calls["patched"] = kernels_hf_hub._get_hf_api is _ms_api
        return Path("/cache/build")

    # `install_kernel_all_variants` lives in the `kernels.install` submodule
    # and is not exported on the `kernels` top level.
    monkeypatch.setattr(kernels.install, "install_kernel_all_variants", fake_install_all)

    result = install_kernel_all_variants("kernels-community/activation", version=1)
    assert result == Path("/cache/build")
    assert calls["patched"] is True
    assert kernels_hf_hub._get_hf_api is original


def test_get_locked_kernel_wrapper_scopes_the_patch(kernels_modules, monkeypatch, tmp_path):
    from modelscope_hub.kernels import get_locked_kernel

    kernels, kernels_hf_hub, _ = kernels_modules
    original = kernels_hf_hub._get_hf_api
    calls = {}

    def fake_get_locked(repo_id, **kwargs):
        calls["patched"] = kernels_hf_hub._get_hf_api is _ms_api
        calls["kwargs"] = kwargs
        return "kernel-module"

    monkeypatch.setattr(kernels, "get_locked_kernel", fake_get_locked)

    lockfile = tmp_path / "kernel-lock.toml"
    result = get_locked_kernel("kernels-community/activation", lockfile=lockfile)
    assert result == "kernel-module"
    assert calls["patched"] is True
    assert calls["kwargs"]["lockfile"] == lockfile
    assert kernels_hf_hub._get_hf_api is original


def test_load_kernel_wrapper_scopes_the_patch(kernels_modules, monkeypatch, tmp_path):
    from modelscope_hub.kernels import load_kernel

    kernels, kernels_hf_hub, _ = kernels_modules
    original = kernels_hf_hub._get_hf_api
    calls = {}

    def fake_load(repo_id, **kwargs):
        calls["patched"] = kernels_hf_hub._get_hf_api is _ms_api
        return "kernel-module"

    monkeypatch.setattr(kernels, "load_kernel", fake_load)

    lockfile = tmp_path / "kernel-lock.toml"
    result = load_kernel("kernels-community/activation", lockfile=lockfile, backend="cpu")
    assert result == "kernel-module"
    assert calls["patched"] is True
    assert kernels_hf_hub._get_hf_api is original


# ---------------------------------------------------------------------------
# Lockfile commit-hash hint
# ---------------------------------------------------------------------------
def test_hf_hub_download_commit_hash_failure_gets_lockfile_hint():
    hf_errors = pytest.importorskip("huggingface_hub.errors")
    fake = mock.MagicMock()
    fake.download_file.side_effect = NotExistError("no such revision")
    api = MsKernelApi(hub_api=fake)

    commit = "a" * 40
    with pytest.raises(hf_errors.EntryNotFoundError) as excinfo:
        api.hf_hub_download("kernels-community/activation", filename="build/x/metadata.json", revision=commit)

    cause = excinfo.value.__cause__
    assert cause is not None
    # PEP 678 notes exist on Python 3.11+; the hint text is in the notes.
    notes = list(getattr(cause, "__notes__", ()))
    if notes:
        assert any("lockfile" in note for note in notes)
        assert commit in "".join(notes)


def test_snapshot_download_commit_hash_failure_propagates_with_hint():
    fake = mock.MagicMock()
    fake.download_repo.side_effect = NotExistError("no such revision")
    api = MsKernelApi(hub_api=fake)

    commit = "b" * 40
    with pytest.raises(NotExistError) as excinfo:
        api.snapshot_download("kernels-community/activation", repo_type="kernel", revision=commit)

    notes = list(getattr(excinfo.value, "__notes__", ()))
    if notes:
        assert any("lockfile" in note for note in notes)


def test_hf_hub_download_branch_failure_gets_no_lockfile_hint():
    hf_errors = pytest.importorskip("huggingface_hub.errors")
    fake = mock.MagicMock()
    fake.download_file.side_effect = NotExistError("no such file")
    api = MsKernelApi(hub_api=fake)

    with pytest.raises(hf_errors.EntryNotFoundError) as excinfo:
        api.hf_hub_download("kernels-community/activation", filename="missing.json", revision="v1")

    assert not list(getattr(excinfo.value.__cause__, "__notes__", ()))


# ---------------------------------------------------------------------------
# Synthetic OIDs (kernels 0.18 validates target_commit as a git object id)
# ---------------------------------------------------------------------------
def test_list_repo_refs_returns_hex_target_commits():
    import re as _re

    fake = mock.MagicMock()
    fake.legacy.list_revisions_detail.return_value = ([{"Revision": "master"}], [])
    api = MsKernelApi(hub_api=fake)

    refs = api.list_repo_refs("kernels-community/activation")
    # kernels 0.18 requires Oid.from_str(target_commit): 40/64 hex digits.
    for ref in refs.branches + refs.tags:
        assert _re.fullmatch(r"[0-9a-f]{40}", ref.target_commit), ref
    # Master-only mirror: the synthesized v1 ref also carries a hex OID.
    v1 = next(r for r in refs.branches if r.name == "v1")
    assert _re.fullmatch(r"[0-9a-f]{40}", v1.target_commit)
    # Deterministic: same repo/ref always maps to the same OID.
    fake.legacy.list_revisions_detail.return_value = ([{"Revision": "master"}], [])
    again = api.list_repo_refs("kernels-community/activation")
    assert next(r for r in again.branches if r.name == "v1").target_commit == v1.target_commit


def test_synthetic_oid_resolved_back_to_branch_on_downloads():
    fake = mock.MagicMock()
    fake.legacy.list_revisions_detail.return_value = ([{"Revision": "master"}], [])
    api = MsKernelApi(hub_api=fake)

    refs = api.list_repo_refs("kernels-community/activation")
    oid = refs.branches[0].target_commit  # master's synthetic OID (0.17-style resolve)
    v1_oid = next(r.target_commit for r in refs.branches if r.name == "v1")

    # kernels passes the OID as the download revision (both 0.17 and 0.18);
    # the shim maps it back to the real ModelScope ref name.
    api.snapshot_download("kernels-community/activation", repo_type="kernel", revision=oid)
    assert fake.download_repo.call_args.kwargs["revision"] == "master"

    api.hf_hub_download("kernels-community/activation", filename="README.md", revision=v1_oid)
    assert fake.download_file.call_args.kwargs["revision"] == "master"

    api.list_repo_tree("kernels-community/activation", revision=oid)
    assert fake.list_repo_files.call_args.kwargs["revision"] == "master"

    api.file_exists("kernels-community/activation", "README.md", revision=oid)
    assert fake.list_repo_files.call_args.kwargs["revision"] == "master"


def test_plain_revision_not_mapped():
    """A normal branch/tag revision passes through untouched."""
    fake = mock.MagicMock()
    api = MsKernelApi(hub_api=fake)

    api.snapshot_download("kernels-community/activation", repo_type="kernel", revision="v2")
    assert fake.download_repo.call_args.kwargs["revision"] == "v2"

    api.snapshot_download("kernels-community/activation", repo_type="kernel", revision=None)
    assert fake.download_repo.call_args.kwargs["revision"] == "master"
