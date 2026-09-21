# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for the agent plugin loader (``modelscope_hub.agent._plugin``).

Happy paths build a real plugin package on disk -- manifest, entry module and all
-- so verification, import and operation negotiation are exercised rather than
mocked. Only the network hop (``snapshot_download``) is stubbed, per the rule
that CI runs with ``MODELSCOPE_RUN_REMOTE_TESTS=false``.

The gates themselves are tested here at function level; that they are *wired
into* ``install_agent`` and map to the right exit codes is covered by
``tests/cli/test_agent_install.py``, so it is not repeated at both levels.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from modelscope_hub import constants
from modelscope_hub.agent import _plugin
from modelscope_hub.errors import InvalidParameter, NotSupportedError

TRUSTED = "mushenL"
PLUGIN_REPO = f"{TRUSTED}/agent-hub-plugin"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


ENTRY_SOURCE = textwrap.dedent(
    """
    from dataclasses import dataclass, field


    @dataclass(frozen=True)
    class Result:
        ok: bool = True
        error: str | None = None
        files_written: tuple = field(default_factory=tuple)
        root: str = "/tmp/ws"
        exit_code: int = 0


    CALLS = []


    def capabilities():
        return {"operations": ("install", "download"), "version": "9.9.9"}


    def install(repo, **kwargs):
        CALLS.append(("install", repo, kwargs))
        return Result(files_written=("SOUL.md",))


    def download(repo, **kwargs):
        CALLS.append(("download", repo, kwargs))
        return Result(files_written=("SOUL.md",))
    """
).lstrip()


def make_plugin(
    root: Path,
    *,
    dirname: str = "plugin",
    entry_module: str = "fake_plugin",
    version: str = "9.9.9",
    operations: tuple[str, ...] = ("install", "download"),
    entry_source: str | None = None,
    extra_files: dict[str, str] | None = None,
    omit_hashes: bool = False,
    omit_entry: bool = False,
) -> Path:
    """Write a plugin package tree and return its directory."""
    directory = root / dirname
    directory.mkdir(parents=True, exist_ok=True)

    source = entry_source if entry_source is not None else ENTRY_SOURCE
    (directory / f"{entry_module}.py").write_text(source, encoding="utf-8")
    for rel, content in (extra_files or {}).items():
        target = directory / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    manifest: dict[str, Any] = {
        "name": "agent-hub-plugin",
        "version": version,
        "kind": "agent-hub-plugin",
        "frameworks": ["qwenpaw", "ms-agent"],
        "api": list(operations),
    }
    if not omit_entry:
        manifest["entry_module"] = entry_module
    if not omit_hashes:
        files = sorted(p for p in directory.rglob("*") if p.is_file())
        manifest["content_sha256"] = {p.relative_to(directory).as_posix(): _sha256(p) for p in files}
    (directory / "plugin.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory


def spec_for(directory: Path, *, manifest: dict | None = None) -> _plugin.PluginSpec:
    resolved = manifest
    if resolved is None:
        resolved = json.loads((directory / "plugin.json").read_text(encoding="utf-8"))
    return _plugin.PluginSpec(
        repo_id=PLUGIN_REPO,
        owner=TRUSTED,
        name="agent-hub-plugin",
        revision="master",
        directory=directory,
        manifest=resolved,
        entry_module=str(resolved.get("entry_module", "fake_plugin")),
    )


def load_entry(directory: Path, module_name: str):
    """Import a generated plugin entry module the way ``load_plugin`` does."""
    return _plugin.load_plugin(spec_for(directory, manifest={"entry_module": module_name}))


def loaded(directory: Path, module_name: str):
    """The module ``install_agent`` loaded from *directory*.

    Reached through the alias, not ``import <module_name>``: registration is
    directory-scoped on purpose, so the plain name is never in ``sys.modules`` and
    the plugin directory is off ``sys.path`` again once the call returns.
    """
    alias = _plugin._module_alias(spec_for(directory, manifest={"entry_module": module_name}))
    return sys.modules[alias]


# ---------------------------------------------------------------------------
# resolve_plugin_repo
# ---------------------------------------------------------------------------
def test_resolve_plugin_repo_resolution_order(monkeypatch):
    """Argument beats environment beats the built-in default."""
    monkeypatch.setenv(constants.ENV_AGENT_PLUGIN_REPO, "env-owner/env-plugin")
    assert _plugin.resolve_plugin_repo("arg-owner/arg-plugin") == "arg-owner/arg-plugin"
    assert _plugin.resolve_plugin_repo(None) == "env-owner/env-plugin"
    assert _plugin.resolve_plugin_repo("  ") == "env-owner/env-plugin"

    monkeypatch.delenv(constants.ENV_AGENT_PLUGIN_REPO, raising=False)
    assert _plugin.resolve_plugin_repo(None) == constants.DEFAULT_AGENT_PLUGIN_REPO


@pytest.mark.parametrize("value", ["noslash", "/noname", "owner/"])
def test_resolve_plugin_repo_requires_owner_slash_name(monkeypatch, value):
    monkeypatch.setenv(constants.ENV_AGENT_PLUGIN_REPO, value)
    with pytest.raises(InvalidParameter):
        _plugin.resolve_plugin_repo(None)


# ---------------------------------------------------------------------------
# assert_trusted_owner
# ---------------------------------------------------------------------------
@pytest.fixture
def allow_list(monkeypatch):
    monkeypatch.setattr(constants, "AGENT_PLUGIN_TRUSTED_OWNERS", frozenset({"mushenL", "modelscope"}))


def test_assert_trusted_owner_accepts_allow_listed(allow_list):
    assert _plugin.assert_trusted_owner("mushenL/agent-hub-plugin") == ("mushenL", "agent-hub-plugin")
    assert _plugin.assert_trusted_owner("modelscope/agent-hub-plugin") == ("modelscope", "agent-hub-plugin")


@pytest.mark.parametrize("owner", ["evil", "mushenL-x", "modelscope2", "ai_modelscope"])
def test_assert_trusted_owner_rejects_others(allow_list, owner):
    with pytest.raises(InvalidParameter) as excinfo:
        _plugin.assert_trusted_owner(f"{owner}/agent-hub-plugin")
    assert owner in str(excinfo.value)
    assert "AGENT_PLUGIN_TRUSTED_OWNERS" in excinfo.value.suggestion


def test_the_allow_list_cannot_be_widened_from_the_environment(monkeypatch):
    """The allow-list is the trust anchor for a command that executes downloaded
    code, so a parent process must not be able to move it. The override that used
    to exist is gone; this pins that it stays gone rather than being reintroduced
    as a convenience."""
    monkeypatch.setenv("MODELSCOPE_AGENT_PLUGIN_TRUSTED_OWNERS", "evilcorp")
    importlib.reload(constants)
    try:
        assert "evilcorp" not in constants.AGENT_PLUGIN_TRUSTED_OWNERS
        assert not hasattr(constants, "ENV_AGENT_PLUGIN_TRUSTED_OWNERS")
        with pytest.raises(InvalidParameter):
            _plugin.assert_trusted_owner("evilcorp/agent-hub-plugin")
    finally:
        importlib.reload(constants)


def test_the_shipped_allow_list_has_no_personal_account():
    """A personal account in the default list is a supply-chain entry point: if it
    is compromised, anything it publishes passes the owner gate."""
    assert constants.AGENT_PLUGIN_TRUSTED_OWNERS == frozenset({"modelscope", "AI-ModelScope"})


@pytest.mark.parametrize("owner", ["mushenl", "MUSHENL", "ModelScope", "MODELSCOPE", "ai-modelscope"])
def test_assert_trusted_owner_matches_case_insensitively(monkeypatch, owner):
    """The registry resolves ids case-insensitively and normalises the owner --
    ``ModelScope/x`` and ``modelscope/x`` are one repository -- so two owners
    differing only in case cannot both exist. Matching exactly would not stop a
    look-alike; it would only reject the casing somebody copied from the website,
    which is how the product writes ``ModelScope``."""
    monkeypatch.setattr(constants, "AGENT_PLUGIN_TRUSTED_OWNERS", frozenset({"mushenL", "modelscope", "AI-ModelScope"}))
    got_owner, got_name = _plugin.assert_trusted_owner(f"{owner}/agent-hub-plugin")
    assert got_name == "agent-hub-plugin"
    # Echoed as typed, so messages and PluginSpec keep the user's spelling.
    assert got_owner == owner


def test_assert_trusted_owner_empty_list_blocks_everything(monkeypatch):
    monkeypatch.setattr(constants, "AGENT_PLUGIN_TRUSTED_OWNERS", frozenset())
    with pytest.raises(InvalidParameter):
        _plugin.assert_trusted_owner("mushenL/agent-hub-plugin")


# ---------------------------------------------------------------------------
# verify_manifest
# ---------------------------------------------------------------------------
def test_verify_manifest_accepts_a_consistent_package(tmp_path):
    directory = make_plugin(tmp_path)
    manifest = _plugin.verify_manifest(directory, PLUGIN_REPO)
    assert manifest["entry_module"] == "fake_plugin"
    assert manifest["version"] == "9.9.9"


@pytest.mark.parametrize(
    "case,expected",
    [
        ("absent", "not an agent plugin"),
        ("no-entry", "entry_module"),
        # Without a digest the import would be unconditional code execution.
        ("no-hashes", "content_sha256"),
    ],
)
def test_verify_manifest_rejects_an_incomplete_manifest(tmp_path, case, expected):
    if case == "absent":
        directory = tmp_path / "empty"
        directory.mkdir()
    elif case == "no-entry":
        directory = make_plugin(tmp_path, omit_entry=True)
    else:
        directory = make_plugin(tmp_path, omit_hashes=True)
    with pytest.raises(NotSupportedError) as excinfo:
        _plugin.verify_manifest(directory, PLUGIN_REPO)
    assert expected in str(excinfo.value)


@pytest.mark.parametrize(
    "case,expected",
    [
        ("tampered", "sha256 mismatch"),
        ("missing", "missing"),
        ("unlisted", "not listed in the manifest"),
    ],
)
def test_verify_manifest_rejects_inconsistent_content(tmp_path, case, expected):
    directory = make_plugin(tmp_path, extra_files={"pkg/mod.py": "x = 1\n"})
    if case == "tampered":
        target = directory / "fake_plugin.py"
        target.write_text(target.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")
    elif case == "missing":
        (directory / "pkg" / "mod.py").unlink()
    else:
        (directory / "surprise.py").write_text("import os\n", encoding="utf-8")
    with pytest.raises(NotSupportedError) as excinfo:
        _plugin.verify_manifest(directory, PLUGIN_REPO)
    assert expected in str(excinfo.value)


def test_verify_manifest_exempts_non_plugin_files(tmp_path):
    """``.gitattributes`` is injected by the hub into every repository and so is
    never in an author's manifest; refusing it made every real package
    unverifiable. ``plugin.json`` cannot hash itself and ``__pycache__`` is
    written locally by a previous import.

    The exemption is exact -- a genuinely unlisted file is still refused.
    """
    directory = make_plugin(tmp_path)
    (directory / ".gitattributes").write_text("*.bin filter=lfs\n", encoding="utf-8")
    cache = directory / "__pycache__"
    cache.mkdir()
    (cache / "fake_plugin.cpython-311.pyc").write_bytes(b"\x00\x01")

    manifest = _plugin.verify_manifest(directory, PLUGIN_REPO)
    listed = manifest["content_sha256"]
    assert "plugin.json" not in listed
    assert ".gitattributes" not in listed
    assert not any("__pycache__" in key for key in listed)

    (directory / "surprise.py").write_text("import os\n", encoding="utf-8")
    with pytest.raises(NotSupportedError) as excinfo:
        _plugin.verify_manifest(directory, PLUGIN_REPO)
    assert "surprise.py" in str(excinfo.value)
    assert ".gitattributes" not in str(excinfo.value)


@pytest.mark.parametrize("key", ["/etc/hosts", "../../etc/hosts", "pkg/../../outside.py"])
def test_verify_manifest_refuses_keys_pointing_outside_the_package(tmp_path, key):
    """Manifest keys are attacker-controlled and become paths. ``directory / key``
    with an absolute key discards the directory outright, so a hostile manifest
    could have the loader hash a file anywhere on disk. Nothing is returned to the
    caller, so it is not a disclosure -- but it is a read the manifest has no
    business requesting, and a package that asks for it is not a corrupt download.
    """
    directory = make_plugin(tmp_path)
    manifest = json.loads((directory / "plugin.json").read_text())
    manifest["content_sha256"][key] = "0" * 64
    (directory / "plugin.json").write_text(json.dumps(manifest))

    with pytest.raises(NotSupportedError) as excinfo:
        _plugin.verify_manifest(directory, PLUGIN_REPO)
    assert "outside the package" in str(excinfo.value)


# ---------------------------------------------------------------------------
# execution audit / no opt-in
# ---------------------------------------------------------------------------
def test_log_execution_records_the_build_before_import(tmp_path, caplog):
    """There is no opt-in to wait for any more -- the allow-list is the
    authorisation -- but executing downloaded code still deserves an audit line
    naming the exact build, emitted before the import so a crash during it leaves
    a trace of what was being loaded."""
    spec = spec_for(make_plugin(tmp_path))
    with caplog.at_level("INFO", logger="modelscope_hub.agent"):
        _plugin.log_execution(spec)
    for expected in (spec.repo_id, spec.revision, spec.entry_module, "9.9.9"):
        assert expected in caplog.text


def test_no_trust_opt_in_is_exposed():
    """Deferred on purpose: the allow-list already decided, so a per-invocation
    flag would imply a choice the user does not have. Pin that it stays gone until
    third-party plugins are supported."""
    import inspect

    assert not hasattr(constants, "AGENT_TRUST_REMOTE_CODE")
    assert not hasattr(constants, "ENV_AGENT_TRUST_REMOTE_CODE")
    assert not hasattr(_plugin, "require_trust")
    assert "trust_remote_code" not in inspect.signature(_plugin.install_agent).parameters

    # Nor in the user-facing docs: naming an opt-in that does not exist invites
    # users to reach for it, and a stale example is a call that raises TypeError.
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "trust_remote_code" not in readme
    assert "trust-remote-code" not in readme


# ---------------------------------------------------------------------------
# fetch_plugin
# ---------------------------------------------------------------------------
def test_fetch_plugin_requests_a_model_repo(monkeypatch, tmp_path):
    seen: list[dict[str, Any]] = []

    def fake_snapshot_download(repo_id, **kwargs):
        seen.append({"repo_id": repo_id, **kwargs})
        return str(tmp_path)

    import modelscope_hub.compat as compat

    monkeypatch.setattr(compat, "snapshot_download", fake_snapshot_download)

    got = _plugin.fetch_plugin(PLUGIN_REPO, revision="v1.2.3", token="tok", endpoint="https://ep")
    assert got == tmp_path
    assert seen[0]["repo_id"] == PLUGIN_REPO
    assert seen[0]["repo_type"] == "model"
    assert seen[0]["revision"] == "v1.2.3"
    assert (seen[0]["token"], seen[0]["endpoint"]) == ("tok", "https://ep")

    _plugin.fetch_plugin(PLUGIN_REPO)
    assert seen[1]["revision"] == constants.DEFAULT_AGENT_PLUGIN_REVISION


def test_fetch_plugin_wraps_download_failure(monkeypatch):
    """``snapshot_download`` re-raises hub errors as ``requests.HTTPError``, so
    the original type is not a reliable discriminator."""

    def boom(repo_id, **kwargs):
        raise RuntimeError("404 not found")

    import modelscope_hub.compat as compat

    monkeypatch.setattr(compat, "snapshot_download", boom)
    with pytest.raises(NotSupportedError) as excinfo:
        _plugin.fetch_plugin(PLUGIN_REPO)
    assert PLUGIN_REPO in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# load_plugin / select_operation
# ---------------------------------------------------------------------------
def test_load_plugin_imports_the_declared_entry_module(tmp_path):
    directory = make_plugin(tmp_path, entry_module="entry_a")
    module = _plugin.load_plugin(spec_for(directory))
    assert module.__file__ == str(directory / "entry_a.py")
    # Registered under a directory-scoped alias, not its own name -- see the next
    # test for why that matters.
    assert module.__name__.startswith("_ms_agent_plugin_entry_a_")
    assert str(directory) in sys.path
    sys.path.remove(str(directory))
    sys.modules.pop(module.__name__, None)


def test_load_plugin_does_not_serve_one_directory_another(tmp_path):
    """Two plugins sharing a repository id, revision and entry module name must
    not share code. ``import_module`` would have returned the first from
    ``sys.modules`` for the second, so installing plugin B ran plugin A."""
    first = make_plugin(tmp_path / "one", dirname="p", entry_module="same_name", entry_source="MARKER = 'first'\n")
    second = make_plugin(tmp_path / "two", dirname="p", entry_module="same_name", entry_source="MARKER = 'second'\n")
    loaded = []
    try:
        for directory in (first, second):
            sys.path.insert(0, str(directory))
            loaded.append(_plugin.load_plugin(spec_for(directory)))
        assert [m.MARKER for m in loaded] == ["first", "second"]
        assert loaded[0] is not loaded[1]
    finally:
        for m in loaded:
            sys.modules.pop(m.__name__, None)
        for directory in (first, second):
            if str(directory) in sys.path:
                sys.path.remove(str(directory))


def test_load_plugin_restores_sys_path_on_failure(tmp_path):
    directory = tmp_path / "plugin"
    directory.mkdir()
    spec = _plugin.PluginSpec(
        repo_id=PLUGIN_REPO,
        owner=TRUSTED,
        name="p",
        revision="master",
        directory=directory,
        manifest={},
        entry_module="does_not_exist_xyz",
    )
    before = list(sys.path)
    with pytest.raises(ImportError):
        _plugin.load_plugin(spec)
    assert sys.path == before


def test_plugin_syspath_is_scoped_to_the_block(tmp_path):
    """A plugin directory left at sys.path[0] lets any file it ships shadow the
    standard library or a dependency for the rest of the process -- and shipping
    one is not a rule violation, since every file has to be listed in the
    manifest. Harmless in a one-shot CLI; a long-lived process calling
    install_agent would stay poisoned."""
    directory = make_plugin(tmp_path)
    before = list(sys.path)
    with _plugin.plugin_syspath(directory):
        assert str(directory) in sys.path
        assert sys.path[0] == str(directory)
    assert sys.path == before


def test_install_agent_leaves_no_plugin_directory_on_sys_path(wired):
    _plugin.install_agent("owner/my-agent", plugin_repo=PLUGIN_REPO)
    assert str(wired) not in sys.path


def test_select_operation_prefers_install(tmp_path):
    module = load_entry(make_plugin(tmp_path, entry_module="sel_install"), "sel_install")
    name, func = _plugin.select_operation(module)
    assert (name, func) == ("install", module.install)


def test_select_operation_falls_back_to_download(tmp_path):
    source = textwrap.dedent(
        """
        def capabilities():
            return {"operations": ("download",)}

        def install(repo, **kwargs):
            raise AssertionError("must not be chosen")

        def download(repo, **kwargs):
            return "downloaded"
        """
    ).lstrip()
    directory = make_plugin(tmp_path, entry_module="sel_download", entry_source=source)
    module = load_entry(directory, "sel_download")
    name, func = _plugin.select_operation(module)
    assert (name, func) == ("download", module.download)


def test_select_operation_rejects_an_undeclared_name(tmp_path):
    """A plugin shipping a name without declaring it in ``capabilities()`` is not
    trusted to have implemented it."""
    source = textwrap.dedent(
        """
        def capabilities():
            return {"operations": ()}

        def install(repo, **kwargs):
            raise AssertionError("must not be chosen")
        """
    ).lstrip()
    directory = make_plugin(tmp_path, entry_module="sel_none", entry_source=source)
    module = load_entry(directory, "sel_none")
    with pytest.raises(NotSupportedError) as excinfo:
        _plugin.select_operation(module)
    assert "sel_none" in str(excinfo.value)


def test_select_operation_refuses_a_broken_capabilities(tmp_path):
    """A ``capabilities()`` that exists and fails means the plugin is broken, not
    that it declares nothing. Swallowing it downgraded selection to "first
    callable attribute wins", which can pick a placeholder the plugin deliberately
    left undeclared."""
    source = textwrap.dedent(
        """
        def capabilities():
            raise RuntimeError("manifest and code disagree")

        def install(repo, **kwargs):
            raise AssertionError("must not be chosen")
        """
    ).lstrip()
    directory = make_plugin(tmp_path, entry_module="broken_caps", entry_source=source)
    module = load_entry(directory, "broken_caps")
    with pytest.raises(NotSupportedError) as excinfo:
        _plugin.select_operation(module)
    assert "capabilities() raised" in str(excinfo.value)
    assert "manifest and code disagree" in str(excinfo.value)


def test_select_operation_without_capabilities_uses_presence(tmp_path):
    source = "def download(repo, **kwargs):\n    return 'ok'\n"
    directory = make_plugin(tmp_path, entry_module="sel_nocaps", entry_source=source)
    assert _plugin.select_operation(load_entry(directory, "sel_nocaps"))[0] == "download"


def test_select_operation_prefers_install_over_fetch_raw(tmp_path):
    """A plugin that owns placement wins over one that only transports bytes, so
    the install layer taking over needs no hub release."""
    source = textwrap.dedent(
        """
        def capabilities():
            return {"operations": ("install", "fetch_raw", "download")}

        def install(repo, **kwargs):
            return "installed"

        def fetch_raw(repo, *, dest, **kwargs):
            raise AssertionError("must not be chosen while install is declared")
        """
    ).lstrip()
    directory = make_plugin(tmp_path, entry_module="sel_pref", entry_source=source)
    module = load_entry(directory, "sel_pref")
    assert _plugin.select_operation(module) == ("install", module.install)


def test_select_operation_falls_back_to_fetch_raw(tmp_path):
    """A transport-only plugin is usable: ``download`` is present but undeclared,
    so it must not be chosen over the operation the plugin actually reports."""
    source = textwrap.dedent(
        """
        def capabilities():
            return {"operations": ("fetch_raw", "restore", "list_backups")}

        def install(repo, **kwargs):
            raise AssertionError("must not be chosen")

        def download(repo, **kwargs):
            raise AssertionError("must not be chosen")

        def fetch_raw(repo, *, dest, **kwargs):
            return "fetched"
        """
    ).lstrip()
    directory = make_plugin(tmp_path, entry_module="sel_fetch", entry_source=source)
    module = load_entry(directory, "sel_fetch")
    assert _plugin.select_operation(module) == ("fetch_raw", module.fetch_raw)


def test_accepted_kwargs_narrowing():
    def positional(repo, name=None):
        return repo, name

    assert _plugin._accepted_kwargs(positional, {"repo": "a/b", "name": "x", "force": True}) == {
        "repo": "a/b",
        "name": "x",
    }

    def variadic(**kwargs):
        return kwargs

    payload = {"repo": "a/b", "anything": 1}
    assert _plugin._accepted_kwargs(variadic, payload) == payload


# ---------------------------------------------------------------------------
# install_agent
# ---------------------------------------------------------------------------
@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Point ``install_agent`` at a real plugin tree with the network stubbed."""
    monkeypatch.setattr(constants, "AGENT_PLUGIN_TRUSTED_OWNERS", frozenset({TRUSTED}))
    monkeypatch.delenv(constants.ENV_AGENT_PLUGIN_REPO, raising=False)
    # install_agent resolves a default staging directory under the cache; keep it
    # out of the real user home.
    monkeypatch.setenv(constants.ENV_CACHE, str(tmp_path / "cache"))
    directory = make_plugin(tmp_path, entry_module="e2e_plugin")
    monkeypatch.setattr(_plugin, "fetch_plugin", lambda repo_id, **kwargs: directory)
    yield directory
    sys.modules.pop(_plugin._module_alias(spec_for(directory)), None)


def test_install_agent_happy_path_and_option_forwarding(wired):
    outcome = _plugin.install_agent("owner/my-agent", plugin_repo=PLUGIN_REPO)
    assert outcome.ok, outcome.error
    assert (outcome.operation, outcome.exit_code) == ("install", 0)
    assert outcome.plugin.repo_id == PLUGIN_REPO
    assert outcome.plugin.version == "9.9.9"

    entry = loaded(wired, "e2e_plugin")
    assert entry.CALLS[-1][:2] == ("install", "owner/my-agent")
    forwarded = entry.CALLS[-1][2]
    # Unset optionals are dropped so the plugin applies its own defaults, but a
    # False boolean is a decision the caller made and is forwarded.
    assert {key: forwarded[key] for key in ("dry_run", "yes", "force", "quiet")} == {
        "dry_run": False,
        "yes": False,
        "force": False,
        "quiet": False,
    }
    # A variadic entry also gets the resolved destination. With no --local-dir
    # that is a fresh staging directory under the cache, not a workspace.
    staged = Path(forwarded["dest"])
    assert staged.parent.name == "agent-staging"
    assert staged.name.startswith("owner--my-agent-")

    _plugin.install_agent(
        "owner/my-agent",
        name="sub",
        local_dir="/tmp/ws",
        dry_run=True,
        force=True,
        endpoint="https://pre.modelscope.cn",
        token="tok",
        plugin_repo=PLUGIN_REPO,
    )
    forwarded = entry.CALLS[-1][2]
    assert forwarded["name"] == "sub"
    assert forwarded["local_dir"] == "/tmp/ws"
    assert forwarded["dest"] == "/tmp/ws"
    assert forwarded["dry_run"] is True
    assert forwarded["force"] is True
    assert forwarded["endpoint"] == "https://pre.modelscope.cn"
    assert forwarded["token"] == "tok"


@pytest.mark.parametrize("repo", ["", "   ", "no-slash", "/noname", "owner/"])
def test_install_agent_validates_the_agent_repo(wired, monkeypatch, repo):
    """``/noname`` and ``owner/`` contain a slash but name no repository; they
    must be rejected before any network call."""

    def no_network(*args, **kwargs):
        raise AssertionError(f"network reached for malformed repo id {repo!r}")

    monkeypatch.setattr(_plugin, "fetch_plugin", no_network)
    import modelscope_hub.compat as compat

    monkeypatch.setattr(compat, "snapshot_download", no_network)

    with pytest.raises(InvalidParameter):
        _plugin.install_agent(repo, plugin_repo=PLUGIN_REPO)


def test_install_agent_reports_plugin_failure(wired, monkeypatch):
    source = textwrap.dedent(
        """
        from dataclasses import dataclass


        @dataclass(frozen=True)
        class Result:
            ok: bool = False
            error: str = "framework not installed"
            exit_code: int = 2


        def capabilities():
            return {"operations": ("install",)}


        def install(repo, **kwargs):
            return Result()
        """
    ).lstrip()
    directory = make_plugin(wired.parent, dirname="fail_plugin", entry_module="fail_plugin", entry_source=source)
    monkeypatch.setattr(_plugin, "fetch_plugin", lambda repo_id, **kwargs: directory)

    outcome = _plugin.install_agent("owner/my-agent", plugin_repo=PLUGIN_REPO)
    assert not outcome.ok
    assert outcome.error == "framework not installed"
    # The install layer's own codes (3/4/5/6) carry meaning and must survive.
    assert outcome.exit_code == 2
    sys.modules.pop("fail_plugin", None)


def test_install_agent_contains_a_plugin_exception(wired, monkeypatch):
    source = textwrap.dedent(
        """
        def capabilities():
            return {"operations": ("install",)}


        def install(repo, **kwargs):
            raise RuntimeError("boom")
        """
    ).lstrip()
    directory = make_plugin(wired.parent, dirname="boom_plugin", entry_module="boom_plugin", entry_source=source)
    monkeypatch.setattr(_plugin, "fetch_plugin", lambda repo_id, **kwargs: directory)

    outcome = _plugin.install_agent("owner/my-agent", plugin_repo=PLUGIN_REPO)
    assert not outcome.ok
    assert "boom" in outcome.error
    assert outcome.exit_code == 1
    sys.modules.pop("boom_plugin", None)


def test_an_unimportable_plugin_says_it_was_downloaded_but_not_run(wired, monkeypatch):
    """The download and the manifest check both passed, so a bare "failed to load"
    reads like a network problem and hides the two facts that matter: the package
    is fine, and no agent work happened."""
    source = "import definitely_not_installed_xyz\n"
    directory = make_plugin(wired.parent, dirname="bad_import", entry_module="bad_import", entry_source=source)
    monkeypatch.setattr(_plugin, "fetch_plugin", lambda repo_id, **kwargs: directory)

    outcome = _plugin.install_agent("owner/my-agent", plugin_repo=PLUGIN_REPO)
    assert not outcome.ok
    assert outcome.exit_code == 1
    assert "downloaded and verified" in outcome.error
    assert "NOT executed" in outcome.error
    assert "no agent was fetched or installed" in outcome.error


def test_a_plugin_with_no_usable_operation_says_it_was_not_run(wired, monkeypatch):
    source = textwrap.dedent(
        """
        def capabilities():
            return {"operations": ()}

        def install(repo, **kwargs):
            raise AssertionError("must not be chosen")
        """
    ).lstrip()
    directory = make_plugin(wired.parent, dirname="no_op", entry_module="no_op", entry_source=source)
    monkeypatch.setattr(_plugin, "fetch_plugin", lambda repo_id, **kwargs: directory)

    with pytest.raises(NotSupportedError) as excinfo:
        _plugin.install_agent("owner/my-agent", plugin_repo=PLUGIN_REPO)
    message = str(excinfo.value)
    assert "downloaded and verified" in message
    assert "NOT executed" in message
    assert "no agent was fetched or installed" in message
    sys.modules.pop("no_op", None)


@pytest.mark.parametrize("returned", ["None", "'done'", "{}"])
def test_install_agent_refuses_a_result_without_ok(wired, monkeypatch, returned):
    """``ok`` is the only signal that decides whether the user is told the agent
    was installed, so a result without it must fail closed. Defaulting it to True
    meant a plugin returning None, a bare string or an empty dict reported
    "Installed" for an install nobody could verify."""
    source = textwrap.dedent(
        f"""
        def capabilities():
            return {{"operations": ("install",)}}

        def install(repo, **kwargs):
            return {returned}
        """
    ).lstrip()
    directory = make_plugin(wired.parent, dirname="no_ok_plugin", entry_module="no_ok_plugin", entry_source=source)
    monkeypatch.setattr(_plugin, "fetch_plugin", lambda repo_id, **kwargs: directory)

    outcome = _plugin.install_agent("owner/my-agent", plugin_repo=PLUGIN_REPO)
    assert not outcome.ok
    assert outcome.exit_code == 1
    assert "no 'ok' attribute" in outcome.error
    sys.modules.pop("no_ok_plugin", None)


# ---------------------------------------------------------------------------
# fetch-only plugins
# ---------------------------------------------------------------------------
def test_default_staging_dir_computes_without_creating(tmp_path, monkeypatch):
    monkeypatch.setenv(constants.ENV_CACHE, str(tmp_path / "cache"))
    path = _plugin.default_staging_dir("owner/my-agent")

    assert path.parent == tmp_path / "cache" / "agent" / "agent-staging"
    # The repository id contains a slash, so it has to be flattened, and the
    # stamp keeps repeats of one repository apart. Resolution is one second, so
    # this separates runs, not concurrent calls.
    assert re.fullmatch(r"owner--my-agent-\d{8}_\d{6}", path.name)
    assert not path.exists(), "computing a destination must not create it"


#: Mirrors the real 0.2.0 plugin: ``dest`` is keyword-only and required, and
#: there is no ``local_dir`` at all, so a hub that does not resolve a
#: destination cannot call it.
FETCH_ONLY_SOURCE = textwrap.dedent(
    """
    from dataclasses import dataclass, field


    @dataclass(frozen=True)
    class Result:
        ok: bool = True
        error: str | None = None
        files_written: tuple = field(default_factory=tuple)
        root: str = ""
        exit_code: int = 0


    CALLS = []


    def capabilities():
        return {"operations": ("fetch_raw", "restore", "list_backups")}


    def install(**kwargs):
        raise AssertionError("placeholder must not be chosen")


    def download(repo, **kwargs):
        raise AssertionError("legacy alias must not be chosen")


    def fetch_raw(repo, *, dest, name=None, framework=None, dry_run=False,
                  quiet=False, endpoint=None, token=None):
        CALLS.append({"repo": repo, "dest": dest, "framework": framework})
        return Result(files_written=("SOUL.md", "AGENTS.md"), root=dest)
    """
).lstrip()


@pytest.fixture
def fetch_only(wired, monkeypatch):
    directory = make_plugin(
        wired.parent,
        dirname="fetch_plugin",
        entry_module="fetch_plugin",
        entry_source=FETCH_ONLY_SOURCE,
    )
    monkeypatch.setattr(_plugin, "fetch_plugin", lambda repo_id, **kwargs: directory)
    yield directory
    sys.modules.pop(_plugin._module_alias(spec_for(directory)), None)


def test_install_agent_drives_a_fetch_only_plugin(fetch_only):
    """The regression test for the joint-testing failure: a transport-only plugin
    used to be rejected outright, and then called without its required ``dest``."""
    outcome = _plugin.install_agent(
        "owner/my-agent",
        framework="qwenpaw",
        plugin_repo=PLUGIN_REPO,
    )
    assert outcome.ok, outcome.error
    assert outcome.operation == "fetch_raw"

    call = loaded(fetch_only, "fetch_plugin").CALLS[-1]
    assert call["repo"] == "owner/my-agent"
    assert call["framework"] == "qwenpaw"
    assert Path(call["dest"]).parent.name == "agent-staging"
    # The destination the plugin reports is the one the hub resolved.
    assert str(outcome.result.root) == call["dest"]


def test_install_agent_maps_local_dir_onto_dest(fetch_only):
    outcome = _plugin.install_agent(
        "owner/my-agent",
        local_dir="/tmp/joint/staging",
        plugin_repo=PLUGIN_REPO,
    )
    assert outcome.ok, outcome.error

    assert loaded(fetch_only, "fetch_plugin").CALLS[-1]["dest"] == "/tmp/joint/staging"


def test_dest_is_not_forwarded_to_an_operation_that_does_not_accept_it(wired, monkeypatch):
    """Narrowing keeps the legacy path working: an operation with no ``dest``
    parameter must not be handed one, or every 0.1.x plugin would break."""
    source = textwrap.dedent(
        """
        def capabilities():
            return {"operations": ("download",)}

        CALLS = []

        def download(repo, *, local_dir=None, dry_run=False):
            CALLS.append({"repo": repo, "local_dir": local_dir})
            return type("R", (), {"ok": True})()
        """
    ).lstrip()
    directory = make_plugin(wired.parent, dirname="legacy_plugin", entry_module="legacy_plugin", entry_source=source)
    monkeypatch.setattr(_plugin, "fetch_plugin", lambda repo_id, **kwargs: directory)

    outcome = _plugin.install_agent(
        "owner/my-agent",
        local_dir="/tmp/ws",
        plugin_repo=PLUGIN_REPO,
    )
    assert outcome.ok, outcome.error
    assert outcome.operation == "download"

    assert loaded(directory, "legacy_plugin").CALLS[-1] == {
        "repo": "owner/my-agent",
        "local_dir": "/tmp/ws",
    }
