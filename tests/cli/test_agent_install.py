# Copyright (c) Alibaba, Inc. and its affiliates.
"""CLI tests for ``ms agent install``.

These cover what the CLI layer owns: argument wiring, gate failures mapping to
exit 2, the plugin's exit code passing through unchanged, and output routing
(including ``-q``). The gates' own logic is tested in ``tests/test_agent_plugin.py``
and is not repeated here.

Mock-only: CI runs with ``MODELSCOPE_RUN_REMOTE_TESTS=false``. The trust gate
fires after the package is on disk, so it stubs ``fetch_plugin`` and points at a
real plugin tree -- manifest verification and the refusal message are genuine.
"""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from modelscope_hub import constants
from modelscope_hub.agent import InstallOutcome, PluginSpec, _plugin
from modelscope_hub.cli.agent import AgentCommand
from modelscope_hub.errors import NotSupportedError

from .conftest import run_cli

TRUSTED = "mushenL"
PLUGIN_REPO = f"{TRUSTED}/agent-hub-plugin"
AGENT_REPO = "owner/my-agent"

ENTRY = textwrap.dedent(
    """
    def capabilities():
        return {"operations": ("install",)}

    def install(repo, **kwargs):
        return type("R", (), {"ok": True, "error": None, "exit_code": 0,
                              "files_written": ("SOUL.md",), "root": "/tmp/ws"})()
    """
).lstrip()

ALL_OPTIONS = [
    "agent",
    "install",
    "-r",
    AGENT_REPO,
    "-n",
    "sub",
    "--framework",
    "qwenpaw",
    "--local-dir",
    "/tmp/ws",
    "--plugin-repo",
    PLUGIN_REPO,
    "--plugin-revision",
    "v1.0.0",
    "--trust-remote-code",
    "--dry-run",
    "-y",
    "--force",
    "-q",
]

MINIMAL = ["agent", "install", "-r", AGENT_REPO, "--plugin-repo", PLUGIN_REPO, "--trust-remote-code"]


def build_plugin(root: Path, *, entry_module: str = "cli_fake_plugin") -> Path:
    directory = root / "plugin"
    directory.mkdir(parents=True, exist_ok=True)
    entry = directory / f"{entry_module}.py"
    entry.write_text(ENTRY, encoding="utf-8")
    manifest = {
        "name": "agent-hub-plugin",
        "version": "9.9.9",
        "entry_module": entry_module,
        "frameworks": ["qwenpaw", "ms-agent"],
        "api": ["install"],
        "content_sha256": {f"{entry_module}.py": hashlib.sha256(entry.read_bytes()).hexdigest()},
    }
    (directory / "plugin.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return directory


def outcome(**kwargs) -> InstallOutcome:
    return InstallOutcome(**kwargs)


def spec(revision: str = "v1.0.0") -> PluginSpec:
    return PluginSpec(
        repo_id=PLUGIN_REPO,
        owner=TRUSTED,
        name="agent-hub-plugin",
        revision=revision,
        directory=Path("/tmp/nowhere"),
        manifest={"version": "9.9.9"},
        entry_module="mod",
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.setattr(constants, "AGENT_PLUGIN_TRUSTED_OWNERS", frozenset({TRUSTED, "modelscope"}))
    monkeypatch.setattr(constants, "AGENT_TRUST_REMOTE_CODE", False)
    monkeypatch.delenv(constants.ENV_AGENT_PLUGIN_REPO, raising=False)
    monkeypatch.delenv(constants.ENV_AGENT_TRUST_REMOTE_CODE, raising=False)
    yield
    sys.modules.pop("cli_fake_plugin", None)


@pytest.fixture
def stub_sdk(monkeypatch):
    """Replace the SDK entry point and capture what the CLI forwarded."""
    seen: dict[str, Any] = {}

    def fake_install(repo, **kwargs):
        seen.update(repo=repo, **kwargs)
        return InstallOutcome(ok=True, operation="install")

    monkeypatch.setattr("modelscope_hub.cli.agent.install_agent", fake_install)
    return seen


# ---------------------------------------------------------------------------
# argument wiring
# ---------------------------------------------------------------------------
def test_parser_wires_install(parser):
    args = parser.parse_args(ALL_OPTIONS)
    assert args._command is AgentCommand
    assert args.agent_command == "install"
    assert (args.repo, args.name, args.framework, args.local_dir) == (
        AGENT_REPO,
        "sub",
        "qwenpaw",
        "/tmp/ws",
    )
    assert (args.plugin_repo, args.plugin_revision) == (PLUGIN_REPO, "v1.0.0")
    assert args.trust_remote_code is True
    assert args.dry_run and args.yes and args.force and args.quiet


def test_parser_install_defaults(parser):
    args = parser.parse_args(["agent", "install", "-r", AGENT_REPO])
    assert args.trust_remote_code is False
    assert args.plugin_repo is None
    assert args.plugin_revision is None
    assert args.dry_run is False
    assert args.yes is False
    assert args.force is False
    assert args.quiet is False
    assert args.name is None
    assert args.framework is None
    assert args.local_dir is None


def test_forwards_every_option_to_the_sdk(stub_sdk):
    code, _, err = run_cli(ALL_OPTIONS)
    assert code == 0, err
    assert stub_sdk["repo"] == AGENT_REPO
    assert stub_sdk["name"] == "sub"
    assert stub_sdk["framework"] == "qwenpaw"
    assert stub_sdk["local_dir"] == "/tmp/ws"
    assert stub_sdk["plugin_repo"] == PLUGIN_REPO
    assert stub_sdk["plugin_revision"] == "v1.0.0"
    assert stub_sdk["trust_remote_code"] is True
    assert stub_sdk["dry_run"] and stub_sdk["yes"]
    assert stub_sdk["force"] and stub_sdk["quiet"]


def test_credentials_reach_the_sdk(stub_sdk):
    code, _, err = run_cli(MINIMAL, token="tok-123", endpoint="https://pre.modelscope.cn")
    assert code == 0, err
    assert stub_sdk["token"] == "tok-123"
    assert stub_sdk["endpoint"] == "https://pre.modelscope.cn"


def test_install_does_not_resolve_a_username(monkeypatch, stub_sdk):
    """``install`` always receives ``owner/name``, so it must not pay for a whoami
    round trip -- nor fail when that endpoint is unavailable."""
    from modelscope_hub import _openapi

    client = MagicMock(side_effect=AssertionError("whoami must not be called"))
    monkeypatch.setattr(_openapi, "OpenAPIClient", client)

    code, _, err = run_cli(MINIMAL, token="tok-123")
    assert code == 0, err
    client.assert_not_called()


# ---------------------------------------------------------------------------
# gates: exit code 2, and where the message lands
# ---------------------------------------------------------------------------
def test_unfetchable_default_plugin_repo_points_at_the_override(monkeypatch):
    """The default repository is not the user's choice, so a bare download error
    would leave them nothing to act on. Stubbed: this must not reach the network."""

    def refused(repo_id, **kwargs):
        assert repo_id == constants.DEFAULT_AGENT_PLUGIN_REPO, "the built-in default should be used"
        raise NotSupportedError(f"failed to download agent plugin {repo_id}@master: record not found")

    monkeypatch.setattr(_plugin, "fetch_plugin", refused)
    code, out, err = run_cli(["agent", "install", "-r", AGENT_REPO, "--trust-remote-code"])
    assert code != 0
    combined = out + err
    assert constants.DEFAULT_AGENT_PLUGIN_REPO in combined
    assert "--plugin-repo" in combined
    assert constants.ENV_AGENT_PLUGIN_TRUSTED_OWNERS in combined


def test_untrusted_plugin_owner_exits_2():
    code, out, err = run_cli(
        ["agent", "install", "-r", AGENT_REPO, "--plugin-repo", "evil/plugin", "--trust-remote-code"]
    )
    assert code == 2
    assert "evil" in err
    assert constants.ENV_AGENT_PLUGIN_TRUSTED_OWNERS in out + err


def test_trust_gate_refuses_and_explains(monkeypatch, tmp_path):
    """Without the opt-in the command stops before importing, and says what it
    would have run."""
    directory = build_plugin(tmp_path)
    monkeypatch.setattr(_plugin, "fetch_plugin", lambda repo_id, **kwargs: directory)

    code, out, err = run_cli(["agent", "install", "-r", AGENT_REPO, "--plugin-repo", PLUGIN_REPO])
    assert code == 2
    combined = out + err
    for expected in ("--trust-remote-code", PLUGIN_REPO, "9.9.9", "cli_fake_plugin"):
        assert expected in combined


def test_trust_gate_can_be_satisfied_by_env(monkeypatch, tmp_path):
    directory = build_plugin(tmp_path)
    monkeypatch.setattr(_plugin, "fetch_plugin", lambda repo_id, **kwargs: directory)
    monkeypatch.setattr(constants, "AGENT_TRUST_REMOTE_CODE", True)

    code, out, err = run_cli(["agent", "install", "-r", AGENT_REPO, "--plugin-repo", PLUGIN_REPO])
    assert code == 0, err
    assert "Installed" in out


# ---------------------------------------------------------------------------
# reporting and exit-code mapping
# ---------------------------------------------------------------------------
def test_reports_plugin_and_entry(monkeypatch):
    monkeypatch.setattr(
        "modelscope_hub.cli.agent.install_agent",
        lambda *a, **k: outcome(ok=True, operation="install", plugin=spec()),
    )
    code, out, _ = run_cli(MINIMAL)
    assert code == 0
    assert f"{PLUGIN_REPO}@v1.0.0" in out
    assert "9.9.9" in out
    assert "mod.install()" in out


@pytest.mark.parametrize(
    ("operation", "verb", "where"),
    [
        ("install", "Installed", "under"),
        ("download", "Installed", "under"),
        ("fetch_raw", "Fetched", "to"),
    ],
)
def test_success_wording_follows_the_negotiated_operation(monkeypatch, operation, verb, where):
    """A transport-only plugin stages files; reporting "Installed" would hide
    that no framework was touched."""
    result = type("R", (), {"files_written": ("SOUL.md", "AGENTS.md"), "root": "/tmp/staged"})()
    monkeypatch.setattr(
        "modelscope_hub.cli.agent.install_agent",
        lambda *a, **k: outcome(ok=True, operation=operation, plugin=spec(), result=result),
    )
    code, out, _ = run_cli(MINIMAL)
    assert code == 0
    assert f"{verb} {AGENT_REPO}: 2 file(s) {where} /tmp/staged" in out


def test_reports_the_supported_scope(monkeypatch):
    """A successful run must show what that plugin build covers, not just an exit
    code -- which frameworks, which operations work, and which are declared but
    not implemented yet."""
    rich = PluginSpec(
        repo_id=PLUGIN_REPO,
        owner=TRUSTED,
        name="agent-hub-plugin",
        revision="v0.2.0",
        directory=Path("/tmp/nowhere"),
        manifest={
            "version": "0.2.0",
            "frameworks": ["ms-agent", "qwenpaw"],
            "api": ["fetch_raw", "list_backups", "restore"],
            "roadmap": {"install": "entry package", "upload": "P1", "convert": "P2"},
            "content_sha256": {"agent_hub_core/__init__.py": "0" * 64},
        },
        entry_module="agent_hub_core",
    )
    result = type("R", (), {"files_written": ("SOUL.md",), "root": "/tmp/staged"})()
    monkeypatch.setattr(
        "modelscope_hub.cli.agent.install_agent",
        lambda *a, **k: outcome(ok=True, operation="fetch_raw", plugin=rich, result=result),
    )
    code, out, _ = run_cli(MINIMAL)
    assert code == 0
    assert "scope : frameworks ms-agent, qwenpaw" in out
    assert "operations fetch_raw, list_backups, restore" in out
    assert "planned convert (P2), install (entry package), upload (P1)" in out


def test_quiet_suppresses_all_hub_output(monkeypatch):
    monkeypatch.setattr(
        "modelscope_hub.cli.agent.install_agent",
        lambda *a, **k: outcome(ok=True, operation="install", plugin=spec()),
    )
    code, out, _ = run_cli([*MINIMAL, "-q"])
    assert code == 0
    assert out == ""


@pytest.mark.parametrize("plugin_code", [1, 3, 6])
def test_plugin_exit_code_is_passed_through(monkeypatch, plugin_code):
    """The install layer gives 3/4/5/6 distinct meanings (already exists, refused
    to overwrite, install or self-check failed, framework mismatch); collapsing
    them to 1 would discard the only machine-readable signal a caller has."""
    monkeypatch.setattr(
        "modelscope_hub.cli.agent.install_agent",
        lambda *a, **k: outcome(ok=False, error="nope", exit_code=plugin_code),
    )
    code, _, err = run_cli(MINIMAL)
    assert code == plugin_code
    assert "nope" in err


def test_failure_without_an_exit_code_becomes_1(monkeypatch):
    monkeypatch.setattr(
        "modelscope_hub.cli.agent.install_agent",
        lambda *a, **k: outcome(ok=False, error="download failed"),
    )
    code, _, err = run_cli(MINIMAL)
    assert code == 1
    assert "download failed" in err
