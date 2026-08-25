"""Tests for console-script ownership and per-alias invocation identity.

This package installs all four ModelScope console scripts, so one parser has to
answer "which command am I?" correctly: the program name in usage, the
``--version`` line, and whether to advertise the umbrella SDK. Split ownership
is what previously stranded users without a CLI at all -- two distributions
writing the same file meant upgrading either one could delete it -- so the
packaging declaration is pinned here alongside the runtime behaviour that
depends on it.
"""

from __future__ import annotations

import importlib.metadata
import logging
import sys
from pathlib import Path

import pytest

from modelscope_hub import __version__
from modelscope_hub.cli import main as cli_main
from modelscope_hub.cli.main import _brand_epilog, _build_parser, _version_text, run_cmd

# Every alias this distribution owns, and the single callable they all reach.
_EXPECTED_SCRIPTS = frozenset({"modelscope", "ms", "modelscope-hub", "ms-hub"})
_ENTRY_TARGET = "modelscope_hub.cli.main:run_cmd"

# Aliases that must never mention the umbrella SDK: they promise the hub only.
_HUB_ALIASES = ("ms-hub", "modelscope-hub")
# Aliases that front the whole brand and therefore report the SDK when present.
_BRAND_ALIASES = ("ms", "modelscope")

_FAKE_SDK_VERSION = "9.9.9"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _clear_sdk_cache():
    """``_sdk_version`` caches for the process; each test needs a fresh answer."""
    cli_main._sdk_version.cache_clear()
    yield
    cli_main._sdk_version.cache_clear()


@pytest.fixture
def invoked_as(monkeypatch):
    """Pretend the process was started by a particular console script."""

    def _set(prog: str) -> None:
        # Full path on purpose: the real argv[0] is absolute, and only the
        # basename may drive the decision.
        monkeypatch.setattr(sys, "argv", [f"/usr/local/bin/{prog}"])
        cli_main._sdk_version.cache_clear()

    return _set


@pytest.fixture
def sdk_absent(monkeypatch):
    """Simulate a lightweight hub-only install (no ``modelscope`` distribution)."""
    real = importlib.metadata.version

    def _fake(dist: str) -> str:
        if dist == cli_main._SDK_DIST:
            raise importlib.metadata.PackageNotFoundError(dist)
        return real(dist)

    monkeypatch.setattr(importlib.metadata, "version", _fake)


@pytest.fixture
def sdk_present(monkeypatch):
    """Simulate the umbrella SDK installed at a known version."""
    real = importlib.metadata.version

    def _fake(dist: str) -> str:
        return _FAKE_SDK_VERSION if dist == cli_main._SDK_DIST else real(dist)

    monkeypatch.setattr(importlib.metadata, "version", _fake)


class _FakeEntryPoint:
    """Minimal stand-in for :class:`importlib.metadata.EntryPoint`."""

    def __init__(self, name: str, target: object, *, boom: bool = False) -> None:
        self.name = name
        self._target = target
        self._boom = boom

    def load(self) -> object:
        if self._boom:
            raise ImportError("optional dependency missing")
        return self._target


@pytest.fixture
def only_plugins(monkeypatch):
    """Replace plugin discovery with a fixed set, so the env cannot leak in."""

    def _install(*eps: _FakeEntryPoint) -> None:
        monkeypatch.setattr(
            importlib.metadata,
            "entry_points",
            lambda group=None: list(eps),
        )

    return _install


@pytest.fixture
def plugin_warnings():
    """Collect warnings emitted by the CLI module.

    The SDK keeps ``propagate = False`` on its own logger tree so it never
    hijacks application logging -- which also means ``caplog``, wired to the
    root logger, cannot see these records. Attach a handler to the emitting
    logger instead, and pin its level so an ambient ``MODELSCOPE_LOG_LEVEL``
    cannot filter the record away.
    """
    logger = logging.getLogger(cli_main.__name__)
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collector(level=logging.WARNING)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


# ---------------------------------------------------------------------------
# Packaging declaration
# ---------------------------------------------------------------------------
class TestPackagingDeclaration:
    """The four aliases must be declared here, by exactly one distribution."""

    @staticmethod
    def _scripts() -> dict[str, str]:
        # tomllib is stdlib from 3.11; the declaration it reads is not
        # version-specific, so checking it on newer interpreters is enough.
        tomllib = pytest.importorskip("tomllib", reason="stdlib tomllib requires Python 3.11+")
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        return tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["scripts"]

    def test_declares_exactly_the_four_aliases(self):
        assert set(self._scripts()) == set(_EXPECTED_SCRIPTS)

    def test_every_alias_shares_one_entry_point(self):
        """Divergent targets would let the aliases drift apart in behaviour."""
        assert set(self._scripts().values()) == {_ENTRY_TARGET}


# ---------------------------------------------------------------------------
# Program name
# ---------------------------------------------------------------------------
class TestProgramName:
    """Usage text must name the alias that ran, not a hard-coded default."""

    @pytest.mark.parametrize("prog", sorted(_EXPECTED_SCRIPTS))
    def test_prog_follows_argv0(self, prog, invoked_as):
        invoked_as(prog)
        assert _build_parser().prog == prog


# ---------------------------------------------------------------------------
# Version line
# ---------------------------------------------------------------------------
class TestVersionLine:
    """``--version`` must name the layer the user actually asked about."""

    @pytest.mark.parametrize("prog", _HUB_ALIASES)
    def test_hub_alias_reports_only_this_package(self, prog, invoked_as, sdk_present):
        """Even with the SDK installed, a ``*-hub`` alias speaks for the hub."""
        invoked_as(prog)
        assert _version_text() == f"modelscope-hub {__version__}"

    @pytest.mark.parametrize("prog", _BRAND_ALIASES)
    def test_brand_alias_leads_with_sdk_version(self, prog, invoked_as, sdk_present):
        invoked_as(prog)
        text = _version_text()
        assert text.startswith(f"modelscope {_FAKE_SDK_VERSION}")
        assert f"modelscope-hub {__version__}" in text

    @pytest.mark.parametrize("prog", _BRAND_ALIASES)
    def test_brand_alias_flags_missing_sdk(self, prog, invoked_as, sdk_absent):
        """A brand alias on a hub-only install must not imply the SDK is there."""
        invoked_as(prog)
        text = _version_text()
        assert text.startswith(f"modelscope-hub {__version__}")
        assert "not installed" in text

    @pytest.mark.parametrize("prog", _HUB_ALIASES)
    def test_hub_alias_stays_quiet_about_missing_sdk(self, prog, invoked_as, sdk_absent):
        """Nothing is missing from a hub-only alias, so there is nothing to say."""
        invoked_as(prog)
        assert _version_text() == f"modelscope-hub {__version__}"

    def test_version_flag_prints_to_stdout_and_exits_zero(self, invoked_as, capsys):
        invoked_as("ms-hub")
        with pytest.raises(SystemExit) as exc_info:
            run_cmd(["--version"])
        assert exc_info.value.code == 0
        assert f"modelscope-hub {__version__}" in capsys.readouterr().out

    def test_version_is_not_resolved_while_building_the_parser(self, invoked_as, monkeypatch):
        """Metadata lookups must not be charged to unrelated subcommands.

        ``--version`` is the only consumer of the SDK version, so building the
        parser for e.g. ``download`` should never pay for it.
        """
        invoked_as("ms-hub")

        def _explode(dist: str) -> str:
            raise AssertionError(f"unexpected metadata lookup for {dist!r}")

        monkeypatch.setattr(importlib.metadata, "version", _explode)
        assert _build_parser() is not None


# ---------------------------------------------------------------------------
# Help epilog
# ---------------------------------------------------------------------------
class TestBrandEpilog:
    """Brand aliases exist on a hub-only install; help must explain the gap."""

    @pytest.mark.parametrize("prog", _BRAND_ALIASES)
    def test_epilog_offers_install_hint_when_sdk_absent(self, prog, invoked_as, sdk_absent):
        invoked_as(prog)
        epilog = _brand_epilog()
        assert epilog is not None
        assert "pip install modelscope" in epilog

    @pytest.mark.parametrize("prog", _BRAND_ALIASES)
    def test_no_epilog_when_sdk_present(self, prog, invoked_as, sdk_present):
        """Nothing is missing, so the hint would be noise."""
        invoked_as(prog)
        assert _brand_epilog() is None

    @pytest.mark.parametrize("prog", _HUB_ALIASES)
    def test_no_epilog_for_hub_alias(self, prog, invoked_as, sdk_absent):
        """``ms-hub`` does exactly what it advertises; no SDK is implied."""
        invoked_as(prog)
        assert _brand_epilog() is None

    def test_epilog_reaches_help_output(self, invoked_as, sdk_absent, capsys):
        invoked_as("modelscope")
        with pytest.raises(SystemExit):
            run_cmd(["--help"])
        assert "pip install modelscope" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------
class TestPluginDiscovery:
    """A contributed plugin must never shadow or break the built-in commands."""

    def test_duplicate_name_is_skipped_with_a_warning(self, only_plugins, plugin_warnings):
        """Silently dropping a collision leaves a packaging bug undiagnosable."""

        class _Shadow:
            name = "download"  # collides with a built-in command

            @staticmethod
            def register(subparsers):
                raise AssertionError("collision must be caught before register()")

        only_plugins(_FakeEntryPoint("shadow", _Shadow))

        parser = _build_parser()

        assert any("already registered" in record.getMessage() for record in plugin_warnings)
        # The built-in survives intact.
        args = parser.parse_args(["download", "owner/name"])
        assert args.repo_id == "owner/name"

    def test_legacy_studio_plugin_collision_is_silent(self, only_plugins, plugin_warnings):
        """Old ``modelscope`` wheels still advertise ``studio``; hub now owns it."""

        class _LegacyStudio:
            name = "studio"

            @staticmethod
            def register(subparsers):
                raise AssertionError("legacy studio plugin must be skipped")

        only_plugins(_FakeEntryPoint("studio", _LegacyStudio))

        parser = _build_parser()

        assert plugin_warnings == []
        assert parser.parse_args(["studio", "deploy", "owner/demo"]).studio_action == "deploy"

    def test_unimportable_plugin_does_not_break_the_cli(self, only_plugins, plugin_warnings):
        """Optional extras are legitimately absent, so this must stay quiet."""
        only_plugins(_FakeEntryPoint("broken", None, boom=True))

        parser = _build_parser()

        assert parser is not None
        assert plugin_warnings == []

    def test_plugin_registers_its_command(self, only_plugins):
        recorded: list[str] = []

        class _Extra:
            name = "brand-new"

            @staticmethod
            def register(subparsers):
                recorded.append("registered")
                subparsers.add_parser(_Extra.name)

        only_plugins(_FakeEntryPoint("extra", _Extra))

        parser = _build_parser()

        assert recorded == ["registered"]
        assert "brand-new" in parser.parse_args(["brand-new"]).command

    def test_plugin_using_define_args_is_supported(self, only_plugins):
        """Legacy plugins expose ``define_args`` instead of ``register``."""

        class _Legacy:
            name = "legacy-cmd"

            @staticmethod
            def define_args(subparsers):
                subparsers.add_parser(_Legacy.name)

        only_plugins(_FakeEntryPoint("legacy", _Legacy))

        assert _build_parser().parse_args(["legacy-cmd"]).command == "legacy-cmd"
