"""Tests for the ``ms-hub kernels lock`` subcommand."""

from __future__ import annotations

import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

from modelscope_hub.cli.kernels import KernelsCommand, _read_dependencies
from modelscope_hub.errors import InvalidParameter


def _write_pyproject(tmp_path: Path, body: str) -> Path:
    (tmp_path / "pyproject.toml").write_text(body)
    return tmp_path


def test_read_dependencies_parses_tool_kernels_table(tmp_path):
    project = _write_pyproject(
        tmp_path,
        '[tool.kernels.dependencies]\n"kernels-community/flash-attn2" = "3"\n',
    )
    assert _read_dependencies(project) == {"kernels-community/flash-attn2": "3"}


def test_read_dependencies_missing_pyproject(tmp_path):
    with pytest.raises(InvalidParameter):
        _read_dependencies(tmp_path)


def test_read_dependencies_missing_table(tmp_path):
    project = _write_pyproject(tmp_path, "[project]\nname = 'x'\n")
    with pytest.raises(InvalidParameter):
        _read_dependencies(project)


def test_lock_execute_assembles_the_call(tmp_path, monkeypatch, capsys):
    project = _write_pyproject(
        tmp_path,
        '[tool.kernels.dependencies]\n"kernels-community/flash-attn2" = "3"\n',
    )
    # Import the real kernels package *before* stubbing its locking module:
    # kernels/__init__ pulls in kernels.install, which imports
    # kernels.locking, and that must resolve against the real module.
    # importorskip: without the kernels extra installed the CLI reports a
    # friendly error, and these tests skip instead of erroring.
    pytest.importorskip("kernels")

    # Simulate kernels>=0.18 (the locking module exists) so the CLI's version
    # gate passes even on a 0.17 install.
    monkeypatch.setitem(sys.modules, "kernels.locking", types.ModuleType("kernels.locking"))

    captured = {}

    import modelscope_hub.kernels as kernels_pkg

    def fake_lock_kernels(dependencies, *, lockfile=None):
        captured["dependencies"] = dependencies
        captured["lockfile"] = lockfile
        return Path("kernels.lock")

    monkeypatch.setattr(kernels_pkg, "lock_kernels", fake_lock_kernels)

    args = Namespace(kernels_command="lock", project_dir=project, lockfile=None)
    KernelsCommand(args).execute()

    assert captured["dependencies"] == {"kernels-community/flash-attn2": "3"}
    assert captured["lockfile"] == project / "kernels.lock"
    assert "kernels.lock" in capsys.readouterr().out


def test_lock_execute_honours_explicit_lockfile(tmp_path, monkeypatch):
    project = _write_pyproject(
        tmp_path,
        '[tool.kernels.dependencies]\n"kernels-community/activation" = 1\n',
    )
    pytest.importorskip("kernels")  # load before the locking stub, see above

    monkeypatch.setitem(sys.modules, "kernels.locking", types.ModuleType("kernels.locking"))

    captured = {}

    import modelscope_hub.kernels as kernels_pkg

    monkeypatch.setattr(
        kernels_pkg,
        "lock_kernels",
        lambda dependencies, *, lockfile=None: (
            captured.update(dependencies=dependencies, lockfile=lockfile) or Path("out.lock")
        ),
    )

    args = Namespace(
        kernels_command="lock",
        project_dir=project,
        lockfile=tmp_path / "out.lock",
    )
    KernelsCommand(args).execute()
    assert captured["dependencies"] == {"kernels-community/activation": 1}
    assert captured["lockfile"] == tmp_path / "out.lock"


def test_lock_execute_rejects_non_table_dependencies(tmp_path, monkeypatch):
    project = _write_pyproject(tmp_path, "[tool.kernels]\ndependencies = 'not-a-table'\n")
    pytest.importorskip("kernels")  # load before the locking stub, see above

    monkeypatch.setitem(sys.modules, "kernels.locking", types.ModuleType("kernels.locking"))

    args = Namespace(kernels_command="lock", project_dir=project, lockfile=None)
    with pytest.raises(InvalidParameter):
        KernelsCommand(args).execute()
