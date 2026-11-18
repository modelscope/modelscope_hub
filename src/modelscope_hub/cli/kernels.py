"""``ms-hub kernels`` — kernel lockfile helpers backed by ModelScope.

The upstream ``kernels lock`` console command runs in its own process, so it
cannot pick up the in-process ``patch_kernels_hub()`` redirection and always
resolves against the Hugging Face Hub. This subcommand is the ModelScope
equivalent: it reads the same ``[tool.kernels.dependencies]`` table from
``pyproject.toml`` and writes the same ``kernels.lock`` file, with every
dependency resolved against ModelScope instead.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from types import ModuleType

from ..errors import InvalidParameter
from .base import CLICommand, error, info


def _require_kernels() -> ModuleType:
    """Import ``kernels`` or raise an actionable CLI error."""
    try:
        import kernels  # noqa: F401
    except ImportError as e:
        raise InvalidParameter(
            "The `kernels` package is required by `ms-hub kernels` but is not installed. "
            'Install it with: pip install "modelscope-hub[kernels]"',
        ) from e
    return kernels


def _require_kernels_locking() -> None:
    """Fail early with an upgrade hint when ``kernels`` predates lockfiles."""
    try:
        import kernels.locking  # noqa: F401
    except ImportError as e:
        raise InvalidParameter(
            "Kernel lockfiles require `kernels>=0.18`, which is not installed. Upgrade with: pip install -U kernels",
        ) from e


def _read_dependencies(project_dir: Path) -> dict[str, int | str]:
    """Read ``[tool.kernels.dependencies]`` from the project's ``pyproject.toml``."""
    try:
        import tomllib
    except ImportError:
        try:
            from kernels.compat import tomllib  # kernels>=0.18 ships a fallback
        except ImportError:
            try:
                import tomli as tomllib  # explicit fallback for Python < 3.11
            except ImportError as e:
                raise InvalidParameter(
                    "Reading pyproject.toml on Python < 3.11 requires a TOML parser. "
                    "Install one with: pip install tomli",
                ) from e

    pyproject = project_dir / "pyproject.toml"
    if not pyproject.exists():
        raise InvalidParameter(
            f"No pyproject.toml found in {project_dir}; kernel dependencies are declared in its "
            "[tool.kernels.dependencies] table.",
        )
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    dependencies = data.get("tool", {}).get("kernels", {}).get("dependencies")
    if not dependencies:
        raise InvalidParameter(
            "No [tool.kernels.dependencies] table in pyproject.toml; declare the kernels to lock, "
            'e.g. [tool.kernels.dependencies]\n"kernels-community/flash-attn2" = "3".',
        )
    return dependencies


class KernelsCommand(CLICommand):
    """``ms-hub kernels lock``: write a ModelScope-resolved ``kernels.lock``."""

    @staticmethod
    def register(subparsers) -> None:
        parser: ArgumentParser = subparsers.add_parser(
            "kernels",
            help="Kernel lockfile helpers resolved against the ModelScope Hub.",
        )
        sub = parser.add_subparsers(dest="kernels_command", metavar="ACTION")
        sub.required = True

        lock = sub.add_parser(
            "lock",
            help="Write kernels.lock from [tool.kernels.dependencies] (ModelScope commits).",
        )
        lock.add_argument(
            "project_dir",
            nargs="?",
            type=Path,
            default=Path("."),
            help="Project directory holding pyproject.toml (default: current directory).",
        )
        lock.add_argument(
            "--lockfile",
            type=Path,
            default=None,
            help="Lockfile path (default: <project_dir>/kernels.lock).",
        )
        lock.set_defaults(_command=KernelsCommand)

    def execute(self) -> None:
        action = getattr(self.args, "kernels_command", None)
        if action != "lock":  # pragma: no cover - argparse enforces the choice
            error("Unknown kernels action; try `ms-hub kernels --help`.")
            return

        _require_kernels()
        _require_kernels_locking()
        dependencies = _read_dependencies(self.args.project_dir)
        if not isinstance(dependencies, dict) or not all(isinstance(repo, str) for repo in dependencies):
            raise InvalidParameter(
                "[tool.kernels.dependencies] must be a table of repo ids to versions, "
                'e.g. [tool.kernels.dependencies]\n"kernels-community/flash-attn2" = "3".',
            )

        # Imported lazily: the kernels extra is optional and `ms-hub` must stay
        # importable without it.
        from ..kernels import lock_kernels

        default_lockfile = self.args.project_dir / "kernels.lock"
        path = lock_kernels(
            dependencies,
            lockfile=self.args.lockfile or default_lockfile,
        )
        info(f"Wrote {path} — commit it to your project for reproducible kernel loads.")
