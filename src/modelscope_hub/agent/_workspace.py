# Copyright (c) Alibaba, Inc. and its affiliates.
"""Agent workspace file specification and framework registry.

Each agent framework stores its files in a known on-disk layout.  A subclass of
:class:`WorkspaceSpec` declares, for one framework, *where* the files live
(``workspace_root``) and *which* of them are portable (``patterns``).
``collect()`` walks the workspace and returns
``{workspace_relative_path: text_content}``.

Sub-agents
----------
A single installation can host several sub-agents.  There are three layouts:

* **root-per-agent** -- the sub-agent *is* a directory; selecting it changes
  ``workspace_root`` (e.g. qwenpaw ``workspaces/<name>``).
* **file-per-agent + shared** -- the sub-agent is one file inside a shared root,
  collected alongside the shared resources; a ``{name}`` placeholder in
  ``patterns`` is formatted with the sub-agent name (e.g. qoder ``agents/<name>.md``).
* **single-agent** -- one persona per install; the sub-agent name is only the
  repository identity and does not affect file selection.

Framework Registration
----------------------
Use :func:`register_framework` to add a new framework at runtime::

    from modelscope_hub.agent import WorkspaceSpec, register_framework

    class MyFramework(WorkspaceSpec):
        ...

    register_framework("my-framework", MyFramework)
"""
from __future__ import annotations

import fnmatch
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type

from ..utils.logger import get_logger

logger = get_logger("agent")

MAX_FILE_SIZE = 1 * 1024 * 1024  # 1 MB

DEFAULT_AGENT_NAME = "default"
ALL_AGENT_NAME = "all"
GLOBAL_AGENT_NAME = "__global__"


class WorkspaceSpec(ABC):
    """Abstract base for agent framework workspace file specifications.

    :param agent_name: the sub-agent to operate on.  Used to resolve
        ``workspace_root`` (root-per-agent) and/or to format ``{name}``
        placeholders in ``patterns`` (file-per-agent).

        The special value ``"all"`` selects *every* sub-agent at once.

    :param local_dir: explicit workspace root override; when given, it replaces
        the framework's default ``workspace_root``.
    """

    def __init__(
        self, agent_name: str = DEFAULT_AGENT_NAME, local_dir: Path | None = None
    ):
        self.agent_name = agent_name or DEFAULT_AGENT_NAME
        self._local_dir = Path(local_dir).expanduser() if local_dir else None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def product_name(self) -> str:
        ...

    @property
    @abstractmethod
    def default_workspace_root(self) -> Path:
        """Workspace root for ``self.agent_name`` (before ``local_dir`` override)."""
        ...

    @property
    @abstractmethod
    def patterns(self) -> list[str]:
        """fnmatch globs (workspace-relative); may contain ``{name}``."""
        ...

    # ------------------------------------------------------------------
    # All-mode & watch constraint
    # ------------------------------------------------------------------

    def _is_all(self) -> bool:
        """Whether we are in 'all sub-agents' mode."""
        return self.agent_name == ALL_AGENT_NAME

    @property
    def supports_individual_watch(self) -> bool:
        """Whether ``watch`` supports a single sub-agent name.

        File-per-agent+shared products must override this to ``False`` because
        shared files would cascade changes between repos.
        """
        return True

    def _effective_patterns(self) -> list[str]:
        """Patterns to match against.  Root-per-agent classes override this to
        add an agent-name prefix in all mode."""
        return self.patterns

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    @property
    def workspace_root(self) -> Path:
        """Effective root: ``local_dir`` override, else the framework default."""
        return self._local_dir if self._local_dir is not None else self.default_workspace_root

    def _is_global(self) -> bool:
        """Whether we are in global-only mode (shared files only, no sub-agent)."""
        return self.agent_name == GLOBAL_AGENT_NAME

    def resolved_patterns(self) -> list[str]:
        """Resolve glob patterns for the current agent mode.

        Convention: In global mode (``GLOBAL_AGENT_NAME``), patterns containing
        the ``{name}`` placeholder are excluded because they target specific
        sub-agents.  Shared/framework-level patterns (those without ``{name}``)
        remain.
        """
        if self._is_global():
            return [p for p in self._effective_patterns() if "{name}" not in p]
        name = "*" if self._is_all() else self.agent_name
        return [p.format(name=name) for p in self._effective_patterns()]

    def matches(self, rel_path: str, patterns: list[str]) -> bool:
        """Return True if *rel_path* matches any of the given glob *patterns*."""
        for pattern in patterns:
            if fnmatch.fnmatch(rel_path, pattern):
                return True
        return False

    def _walk_matched(self) -> list[tuple[str, Path]]:
        """Walk workspace and return (rel_path, Path) for matched files."""
        root = self.workspace_root
        if not root.is_dir():
            return []
        patterns = self.resolved_patterns()
        matched: list[tuple[str, Path]] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for fname in sorted(filenames):
                if fname.startswith("."):
                    continue
                f = Path(dirpath) / fname
                if f.is_symlink():
                    continue
                try:
                    rel = f.relative_to(root).as_posix()
                except ValueError:
                    continue
                if not self.matches(rel, patterns):
                    continue
                try:
                    if f.stat().st_size > MAX_FILE_SIZE:
                        continue
                except OSError:
                    continue
                matched.append((rel, f))
        return matched

    def collect(self) -> dict[str, str]:
        """Gather allowed workspace files as {relative_path: text_content}."""
        result: dict[str, str] = {}
        for rel, f in self._walk_matched():
            try:
                result[rel] = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                logger.warning("Skip %s: %s", f, e)
        return result

    def collect_bytes(self) -> dict[str, bytes]:
        """Gather allowed workspace files as {relative_path: raw_bytes}.

        Unlike :meth:`collect`, this includes binary files and does not skip
        on UnicodeDecodeError.
        """
        result: dict[str, bytes] = {}
        for rel, f in self._walk_matched():
            try:
                result[rel] = f.read_bytes()
            except OSError as e:
                logger.warning("Skip %s: %s", f, e)
        return result

    def list_agents(self) -> list[str]:
        """Discover sub-agent names available on disk.

        Default: single-agent products report ``["default"]``.  Root-per-agent
        and file-per-agent products override this to enumerate their layout.
        """
        return [DEFAULT_AGENT_NAME]

    def _list_agents_from_dir(self, agents_dir: Path) -> list[str]:
        """List agents from a directory, prepending DEFAULT if not present."""
        agents = _list_agent_files(agents_dir)
        if DEFAULT_AGENT_NAME not in agents:
            agents = [DEFAULT_AGENT_NAME] + agents
        return agents

    def apply(self, resources: dict[str, str]) -> list[str]:
        """Write resource files back to the workspace.  Returns list of written paths."""
        root = self.workspace_root.resolve()
        written: list[str] = []
        for rel_path, content in resources.items():
            target = (root / rel_path).resolve()
            if not target.is_relative_to(root):
                logger.warning("Path traversal blocked: %s", rel_path)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written.append(str(target))
        return written


def _list_agent_files(agents_dir: Path) -> list[str]:
    """Return the stems of ``*.md`` files in an ``agents/`` directory."""
    if not agents_dir.is_dir():
        return []
    return sorted(f.stem for f in agents_dir.glob("*.md") if f.is_file())


# ---------------------------------------------------------------------------
# Framework registry
# ---------------------------------------------------------------------------
FRAMEWORK_REGISTRY: dict[str, Type[WorkspaceSpec]] = {}


def register_framework(name: str, cls: Type[WorkspaceSpec]) -> None:
    """Register a framework workspace spec.  Idempotent.

    Example::

        from modelscope_hub.agent import WorkspaceSpec, register_framework

        class MyCustomWorkspace(WorkspaceSpec):
            ...

        register_framework("my-framework", MyCustomWorkspace)
    """
    FRAMEWORK_REGISTRY[name] = cls
