# Copyright (c) Alibaba, Inc. and its affiliates.
"""QwenPaw workspace specification (root-per-agent)."""
from pathlib import Path
from typing import List

from .._workspace import WorkspaceSpec, register_framework, DEFAULT_AGENT_NAME


class QwenpawWorkspace(WorkspaceSpec):
    """Workspace spec for the QwenPaw agent framework.

    QwenPaw stores per-agent workspaces under ``~/.qwenpaw/workspaces/{id}``;
    the default agent lives in ``workspaces/default``.

    In ``all`` mode, ``workspace_root`` lifts to ``~/.qwenpaw/workspaces/`` and
    patterns are prefixed with ``*/`` so that each agent directory becomes a
    path prefix in the collected resource dict.
    """

    @property
    def product_name(self) -> str:
        return "qwenpaw"

    @property
    def default_workspace_root(self) -> Path:
        base = Path.home() / ".qwenpaw" / "workspaces"
        if self._is_all():
            return base
        return base / self.agent_name

    @property
    def patterns(self) -> List[str]:
        return [
            "AGENTS.md",
            "SOUL.md",
            "PROFILE.md",
            "BOOTSTRAP.md",
            "MEMORY.md",
            "HEARTBEAT.md",
            "memory/*.md",
            "skills/*/SKILL.md",
            "skills/*/_meta.json",
            "skills/*/scripts/*",
        ]

    def _effective_patterns(self) -> List[str]:
        if self._is_all():
            return [f"*/{p}" for p in self.patterns]
        return self.patterns

    def list_agents(self) -> List[str]:
        base = Path.home() / ".qwenpaw" / "workspaces"
        if not base.is_dir():
            return [DEFAULT_AGENT_NAME]
        agents = [d.name for d in sorted(base.iterdir()) if d.is_dir()]
        return agents or [DEFAULT_AGENT_NAME]


register_framework("qwenpaw", QwenpawWorkspace)
