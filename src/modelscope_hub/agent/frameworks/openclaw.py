# Copyright (c) Alibaba, Inc. and its affiliates.
"""OpenClaw workspace specification (root-per-agent)."""
from __future__ import annotations

from pathlib import Path

from .._workspace import WorkspaceSpec, register_framework, DEFAULT_AGENT_NAME


class OpenclawWorkspace(WorkspaceSpec):
    """Workspace spec for the OpenClaw agent framework.

    The default agent lives in ``~/.openclaw/workspace``; named agents live in
    ``~/.openclaw/workspace-<name>``.

    In ``all`` mode, ``workspace_root`` lifts to ``~/.openclaw/`` and patterns
    are prefixed with ``workspace*/`` to match both ``workspace/`` (default) and
    ``workspace-<name>/`` directories.
    """

    @property
    def product_name(self) -> str:
        return "openclaw"

    @property
    def default_workspace_root(self) -> Path:
        base = Path.home() / ".openclaw"
        if self._is_all():
            return base
        if self.agent_name in ("", DEFAULT_AGENT_NAME):
            return base / "workspace"
        return base / f"workspace-{self.agent_name}"

    @property
    def patterns(self) -> list[str]:
        return [
            "AGENTS.md",
            "SOUL.md",
            "USER.md",
            "TOOLS.md",
            "HEARTBEAT.md",
            "IDENTITY.md",
            "BOOTSTRAP.md",
            "MEMORY.md",
            "memory/*.md",
            "memory/*.json",
            "skills/*/SKILL.md",
            "skills/*/_meta.json",
            "skills/*/scripts/*",
        ]

    def _effective_patterns(self) -> list[str]:
        if self._is_all():
            return [f"workspace*/{p}" for p in self.patterns]
        return self.patterns

    def list_agents(self) -> list[str]:
        base = Path.home() / ".openclaw"
        agents: list[str] = []
        if (base / "workspace").is_dir():
            agents.append(DEFAULT_AGENT_NAME)
        if base.is_dir():
            for d in sorted(base.glob("workspace-*")):
                if d.is_dir():
                    agents.append(d.name[len("workspace-"):])
        return agents or [DEFAULT_AGENT_NAME]


register_framework("openclaw", OpenclawWorkspace)
