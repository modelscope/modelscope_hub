# Copyright (c) Alibaba, Inc. and its affiliates.
"""Nanobot workspace specification (file-per-agent + shared)."""
from __future__ import annotations

from pathlib import Path

from .._workspace import WorkspaceSpec, register_framework


class NanobotWorkspace(WorkspaceSpec):
    """Workspace spec for the Nanobot agent framework (file-per-agent + shared)."""

    @property
    def product_name(self) -> str:
        return "nanobot"

    @property
    def supports_individual_watch(self) -> bool:
        return False

    @property
    def default_workspace_root(self) -> Path:
        return Path.home() / ".nanobot" / "workspace"

    @property
    def patterns(self) -> list[str]:
        return [
            "AGENTS.md",
            "SOUL.md",
            "USER.md",
            "TOOLS.md",
            "HEARTBEAT.md",
            "agents/{name}.md",
            "memory/MEMORY.md",
            "memory/HISTORY.md",
            "skills/*/SKILL.md",
            "skills/*/_meta.json",
            "skills/*/scripts/*",
            "skills/*/setup.md",
            "skills/*/operations.md",
            "skills/*/boundaries.md",
        ]

    def list_agents(self) -> list[str]:
        return self._list_agents_from_dir(self.workspace_root / "agents")


register_framework("nanobot", NanobotWorkspace)
