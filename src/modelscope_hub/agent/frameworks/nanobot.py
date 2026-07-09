# Copyright (c) Alibaba, Inc. and its affiliates.
"""Nanobot workspace specification (single-agent install)."""
from __future__ import annotations

from pathlib import Path

from .._workspace import WorkspaceSpec, register_framework


class NanobotWorkspace(WorkspaceSpec):
    """Workspace spec for the Nanobot agent framework (single-agent install).

    Nanobot keeps a single shared workspace at ``~/.nanobot/workspace``.  Its
    ``onboard`` command seeds ``AGENTS.md``, ``SOUL.md``, ``USER.md`` and the
    ``HEARTBEAT.md`` task file; long-term memory lives in ``memory/MEMORY.md``
    with an append-only ``memory/HISTORY.md`` event log.  Sub-agents run as
    background sessions (no on-disk per-agent files), so this is single-agent.
    """

    @property
    def product_name(self) -> str:
        return "nanobot"

    @property
    def default_workspace_root(self) -> Path:
        return Path.home() / ".nanobot" / "workspace"

    @property
    def patterns(self) -> list[str]:
        return [
            "AGENTS.md",
            "SOUL.md",
            "USER.md",
            "HEARTBEAT.md",
            "memory/MEMORY.md",
            "memory/HISTORY.md",
            "skills/*/SKILL.md",
            "skills/*/scripts/*",
        ]


register_framework("nanobot", NanobotWorkspace)
