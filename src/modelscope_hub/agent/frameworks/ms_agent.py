# Copyright (c) Alibaba, Inc. and its affiliates.
"""ms-agent workspace specification (single-agent install)."""
from __future__ import annotations

from pathlib import Path

from .._workspace import WorkspaceSpec, register_framework


class MsAgentWorkspace(WorkspaceSpec):
    """Workspace spec for the ms-agent framework (single-agent install).

    ms-agent keeps its persona, memory and skills under ``~/.ms_agent``:

    * **persona** -- a single ``profile.md`` augmented by injected
      configuration (project-level ``config.yaml``, global ``settings.json``
      and a user-specified ``agent.yaml``).
    * **memory** -- ``MEMORY.md`` plus a structured ``facts.json``.
    * **skills** -- ``skills/<name>/SKILL.md`` with a workspace-level
      ``skill.json`` metadata index.

    Only ``profile.md`` (persona) and ``MEMORY.md`` (memory) carry
    cross-framework semantics; the YAML/JSON config and metadata files are
    ms-agent specific and are preserved on same-framework sync only.
    """

    @property
    def product_name(self) -> str:
        return "ms-agent"

    @property
    def default_workspace_root(self) -> Path:
        return Path.home() / ".ms_agent"

    @property
    def patterns(self) -> list[str]:
        return [
            # Persona + injected configuration
            "profile.md",
            "config.yaml",
            "settings.json",
            "agent.yaml",
            # Memory
            "MEMORY.md",
            "facts.json",
            # Skills
            "skill.json",
            "skills/*/SKILL.md",
        ]


register_framework("ms-agent", MsAgentWorkspace)
