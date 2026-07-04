# Copyright (c) Alibaba, Inc. and its affiliates.
"""OpenHuman workspace specification (single-agent install)."""
from pathlib import Path
from typing import List

from .._workspace import WorkspaceSpec, register_framework


class OpenhumanWorkspace(WorkspaceSpec):
    """Workspace spec for the OpenHuman agent framework (single-agent install).

    OpenHuman keeps its hidden workspace at ``~/.openhuman/workspace`` with an
    Obsidian-style ``wiki/`` memory vault alongside the persona files.
    """

    @property
    def product_name(self) -> str:
        return "openhuman"

    @property
    def default_workspace_root(self) -> Path:
        return Path.home() / ".openhuman" / "workspace"

    @property
    def patterns(self) -> List[str]:
        return [
            "SOUL.md",
            "IDENTITY.md",
            "USER.md",
            "PROFILE.md",
            "MEMORY.md",
            "HEARTBEAT.md",
            "wiki/*.md",
            "wiki/summaries/*.md",
            "wiki/notes/*.md",
            "skills/*/SKILL.md",
            "skills/*/_meta.json",
            "skills/*/scripts/*",
        ]


register_framework("openhuman", OpenhumanWorkspace)
