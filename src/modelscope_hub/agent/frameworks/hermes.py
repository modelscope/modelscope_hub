# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hermes workspace specification (single-agent install)."""
from __future__ import annotations

from pathlib import Path

from .._workspace import WorkspaceSpec, register_framework


class HermesWorkspace(WorkspaceSpec):
    """Workspace spec for the Hermes agent framework (single-agent install)."""

    @property
    def product_name(self) -> str:
        return "hermes"

    @property
    def default_workspace_root(self) -> Path:
        return Path.home() / ".hermes"

    @property
    def patterns(self) -> list[str]:
        return [
            "SOUL.md",
            "memories/*.md",
            "skills/*/SKILL.md",
            "skills/*/DESCRIPTION.md",
            "skills/*/_meta.json",
            "skills/*/scripts/*",
            "skills/*/references/*",
            "skills/*/*/SKILL.md",
            "skills/*/*/_meta.json",
            "skills/*/*/scripts/*",
            "skills/*/*/references/*",
        ]


register_framework("hermes", HermesWorkspace)
