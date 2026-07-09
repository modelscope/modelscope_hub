# Copyright (c) Alibaba, Inc. and its affiliates.
"""OpenHuman workspace specification (single-agent install)."""
from __future__ import annotations

from pathlib import Path

from .._workspace import WorkspaceSpec, register_framework


class OpenhumanWorkspace(WorkspaceSpec):
    """Workspace spec for the OpenHuman agent framework (single-agent install).

    OpenHuman is a Rust/Tauri desktop app whose brain is a local Memory Tree
    (SQLite at ``memory_tree/chunks.db``) mirrored as an Obsidian-style
    ``wiki/`` Markdown vault under ``~/.openhuman``.  Only the human-readable
    ``wiki/`` vault is portable; the SQLite store and app config are not.
    OpenHuman has no OpenClaw-style persona files (SOUL/IDENTITY/USER/...).
    """

    @property
    def product_name(self) -> str:
        return "openhuman"

    @property
    def default_workspace_root(self) -> Path:
        return Path.home() / ".openhuman"

    @property
    def patterns(self) -> list[str]:
        # ``*`` in fnmatch spans ``/`` so this recurses the whole wiki vault.
        return [
            "wiki/*.md",
        ]


register_framework("openhuman", OpenhumanWorkspace)
