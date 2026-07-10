# Copyright (c) Alibaba, Inc. and its affiliates.
"""OpenHuman workspace specification (single-agent install)."""
from __future__ import annotations

import re
from pathlib import Path

from .._workspace import WorkspaceSpec, register_framework


class OpenhumanWorkspace(WorkspaceSpec):
    """Workspace spec for the OpenHuman agent framework (single-agent install).

    OpenHuman is a Rust/Tauri desktop app whose brain is a local Memory Tree
    (SQLite at ``memory_tree/chunks.db``) mirrored as an Obsidian-style
    ``wiki/`` Markdown vault under ``~/.openhuman``.  Per its "move to a new
    PC" guide the portable, human-authored state is: the ``wiki/`` vault, the
    persona files ``SOUL.md`` / ``IDENTITY.md`` / ``HEARTBEAT.md`` and the
    ``config.toml`` settings (models / providers / routing / autonomy).

    Deliberately *not* collected: the SQLite stores (``memory_tree/chunks.db``,
    ``approval/approval.db``, ``mcp_clients/mcp_clients.db``) and the session
    history (``sessions/`` / ``session_raw/``) -- binary / run-time state that
    does not migrate across frameworks (the wiki is the readable mirror).
    """

    # ``config.toml`` keys whose value is a machine-local secret and must be
    # blanked before the file leaves / enters a machine.
    _CONFIG_SECRET_KEYS = frozenset([
        "api_key", "openai_api_key", "anthropic_api_key", "composio_api_key",
        "token", "secret", "password",
    ])

    @property
    def product_name(self) -> str:
        return "openhuman"

    @property
    def default_workspace_root(self) -> Path:
        return Path.home() / ".openhuman"

    @property
    def patterns(self) -> list[str]:
        # fnmatch ``*`` spans ``/`` so ``wiki/*`` / ``skills/*`` recurse the
        # whole vault / skill tree.
        return [
            "SOUL.md",
            "IDENTITY.md",
            "HEARTBEAT.md",
            "config.toml",
            "wiki/*",
            "skills/*",
        ]

    # ------------------------------------------------------------------
    # config.toml secret sanitization on the inbound path
    # ------------------------------------------------------------------

    def sanitize_inbound_file(self, rel_path: str, content: bytes) -> bytes:
        """Blank machine-local secrets in inbound ``config.toml``.

        Line-level rewrite (stdlib has no TOML writer): any ``key = <value>``
        assignment whose key is a known secret has its value cleared to ``""``,
        preserving the rest of the file verbatim. Non-TOML content is left
        untouched.
        """
        if rel_path != "config.toml":
            return content
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return content
        return self._scrub_toml_secrets(text).encode("utf-8")

    def _scrub_toml_secrets(self, text: str) -> str:
        pattern = re.compile(
            r"^(\s*(?P<key>[A-Za-z0-9_-]+)\s*=\s*).*$"
        )
        out: list[str] = []
        for line in text.split("\n"):
            m = pattern.match(line)
            if m and m.group("key").lower() in self._CONFIG_SECRET_KEYS:
                out.append(m.group(1) + '""')
            else:
                out.append(line)
        return "\n".join(out)


register_framework("openhuman", OpenhumanWorkspace)
