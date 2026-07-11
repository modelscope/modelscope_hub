# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hermes workspace specification (root-per-agent).

Hermes stores the *default* agent at the install root (``~/.hermes/``) and every
*named* agent under ``~/.hermes/profiles/<name>/``.  Each agent is a
self-contained directory with its own ``SOUL.md``, ``memories/``, ``skills/``
and a ``config.yaml`` (which also holds the ``mcp_servers`` MCP block) -- i.e. a
root-per-agent layout (1 agent == 1 directory), directly analogous to OpenClaw
(whose named agents live in ``workspace-<name>/``).

File list is aligned with the official Hermes docs
(https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files):
``SOUL.md`` is the *global* personality file loaded only from HERMES_HOME; the
project-scoped context files (``.hermes.md``/``AGENTS.md``/``CLAUDE.md``/
``.cursorrules``) live beside the user's code, not in HERMES_HOME, so they are
out of scope for agent-home migration.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from .._workspace import WorkspaceSpec, register_framework, DEFAULT_AGENT_NAME
from ._bundled_skills import BundledSkillFilterMixin
from ...utils.logger import get_logger

logger = get_logger("agent")


class HermesWorkspace(BundledSkillFilterMixin, WorkspaceSpec):
    """Workspace spec for the Hermes agent framework (root-per-agent).

    * default agent: ``~/.hermes/``
    * named agents:  ``~/.hermes/profiles/<name>/``

    In ``all`` mode, ``workspace_root`` lifts to ``~/.hermes/`` and patterns
    match both the root (default agent) and ``profiles/<name>/`` (named agents).
    """

    # ``config.yaml`` keys that are machine-local secrets and must never be
    # carried across machines. Everything under ``mcp_servers.*.env`` is also
    # treated as a secret bag (API keys live there).
    _CONFIG_SECRET_KEYS = frozenset([
        "api_key", "api_keys", "openrouter_api_key", "anthropic_api_key",
        "openai_api_key", "token", "secret", "password",
    ])

    @property
    def product_name(self) -> str:
        return "hermes"

    @property
    def default_root(self) -> Path:
        return Path.home() / ".hermes"

    @property
    def workspace_root(self) -> Path:
        base = self.root
        if self._is_all() or self.agent_name in ("", DEFAULT_AGENT_NAME):
            return base
        return base / "profiles" / self.agent_name

    @property
    def patterns(self) -> list[str]:
        # NOTE: fnmatch ``*`` spans ``/``, so ``skills/*`` matches every file
        # under ``skills/`` at any nesting depth (category dirs, scripts,
        # references, schemas -- observed up to 7 levels deep). Hermes ships
        # both a bundled ``skills/`` and an official ``optional-skills/`` tree
        # sharing the same structure, so both are captured recursively.
        #
        # ``config.yaml`` carries the ``mcp_servers`` MCP block (plus model /
        # terminal settings); it is collected here and stripped of secrets on
        # the inbound path via ``sanitize_inbound_file``.
        return [
            "SOUL.md",
            "memories/*.md",
            "skills/*",
            "optional-skills/*",
            "config.yaml",
        ]

    def _effective_patterns(self) -> list[str]:
        # In all mode we must capture BOTH the default agent (bare files at the
        # root) and every named agent (under ``profiles/<name>/``).  The two
        # pattern groups are mutually exclusive: bare ``SOUL.md`` never matches
        # ``profiles/x/SOUL.md`` (literal prefix differs) and ``profiles/*/...``
        # never matches the root files.
        if self._is_all():
            return self.patterns + [f"profiles/*/{p}" for p in self.patterns]
        return self.patterns

    @property
    def is_root_per_agent(self) -> bool:
        return True

    def split_all_path(self, rel_path: str) -> tuple[str | None, str]:
        # Named agents live under ``profiles/<name>/``; anything else is the
        # default agent living at the root.
        if rel_path.startswith("profiles/"):
            rest = rel_path[len("profiles/"):]
            if "/" in rest:
                head, bare = rest.split("/", 1)
                return (head, bare)
            return (None, rel_path)
        return (DEFAULT_AGENT_NAME, rel_path)

    def join_all_path(self, agent_name: str, bare_path: str) -> str:
        if agent_name in ("", DEFAULT_AGENT_NAME):
            return bare_path
        return f"profiles/{agent_name}/{bare_path}"

    def list_agents(self) -> list[str]:
        base = self.root
        agents: list[str] = []
        if (base / "SOUL.md").is_file():
            agents.append(DEFAULT_AGENT_NAME)
        profiles = base / "profiles"
        if profiles.is_dir():
            for d in sorted(profiles.iterdir()):
                if d.is_dir() and not d.name.startswith("."):
                    agents.append(d.name)
        return agents or [DEFAULT_AGENT_NAME]

    # ------------------------------------------------------------------
    # config.yaml secret sanitization on the inbound path
    # ------------------------------------------------------------------

    def _sanitize_config_yaml(self, text: str) -> str:
        """Blank out machine-local secrets in ``config.yaml``.

        Keeps the structure intact (model / terminal / mcp_servers) but empties
        API keys / tokens, both at the top level and inside every
        ``mcp_servers.*.env`` block. On parse failure the original text is
        returned unchanged (best-effort; a malformed file must not abort the
        whole download).
        """
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            logger.warning("hermes config.yaml is not valid YAML; writing as-is")
            return text
        if not isinstance(data, dict):
            return text

        def _scrub(node) -> None:
            # Recursively blank secret-looking scalar values anywhere in the
            # tree (e.g. top-level ``model.api_key`` as well as nested blocks).
            if isinstance(node, dict):
                for key in list(node.keys()):
                    val = node[key]
                    if key in self._CONFIG_SECRET_KEYS and not isinstance(val, (dict, list)):
                        node[key] = ""
                    else:
                        _scrub(val)
            elif isinstance(node, list):
                for item in node:
                    _scrub(item)

        _scrub(data)
        # ``mcp_servers.*.env`` is a free-form secret bag (API keys live under
        # arbitrary key names), so blank every value regardless of key name.
        servers = data.get("mcp_servers")
        if isinstance(servers, dict):
            for srv in servers.values():
                if isinstance(srv, dict) and isinstance(srv.get("env"), dict):
                    srv["env"] = {k: "" for k in srv["env"]}
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)

    def sanitize_inbound_file(self, rel_path: str, content: bytes) -> bytes:
        """Strip secrets from inbound ``config.yaml`` (root or profiles/<name>/)."""
        if self._is_all():
            _agent, bare = self.split_all_path(rel_path)
            if bare != "config.yaml":
                return content
        else:
            if rel_path != "config.yaml":
                return content
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return content
        return self._sanitize_config_yaml(text).encode("utf-8")


register_framework("hermes", HermesWorkspace)
