# Copyright (c) Alibaba, Inc. and its affiliates.
"""QwenPaw workspace specification (root-per-agent)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .._workspace import (
    ALL_AGENT_NAME,
    DEFAULT_AGENT_NAME,
    GLOBAL_AGENT_NAME,
    WorkspaceSpec,
    register_framework,
)
from ...utils.logger import get_logger

logger = get_logger("agent")


class QwenpawWorkspace(WorkspaceSpec):
    """Workspace spec for the QwenPaw (a.k.a. CoPaw) agent framework.

    QwenPaw stores per-agent workspaces under ``~/.qwenpaw/workspaces/{id}`` (legacy ``~/.copaw``);
    the default agent lives in ``workspaces/default``. Unlike other products,
    QwenPaw does *not* discover agents by scanning the workspaces directory:
    the desktop app loads agents from a central registry in
    ``config.json`` (the ``agents.profiles`` map). An agent whose files
    exist on disk but is absent from that registry is invisible in the UI, so
    :meth:`apply` both writes the files *and* registers the agent.

    Portable files:

    * persona markdown (``AGENTS.md``/``SOUL.md``/``PROFILE.md``/... )
    * ``skills/`` (recursively) plus the workspace ``skill.json`` manifest
    * ``agent.json`` -- the agent configuration. It is collected as-is but
      sanitized on :meth:`apply` (machine-specific ``workspace_dir``/``id`` are
      rewritten and per-channel secrets/tokens are stripped).

    In ``all`` mode, ``workspace_root`` lifts to ``<config>/workspaces/`` and
    patterns are prefixed with ``*/`` so that each agent directory becomes a
    path prefix in the collected resource dict.
    """

    # Brand rename CoPaw -> QwenPaw moved the data root from ``~/.copaw``
    # to ``~/.qwenpaw``. Probe both, preferring the current brand name;
    # fall back to the new name when neither exists yet (fresh install).
    _CONFIG_DIRNAMES = (".qwenpaw", ".copaw")

    # channel keys whose values are machine-specific secrets/paths and must
    # never be carried across machines.
    _CHANNEL_SECRET_KEYS = frozenset([
        "bot_token", "bot_token_file", "app_secret", "client_secret",
        "access_token", "secret", "sk", "encrypt_key", "verification_token",
        "twilio_auth_token", "sip_password", "password", "dashscope_api_key",
        "livekit_api_key", "livekit_api_secret", "db_path",
    ])

    @property
    def product_name(self) -> str:
        return "qwenpaw"

    @property
    def _config_root(self) -> Path:
        # Prefer an existing data root; ``.qwenpaw`` wins over the legacy
        # ``.copaw`` when both are present. Default to ``.qwenpaw`` (the
        # current brand) when neither exists yet.
        home = Path.home()
        for name in self._CONFIG_DIRNAMES:
            candidate = home / name
            if candidate.is_dir():
                return candidate
        return home / self._CONFIG_DIRNAMES[0]

    def _effective_config_root(self) -> Path:
        """Config root that owns the *current* workspace.

        Derived from ``workspace_root`` so that a ``--local_dir`` override never
        touches the user's real data root (sanitized ``workspace_dir`` and
        the ``config.json`` registry both live beside the workspace on disk):

        * single-agent: ``<config>/workspaces/<agent>`` -> up two levels.
        * all-mode:     ``<config>/workspaces``          -> up one level.
        """
        root = self.workspace_root
        if self._is_all():
            return root.parent
        return root.parent.parent

    @property
    def default_workspace_root(self) -> Path:
        base = self._config_root / "workspaces"
        if self._is_all():
            return base
        return base / self.agent_name

    @property
    def patterns(self) -> list[str]:
        return [
            "AGENTS.md",
            "SOUL.md",
            "PROFILE.md",
            "BOOTSTRAP.md",
            "MEMORY.md",
            "HEARTBEAT.md",
            "memory/*.md",
            "agent.json",
            "skill.json",
            # fnmatch ``*`` spans ``/`` so ``skills/*`` recurses the
            # whole skill tree (SKILL.md + references/assets/scripts).
            "skills/*",
        ]

    def _effective_patterns(self) -> list[str]:
        if self._is_all():
            return [f"*/{p}" for p in self.patterns]
        return self.patterns

    @property
    def is_root_per_agent(self) -> bool:
        return True

    def split_all_path(self, rel_path: str) -> tuple[str | None, str]:
        # agent directory name IS the agent name: ``<agent>/<bare>``.
        if "/" in rel_path:
            head, rest = rel_path.split("/", 1)
            return (head, rest)
        return (None, rel_path)

    def join_all_path(self, agent_name: str, bare_path: str) -> str:
        return f"{agent_name}/{bare_path}"

    def list_agents(self) -> list[str]:
        base = self._config_root / "workspaces"
        if not base.is_dir():
            return [DEFAULT_AGENT_NAME]
        agents = [d.name for d in sorted(base.iterdir()) if d.is_dir()]
        return agents or [DEFAULT_AGENT_NAME]

    # ------------------------------------------------------------------
    # agent.json sanitization + config.json registration on apply
    # ------------------------------------------------------------------

    def _sanitize_agent_json(self, agent_name: str, content: str) -> str:
        """Rewrite machine-specific fields and strip per-channel secrets.

        Returns the sanitized JSON text. On parse failure the original text is
        returned unchanged (best-effort; a malformed file should not abort the
        whole download).
        """
        try:
            data: Any = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            logger.warning("agent.json is not valid JSON; writing as-is")
            return content
        if not isinstance(data, dict):
            return content
        # Rebind identity/location to the local target agent.
        data["id"] = agent_name
        data["workspace_dir"] = str(self._effective_config_root() / "workspaces" / agent_name)
        # Strip secrets from every channel config.
        channels = data.get("channels")
        if isinstance(channels, dict):
            for ch in channels.values():
                if not isinstance(ch, dict):
                    continue
                for key in list(ch.keys()):
                    if key in self._CHANNEL_SECRET_KEYS:
                        ch[key] = ""
        # Strip MCP env secrets (API keys live under mcp.clients.*.env).
        mcp = data.get("mcp")
        if isinstance(mcp, dict):
            clients = mcp.get("clients")
            if isinstance(clients, dict):
                for client in clients.values():
                    if isinstance(client, dict) and isinstance(client.get("env"), dict):
                        client["env"] = {k: "" for k in client["env"]}
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _register_agent(self, agent_name: str) -> None:
        """Add *agent_name* to the data root's ``config.json`` ``agents`` registry.

        Best-effort and idempotent: an existing profile is updated in place and
        the agent is appended to ``agent_order`` only if absent. When the config
        file is missing or unparsable the registration is skipped.
        """
        if agent_name in (ALL_AGENT_NAME, GLOBAL_AGENT_NAME):
            return
        config_path = self._effective_config_root() / "config.json"
        if not config_path.is_file():
            logger.warning(
                "QwenPaw config.json not found at %s; agent %r written but not "
                "registered (it will be invisible in the UI until QwenPaw "
                "rescans).", config_path, agent_name,
            )
            return
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.warning("Cannot read QwenPaw config.json (%s); skip register", e)
            return
        if not isinstance(cfg, dict):
            return
        agents = cfg.get("agents")
        if not isinstance(agents, dict):
            agents = {}
            cfg["agents"] = agents
        profiles = agents.get("profiles")
        if not isinstance(profiles, dict):
            profiles = {}
            agents["profiles"] = profiles
        workspace_dir = str(self._effective_config_root() / "workspaces" / agent_name)
        existing = profiles.get(agent_name)
        profile = existing if isinstance(existing, dict) else {}
        profile.update({
            "id": agent_name,
            "workspace_dir": workspace_dir,
            "enabled": True,
        })
        profiles[agent_name] = profile
        order = agents.get("agent_order")
        if not isinstance(order, list):
            order = list(profiles.keys())
            agents["agent_order"] = order
        if agent_name not in order:
            order.append(agent_name)
        try:
            config_path.write_text(
                json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as e:
            logger.warning("Cannot write QwenPaw config.json (%s); skip register", e)

    def sanitize_inbound_file(self, rel_path: str, content: bytes) -> bytes:
        """Sanitize inbound ``agent.json`` (strip secrets, rebind identity).

        Applied uniformly on every inbound write path (full ``apply`` and
        incremental ``pull_incremental``), so a remote ``agent.json`` never
        lands on disk with secrets regardless of sync mode.
        """
        if self._is_all():
            agent, bare = self.split_all_path(rel_path)
            if not (agent and bare == "agent.json"):
                return content
            target_agent = agent
        else:
            if rel_path != "agent.json":
                return content
            target_agent = self.agent_name or DEFAULT_AGENT_NAME
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return content
        return self._sanitize_agent_json(target_agent, text).encode("utf-8")

    def apply(self, resources: dict[str, str]) -> list[str]:
        """Write files (``agent.json`` sanitized via the inbound hook) then
        register the agent(s) according to scope.

        Registration policy differs by scope:

        * ``-n all`` (whole-repo batch sync) writes files **without touching**
          the ``config.json`` registry. Batch sync only mirrors file content;
          which agents are visible in the UI is owned by the user (via the
          QwenPaw UI). Registering here would resurrect agents the user has
          deleted in the UI (their ``agent.json`` may still linger on disk,
          get re-uploaded, then pulled back), so ``all`` never registers.
        * ``-n <agent>`` (explicit single agent) registers that agent, so a
          freshly downloaded agent becomes visible in the UI as intended.
        """
        # ``super().apply`` runs ``sanitize_inbound_file`` per file, so
        # ``agent.json`` sanitization happens there -- do not re-sanitize here.
        written = super().apply(resources)
        if self._is_all():
            logger.info(
                "Applied %d file(s) in all-mode; registry left unchanged "
                "(agent visibility is owned by the QwenPaw UI).", len(written)
            )
            return written
        agent_name = self.agent_name or DEFAULT_AGENT_NAME
        self._register_agent(agent_name)
        return written


register_framework("qwenpaw", QwenpawWorkspace)
