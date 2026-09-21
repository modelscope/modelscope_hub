# Copyright (c) Alibaba, Inc. and its affiliates.
"""Agent repository transport SDK for ModelScope Hub.

This package provides the low-level HTTP client for agent repositories, plus the
plugin loader behind ``ms agent install``. Neither carries framework knowledge:
workspace management (frameworks, conversion, sync, watch, backups) lives in
**modelscope-agent** (``ms_agent.agent_hub``), and the plugin loader only fetches
and invokes a plugin that a publisher ships as a model repository.

Public API
----------
- :class:`AgentApi` -- HTTP client for agent repository operations
  (download/commit/LFS/list/create/delete).
- :class:`RemoteFileInfo` -- metadata for a single remote file.
- :func:`is_lfs_file` -- decide whether a file must use the LFS upload path.
- ``agent_visibility_label`` / ``agent_last_modified`` -- read renamed agent
  metadata fields from an API item, tolerating both JSON spellings
  (snake_case and PascalCase) and legacy keys.
- :func:`install_agent` -- fetch or install an agent through its framework
  plugin, the choice being negotiated with the plugin.
- :func:`default_staging_dir` -- where a fetch-only plugin's files land when the
  caller named no directory.
"""

from ._api import AgentApi, RemoteFileInfo, agent_last_modified, agent_visibility_label, is_lfs_file
from ._plugin import (
    ENTRY_OPERATIONS,
    InstallOutcome,
    PluginSpec,
    assert_trusted_owner,
    default_staging_dir,
    fetch_plugin,
    install_agent,
    load_plugin,
    plugin_syspath,
    resolve_plugin_repo,
    select_operation,
    verify_manifest,
)

__all__ = [
    "AgentApi",
    "RemoteFileInfo",
    "is_lfs_file",
    "agent_visibility_label",
    "agent_last_modified",
    "install_agent",
    "PluginSpec",
    "InstallOutcome",
    "ENTRY_OPERATIONS",
    "resolve_plugin_repo",
    "assert_trusted_owner",
    "fetch_plugin",
    "verify_manifest",
    "load_plugin",
    "plugin_syspath",
    "select_operation",
    "default_staging_dir",
]
