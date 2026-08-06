# Copyright (c) Alibaba, Inc. and its affiliates.
"""Agent repository transport SDK for ModelScope Hub.

This package provides only the low-level HTTP client for agent repositories.
Framework-aware workspace management (frameworks, conversion, sync, watch,
backups) lives in **modelscope-agent** (``ms_agent.agent_hub``).

Public API
----------
- :class:`AgentApi` -- HTTP client for agent repository operations
  (download/commit/LFS/list/create/delete).
- :class:`RemoteFileInfo` -- metadata for a single remote file.
- :func:`is_lfs_file` -- decide whether a file must use the LFS upload path.
- ``agent_visibility_label`` / ``agent_last_modified`` -- read renamed agent
  metadata fields from an API item, tolerating both JSON spellings
  (snake_case and PascalCase) and legacy keys.
"""

from ._api import AgentApi, RemoteFileInfo, agent_last_modified, agent_visibility_label, is_lfs_file

__all__ = [
    "AgentApi",
    "RemoteFileInfo",
    "is_lfs_file",
    "agent_visibility_label",
    "agent_last_modified",
]
