# Copyright (c) Alibaba, Inc. and its affiliates.
"""Agent workspace management SDK for ModelScope Hub.

Public API
----------
- :class:`AgentApi` -- HTTP client for agent repository operations.
- :class:`WorkspaceSpec` -- Abstract base for framework workspace definitions.
- :data:`FRAMEWORK_REGISTRY` -- Mapping of framework names to WorkspaceSpec classes.
- :func:`register_framework` -- Register a new framework at runtime.
- :func:`get_defaults` -- Load default templates for a framework.

Built-in frameworks (auto-registered on import):
  qoder, qwenpaw, openclaw, hermes, nanobot, openhuman
"""
from ._api import AgentApi
from ._workspace import (
    DEFAULT_AGENT_NAME,
    ALL_AGENT_NAME,
    GLOBAL_AGENT_NAME,
    FRAMEWORK_REGISTRY,
    WorkspaceSpec,
    register_framework,
)
from ._defaults import get_defaults
from ._merge import (
    FullMergeResult,
    MergeAction,
    MergeResult,
    SectionMerger,
    HeartbeatMerger,
    merge_resources,
)
from ._commands import (
    api_error_message,
    available_frameworks,
    build_spec,
    cmd_convert,
    cmd_download,
    cmd_recover,
    cmd_status,
    cmd_stop,
    cmd_upload,
    cmd_watch,
    convert_resources,
    convert_workspace,
    repo_name,
    resolve_local_name,
    resolve_remote,
)

# Trigger auto-registration of all built-in frameworks.
from . import frameworks as _frameworks  # noqa: F401

__all__ = [
    "AgentApi",
    "WorkspaceSpec",
    "FRAMEWORK_REGISTRY",
    "register_framework",
    "get_defaults",
    "DEFAULT_AGENT_NAME",
    "ALL_AGENT_NAME",
    "GLOBAL_AGENT_NAME",
    "FullMergeResult",
    "MergeAction",
    "MergeResult",
    "SectionMerger",
    "HeartbeatMerger",
    "merge_resources",
    "api_error_message",
    "available_frameworks",
    "build_spec",
    "cmd_convert",
    "cmd_download",
    "cmd_recover",
    "cmd_status",
    "cmd_stop",
    "cmd_upload",
    "cmd_watch",
    "convert_resources",
    "convert_workspace",
    "repo_name",
    "resolve_local_name",
    "resolve_remote",
]
