# Copyright (c) Alibaba, Inc. and its affiliates.
"""Core command logic for agent workspace management.

This module contains the business logic for agent upload, download, convert,
watch, stop, and recover operations.  The CLI adapter (``modelscope_hub.cli.agent``)
calls these functions to perform the actual work.
"""
from __future__ import annotations

import getpass
import os
import sys
import zipfile
from pathlib import Path

from ..utils.logger import get_logger
from ._workspace import (
    FRAMEWORK_REGISTRY,
    ALL_AGENT_NAME,
    DEFAULT_AGENT_NAME,
    GLOBAL_AGENT_NAME,
    WorkspaceSpec,
)
from ._defaults import get_defaults
from ._merge import merge_resources
from ._api import AgentApi
from ..errors import APIError

logger = get_logger("agent")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _fail(message: str) -> int:
    """Print an error and return exit code 1."""
    print(f"Error: {message}", file=sys.stderr)
    return 1


def api_error_message(e: APIError, action: str = "request") -> str:
    """Return a user-friendly message based on the HTTP status code."""
    status = e.status_code or 0
    if status == 401:
        return "authentication failed. Please login again."
    if status == 403:
        return "permission denied. You do not have access to this resource."
    if status == 404:
        return "resource not found. Check the repository name and try again."
    if status >= 500:
        return "server encountered an issue. Please wait a moment and try again."
    return f"{action} failed (HTTP {status}: {e.message})"


def repo_name(framework: str, name: str) -> str:
    """Derive the remote repository name from framework and sub-agent name.

    - name is "all" or empty: use framework alone
    - Both provided: ``{framework}-{name}``
    - Only one provided: use that value directly
    - Neither provided: ``"default"``
    """
    fw = (framework or "").strip()
    n = (name or "").strip()
    if n == ALL_AGENT_NAME:
        n = ""
    if fw and n:
        return f"{fw}-{n}"
    if fw:
        return fw
    if n:
        return n
    return "default"


def resolve_remote(
    repo: str | None = None,
    name: str | None = None,
    framework: str = "",
    username: str = "",
) -> tuple[str, str]:
    """Resolve remote target as (group, repo_name).

    - repo contains '/' -> split into (group, repo_name), ignore username
    - repo without '/' -> (username, repo)
    - repo is None/empty -> derive from name+framework using repo_name logic
    """
    if repo:
        if "/" in repo:
            parts = repo.split("/", 1)
            return parts[0], parts[1]
        return username, repo
    derived = repo_name(framework, name or "")
    return username, derived


def resolve_local_name(name: str | None, framework: str, local_dir=None):
    """Resolve local agent name when --name is omitted.

    Returns (resolved_name, error_message).
    - If *name* is given -> use it directly.
    - If omitted:
      - root-per-agent / single-agent layout (no ``{name}`` placeholder) ->
        always ``DEFAULT_AGENT_NAME``.  'default' is a real workspace
        directory, so sibling sub-agents never trigger auto-select or an error.
      - file-per-agent+shared layout (patterns use ``{name}``) -> inspect the
        ``agents/`` files: exactly 1 non-default -> auto-select it; 0 ->
        ``GLOBAL_AGENT_NAME`` (shared files only); multiple -> error.
    """
    if name:
        return name, None

    spec_cls = FRAMEWORK_REGISTRY[framework]
    local = Path(local_dir).expanduser() if local_dir else None
    tmp_spec = spec_cls(agent_name=DEFAULT_AGENT_NAME, local_dir=local)

    # root-per-agent / single-agent: an omitted --name always means the default
    # agent.  Only layouts with a ``{name}`` placeholder (file-per-agent+shared)
    # have a meaningful shared/global mode or per-agent auto-selection.
    has_shared_mode = any("{name}" in p for p in tmp_spec.patterns)
    if not has_shared_mode:
        return DEFAULT_AGENT_NAME, None

    agents = tmp_spec.list_agents()
    real_agents = [a for a in agents if a != DEFAULT_AGENT_NAME]
    if len(real_agents) == 1:
        return real_agents[0], None
    if len(real_agents) == 0:
        return GLOBAL_AGENT_NAME, None
    return None, (
        f"multiple sub-agents found: {', '.join(agents)}. "
        f"Please specify --name to select one."
    )


def available_frameworks() -> str:
    """Comma-separated list of registered frameworks."""
    return ", ".join(sorted(FRAMEWORK_REGISTRY))


def build_spec(framework: str, name: str, local_dir=None) -> WorkspaceSpec:
    """Build a WorkspaceSpec instance for the given framework and agent name."""
    spec_cls = FRAMEWORK_REGISTRY[framework]
    local = Path(local_dir).expanduser() if local_dir else None
    return spec_cls(agent_name=name, local_dir=local)


def convert_resources(
    resources: dict,
    source_fw: str,
    target_fw: str,
    existing_files: set[str] | None = None,
    *,
    all_mode: bool = False,
    src_spec: WorkspaceSpec | None = None,
    dst_spec: WorkspaceSpec | None = None,
) -> dict:
    """Convert workspace resources from one framework format to another.

    Reuses the cross-framework merge engine.  No-op when source == target.

    When *existing_files* is provided, default template files that already
    exist on the target are filtered out so the target's custom content is
    preserved.  Default templates for files the target does NOT have are kept.

    When *all_mode* is True (root-per-agent -> root-per-agent), paths carry an
    agent prefix; each agent is split out, converted independently as a single
    agent, then re-prefixed for the target framework.  Requires *src_spec* and
    *dst_spec*.
    """
    if source_fw == target_fw:
        return resources
    if all_mode:
        return _convert_resources_all(resources, source_fw, target_fw, src_spec, dst_spec)
    result = merge_resources(
        incoming=resources,
        source_product=source_fw,
        target_product=target_fw,
        source_defaults=get_defaults(source_fw),
        target_defaults=get_defaults(target_fw),
    )
    merged = result.merged_files
    if existing_files is not None:
        default_paths = {a.path for a in result.actions if a.action == "default"}
        merged = {
            k: v for k, v in merged.items()
            if k not in default_paths or k not in existing_files
        }
    return merged


def _convert_resources_all(
    resources: dict,
    source_fw: str,
    target_fw: str,
    src_spec: WorkspaceSpec,
    dst_spec: WorkspaceSpec,
) -> dict:
    """All-mode cross-framework convert (root-per-agent -> root-per-agent).

    Group incoming files by their source agent prefix, convert each agent as an
    isolated single-agent workspace, then re-prefix the results using the target
    framework's convention.  Top-level files without an agent prefix (e.g.
    README.md) belong to no agent and are dropped.
    """
    groups: dict[str, dict[str, str]] = {}
    for path, content in resources.items():
        agent, bare = src_spec.split_all_path(path)
        if agent is None:
            continue
        groups.setdefault(agent, {})[bare] = content

    src_defaults = get_defaults(source_fw)
    tgt_defaults = get_defaults(target_fw)
    out: dict[str, str] = {}
    for agent, bare_files in groups.items():
        result = merge_resources(
            incoming=bare_files,
            source_product=source_fw,
            target_product=target_fw,
            source_defaults=src_defaults,
            target_defaults=tgt_defaults,
        )
        for bare_path, content in result.merged_files.items():
            out[dst_spec.join_all_path(agent, bare_path)] = content
    return out


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_status(framework: str, local_dir=None) -> int:
    """List discoverable sub-agents for a framework."""
    if framework not in FRAMEWORK_REGISTRY:
        return _fail(f"unknown framework '{framework}'. Available: {available_frameworks()}")

    spec = build_spec(framework, DEFAULT_AGENT_NAME, local_dir)
    agents = spec.list_agents()
    print(f"Agents for {framework}:")
    for a in agents:
        tmp = build_spec(framework, a, local_dir)
        files = tmp.collect_bytes()
        print(f"  {a} — {len(files)} file(s), root: {tmp.workspace_root}")
        for rel in sorted(files):
            print(f"    {rel}")
    return 0


def cmd_upload(
    framework: str,
    name: str | None = None,
    local_dir=None,
    repo: str | None = None,
    dry_run: bool = False,
    *,
    endpoint: str | None = None,
    token: str | None = None,
    username: str | None = None,
) -> int:
    """Upload local agent files to remote."""
    if framework not in FRAMEWORK_REGISTRY:
        return _fail(f"unknown framework '{framework}'. Available: {available_frameworks()}")

    local_name, err = resolve_local_name(name, framework, local_dir)
    if err:
        return _fail(err)

    spec = build_spec(framework, local_name, local_dir)
    root = spec.workspace_root
    resources: dict[str, bytes] = spec.collect_bytes()
    if not resources:
        display_name = local_name if local_name != GLOBAL_AGENT_NAME else "global"
        return _fail(
            f"no files found for {framework}/{display_name} under {root}. "
            f"Check the path or pass --local_dir."
        )

    total_bytes = sum(len(v) for v in resources.values())
    print(f"Found {len(resources)} file(s) ({total_bytes} bytes) under {root}:")
    for rel in sorted(resources):
        print(f"  {rel} ({len(resources[rel])} B)")

    if dry_run:
        print("\n[dry-run] nothing uploaded.")
        return 0

    if not endpoint or not token:
        return _fail("not logged in. Provide endpoint and token.")
    if not username:
        return _fail("missing username.")

    client = AgentApi(endpoint=endpoint, token=token)

    effective_name = local_name if local_name != GLOBAL_AGENT_NAME else None
    group, repo_n = resolve_remote(
        repo=repo, name=effective_name, framework=framework, username=username,
    )

    try:
        from ._sync import push_resources
        push_resources(client, group, repo_n, framework, resources)
    except APIError as e:
        return _fail(api_error_message(e, "upload"))
    except Exception as e:
        return _fail(f"upload failed: {e}")

    print(f"\nUploaded {len(resources)} file(s) to {group}/{repo_n}.")
    return 0


def cmd_download(
    framework: str,
    repo: str,
    name: str | None = None,
    target: str | None = None,
    local_dir=None,
    dry_run: bool = False,
    *,
    endpoint: str | None = None,
    token: str | None = None,
    username: str | None = None,
) -> int:
    """Download remote agent files to local.

    Token is optional for public repos.  However, when *repo* does not
    contain ``/`` we need *username* to derive the group, which requires
    authentication.
    """
    if not repo:
        return _fail("--repo is required for download (the remote repository name)")
    if framework not in FRAMEWORK_REGISTRY:
        return _fail(f"unknown framework '{framework}'. Available: {available_frameworks()}")

    if not endpoint:
        return _fail("not logged in. Provide endpoint.")

    # Token is optional for download (public repos don't require auth).
    # But if --repo doesn't contain '/', we need username to derive group.
    if '/' not in repo and not token:
        return _fail(
            f"--repo '{repo}' requires login to resolve owner. "
            f"Use 'owner/name' format or run 'ms login' first.")
    if not token:
        token = ''
    if not username:
        username = ''

    group, repo_n = resolve_remote(
        repo=repo, name=name, framework=framework, username=username,
    )

    client = AgentApi(endpoint=endpoint, token=token)
    try:
        info = client.repo_info(group, repo_n)
        if info is None:
            return _fail(f"repository {group}/{repo_n} not found.")
        paths = client.list_repo_files(group, repo_n)
        if not paths:
            return _fail(f"repository {group}/{repo_n} has no files.")
        resources = {p: client.download_repo_file(group, repo_n, p) for p in paths}
    except APIError as e:
        return _fail(api_error_message(e, "download"))
    except Exception as e:
        return _fail(f"download failed: {e}")

    # Optional format conversion.
    target_fw = target or framework
    if target_fw not in FRAMEWORK_REGISTRY:
        return _fail(f"unknown target framework '{target_fw}'. Available: {available_frameworks()}")

    local_name = name or DEFAULT_AGENT_NAME
    spec = build_spec(target_fw, local_name, local_dir)
    root = spec.workspace_root

    if target_fw != framework:
        if name == ALL_AGENT_NAME:
            # All-mode conversion only makes sense between two root-per-agent
            # frameworks (1:1 agent-directory mapping).  Other layouts (e.g.
            # file-per-agent qoder, single-agent) would collapse N agents into
            # a shared root, which is lossy and ambiguous -- reject explicitly.
            src_spec = build_spec(framework, local_name, local_dir)
            if not (src_spec.is_root_per_agent and spec.is_root_per_agent):
                return _fail(
                    "cross-framework conversion with --name all is only supported "
                    "between root-per-agent frameworks (e.g. qwenpaw <-> openclaw). "
                    "For other layouts, convert one agent at a time: "
                    "-n <agent> --target-framework <fw>.")
            resources = convert_resources(
                resources, framework, target_fw,
                all_mode=True, src_spec=src_spec, dst_spec=spec,
            )
        else:
            existing_files = set(spec.collect().keys())
            resources = convert_resources(resources, framework, target_fw, existing_files=existing_files)
        print(f"Converted {framework} -> {target_fw} ({len(resources)} file(s)).")

    patterns = spec.resolved_patterns()
    filtered = {k: v for k, v in resources.items() if spec.matches(k, patterns)}
    skipped = set(resources.keys()) - set(filtered.keys())
    if skipped:
        print(f"Skipped {len(skipped)} file(s) not matching workspace spec:")
        for s in sorted(skipped):
            print(f"  [skip] {s}")

    if not filtered:
        return _fail("no downloaded files match the local workspace spec patterns.")

    print(f"{len(filtered)} file(s) for {group}/{repo_n} (framework={target_fw}):")
    for rel in sorted(filtered):
        print(f"  {rel} -> {root / rel}")

    if dry_run:
        print("\n[dry-run] nothing written.")
        return 0

    written = spec.apply(filtered)
    print(f"\nWrote {len(written)} file(s) under {root}.")
    return 0


def _file_per_agent_identity_path(dst_spec: WorkspaceSpec) -> str | None:
    """Resolve the per-agent identity file for a file-per-agent target.

    File-per-agent frameworks (e.g. qoder) declare a ``{name}`` placeholder
    pattern such as ``agents/{name}.md``.  Format it with the destination
    agent name so converted persona content can be routed into that file.
    Returns ``None`` when the layout has no single ``{name}`` file pattern.
    """
    name = dst_spec.agent_name or DEFAULT_AGENT_NAME
    for pattern in dst_spec.patterns:
        # Only single-file placeholders (no wildcard) identify the persona file;
        # skip glob patterns like ``skills/{name}/*`` if any exist.
        if "{name}" in pattern and "*" not in pattern:
            return pattern.format(name=name)
    return None


def convert_workspace(
    src_spec: WorkspaceSpec,
    source_fw: str,
    target_fw: str,
    dst_spec: WorkspaceSpec,
    dry_run: bool = False,
) -> int:
    """Shared convert logic: merge -> filter defaults -> backup -> write.

    Returns 0 on success, 1 on failure.
    """
    src_root = src_spec.workspace_root
    resources = src_spec.collect()
    if not resources:
        return _fail(f"no {source_fw} files found under {src_root}.")

    existing = dst_spec.collect()
    existing_paths = set(existing.keys())

    if source_fw == target_fw:
        converted = resources
        default_paths: set = set()
    else:
        # File-per-agent targets (e.g. qoder ``agents/{name}.md``) keep
        # per-agent identity in a dedicated sub-agent file; route overflow
        # (persona content with no shared mapping) there instead of the
        # shared catch-all so it does not pollute other sub-agents.
        overflow_target = None
        if any("{name}" in p for p in dst_spec.patterns):
            overflow_target = _file_per_agent_identity_path(dst_spec)
        result = merge_resources(
            incoming=resources,
            source_product=source_fw,
            target_product=target_fw,
            source_defaults=get_defaults(source_fw),
            target_defaults=get_defaults(target_fw),
            overflow_target=overflow_target,
        )
        default_paths = {a.path for a in result.actions if a.action == "default"}
        converted = result.merged_files

    dst_root = dst_spec.workspace_root
    # Non-default files: always write (overwrite if exists).
    # Default files: write only if target does NOT already have them.
    effective = {
        k: v for k, v in converted.items()
        if k not in default_paths or k not in existing_paths
    }
    skipped_defaults = sorted(default_paths & existing_paths)
    added_defaults = sorted(default_paths - existing_paths)

    print(
        f"Convert {source_fw}/{src_spec.agent_name} ({src_root}) -> "
        f"{target_fw}/{dst_spec.agent_name} ({dst_root}): "
        f"{len(resources)} in, {len(effective)} out"
    )
    for rel in sorted(effective):
        print(f"  {rel} -> {dst_root / rel}")
    if skipped_defaults:
        print(f"  ({len(skipped_defaults)} existing default(s) preserved: "
              f"{', '.join(skipped_defaults)})")
    if added_defaults:
        print(f"  ({len(added_defaults)} default template(s) added: "
              f"{', '.join(added_defaults)})")

    if dry_run:
        print("\n[dry-run] nothing written.")
        return 0

    if not effective:
        print("\nNo effective files to write.")
        return 0

    # Backup existing target files before overwriting
    from ._sync import backup_local
    if existing:
        backup_path = backup_local(dst_spec, f"{target_fw}_{dst_spec.agent_name}")
        print(f"  Backup: {backup_path}")

    written = dst_spec.apply(effective)
    print(f"\nWrote {len(written)} file(s) under {dst_root}.")
    return 0


def cmd_convert(
    source_fw: str,
    target_fw: str,
    from_name: str | None = None,
    target_name: str | None = None,
    local_dir=None,
    out_dir=None,
    dry_run: bool = False,
) -> int:
    """Local-only format conversion: read a workspace, convert, write it out."""
    for fw, label in ((source_fw, "--from-framework"), (target_fw, "--target-framework")):
        if fw not in FRAMEWORK_REGISTRY:
            return _fail(f"unknown framework '{fw}' for {label}. Available: {available_frameworks()}")

    src_name = from_name or DEFAULT_AGENT_NAME
    dst_name = target_name or src_name
    src_spec = build_spec(source_fw, src_name, local_dir)
    dst_spec = build_spec(target_fw, dst_name, out_dir)
    return convert_workspace(src_spec, source_fw, target_fw, dst_spec, dry_run=dry_run)


def cmd_watch(
    framework: str,
    name: str | None = None,
    local_dir=None,
    repo: str | None = None,
    pull: bool = False,
    *,
    endpoint: str | None = None,
    token: str | None = None,
    username: str | None = None,
) -> int:
    """Start background bidirectional sync for agent files."""
    from ._cache import pid_file
    from ._watcher import daemonize, watch_loop

    if framework not in FRAMEWORK_REGISTRY:
        return _fail(f"unknown framework '{framework}'. Available: {available_frameworks()}")

    if name:
        local_name, err = resolve_local_name(name, framework, local_dir)
        if err:
            return _fail(err)
    else:
        local_name = ALL_AGENT_NAME

    if not endpoint or not token:
        return _fail("not logged in. Provide endpoint and token.")
    if not username:
        return _fail("missing username.")

    # Ensure no stale watch processes are running.
    pf = pid_file()
    from ._watcher import stop_daemon
    stop_daemon()

    spec = build_spec(framework, local_name, local_dir)
    client = AgentApi(endpoint=endpoint, token=token)

    # Guard: file-per-agent frameworks with a specific agent name.
    if (not spec.supports_individual_watch
            and local_name not in (GLOBAL_AGENT_NAME, ALL_AGENT_NAME, DEFAULT_AGENT_NAME)):
        return _fail(
            f"'{framework}' has shared files across sub-agents; "
            f"watch only supports global/default mode to avoid sync conflicts. "
            f"Use upload/download -n {local_name} for individual sub-agent operations."
        )

    # Resolve remote target.
    effective_name = name if name else None
    group, repo_n = resolve_remote(
        repo=repo, name=effective_name, framework=framework, username=username,
    )

    # Guard: check remote repo framework matches local.
    try:
        info = client.repo_info(group, repo_n)
        if info:
            remote_fw = info.get("Framework", "")
            if remote_fw and remote_fw != framework:
                return _fail(
                    f"framework mismatch: local={framework}, remote={remote_fw}. "
                    f"Use convert or download --target for cross-framework sync."
                )
    except APIError as e:
        if e.status_code in (403, 401):
            return _fail(api_error_message(e, "watch"))
        elif e.status_code == 404:
            pass  # repo not found — first push will create it
        else:
            return _fail(f"failed to get repository info (HTTP {e.status_code}: {e.message})")
    except Exception as e:
        return _fail(f"failed to get repository info: {e}")

    interval = 60
    push_only = not pull
    print(f"Starting sync for {group}/{repo_n} (interval={interval}s)...")
    print(f"  Framework: {framework}")
    print(f"  Root: {spec.workspace_root}")
    if push_only:
        print("  Mode: push-only (local -> remote, will NOT pull remote changes)")
    else:
        print("  Mode: bidirectional (local <-> remote, WILL pull remote changes)")
    print(f"  Stop: ms agent stop")

    daemonize(watch_loop, spec, client, group, repo_n, framework, interval, push_only=push_only)
    from ._cache import log_file
    print(f"  Watch started (PID file: {pf}).")
    print(f"  Log: {log_file()}")
    return 0


def cmd_stop() -> int:
    """Stop the background watch process."""
    from ._watcher import stop_daemon

    stopped = stop_daemon()
    if stopped:
        print("Watch process stopped.")
    else:
        print("No watch process running.")
    return 0


_BACKUP_WRAPPER = "agent/"


def _strip_backup_wrapper(filename: str) -> str:
    """Strip the legacy ``agent/`` wrapper dir from a backup zip entry.

    Older backups stored files under an ``agent/`` prefix; new backups don't.
    Stripping keeps restore aligned with the workspace-relative layout used by
    ``collect``/``apply`` regardless of which format the zip uses.
    """
    if filename.startswith(_BACKUP_WRAPPER):
        return filename[len(_BACKUP_WRAPPER):]
    return filename


def _parse_backup_meta(stem: str) -> tuple[str, str]:
    """Parse a backup filename stem into ``(framework, name)``.

    Filenames look like ``{fw}{delim}{name}_{date}_{time}``; the leading
    ``{fw}{delim}{name}`` prefix is split on ``_`` (watch backups) or ``-``
    (upload backups).
    """
    parts = stem.rsplit("_", 2)
    prefix = parts[0] if len(parts) >= 3 else stem
    delim = "_" if "_" in prefix else "-"
    fw, _, nm = prefix.partition(delim)
    return fw, nm


def _filter_backups(backups, fw_filter, name_filter):
    """Filter backup files by framework/name parsed from their filenames."""
    if not (fw_filter or name_filter):
        return backups
    out = []
    for f in backups:
        fw, nm = _parse_backup_meta(f.stem)
        if fw_filter and fw != fw_filter:
            continue
        if name_filter and nm != name_filter:
            continue
        out.append(f)
    return out


def cmd_recover(
    target: str | None = None,
    framework: str | None = None,
    name: str | None = None,
    local_dir=None,
    list_backups: bool = False,
) -> int:
    """Restore agent files from a backup zip."""
    import datetime as _dt
    from ._cache import cache_dir

    cdir = cache_dir()

    backups = sorted(
        (f for f in cdir.iterdir() if f.suffix == ".zip" and f.is_file()),
        key=lambda f: f.stat().st_mtime,
    )

    # --list mode
    if list_backups:
        backups = _filter_backups(backups, framework, name)

        if not backups:
            print("No backups found.")
            return 0
        print(f"Backups in {cdir}:\n")
        last = backups[-1]
        for f in backups:
            mtime = _dt.datetime.fromtimestamp(f.stat().st_mtime)
            marker = "  [LAST]" if f == last else ""
            print(f"  {f.name}  ({mtime:%Y-%m-%d %H:%M:%S}){marker}")
        print(f"\n{len(backups)} backup(s) total.")
        return 0

    # Restore mode
    if not target:
        return _fail("specify a target: 'last' or a backup filename. Use --list to see available backups.")

    backups = _filter_backups(backups, framework, name)

    if target == "last":
        if not backups:
            return _fail("no backups found.")
        zip_path = backups[-1]
    else:
        fname = target if target.endswith(".zip") else f"{target}.zip"
        zip_path = cdir / fname
        if not zip_path.exists():
            zip_path = Path(target)
        if not zip_path.exists():
            return _fail(f"backup not found: {fname} (looked in {cdir})")

    if framework and framework not in FRAMEWORK_REGISTRY:
        return _fail(f"unknown framework '{framework}'. Available: {available_frameworks()}")

    # Parse (framework, name) from the zip filename once, honoring both the
    # ``-`` (upload) and ``_`` (watch) delimiters, and fill in whatever the
    # caller left unspecified.  Reusing _parse_backup_meta keeps this aligned
    # with the --list/restore filtering above.
    parsed_fw, parsed_name = _parse_backup_meta(zip_path.stem)
    if not framework:
        if parsed_fw in FRAMEWORK_REGISTRY:
            framework = parsed_fw
        else:
            return _fail("cannot infer framework. Pass --framework explicitly.")

    # Determine the restore SCOPE (which agent directory the backup belongs to).
    # An all-scope backup is named ``{fw}_{date}_{time}`` (no name segment), so
    # ``parsed_name`` is empty -> restore into the all-root.  A single-agent
    # backup is ``{fw}{delim}{name}_...`` -> restore into that agent only.
    # A hardcoded "all" here would lift root-per-agent frameworks to the shared
    # workspaces/ parent and, because a single-agent zip stores bare (unprefixed)
    # paths, wrongly treat every sibling agent's files as "extra" and delete them.
    restore_name = name or parsed_name or ALL_AGENT_NAME
    if not name:
        name = parsed_name or parsed_fw

    spec = build_spec(framework, restore_name, local_dir)
    root = spec.workspace_root

    # Backup current local files
    from ._sync import backup_local
    current_resources = spec.collect()
    if current_resources:
        pre_restore_backup = backup_local(spec, name)
        print(f"Pre-restore backup: {pre_restore_backup.name}")
    else:
        print("No existing files to backup.")

    # Determine which files are in the zip (strip legacy wrapper prefix).
    with zipfile.ZipFile(zip_path, "r") as zf:
        zip_entries = {
            _strip_backup_wrapper(info.filename)
            for info in zf.infolist() if not info.is_dir()
        }

    # Delete local files not in the zip
    deleted = 0
    for rel in sorted(current_resources.keys()):
        if rel not in zip_entries:
            target_file = root / rel
            if target_file.exists():
                target_file.unlink()
                print(f"  Removed: {rel}")
                deleted += 1

    # Extract zip
    resolved_root = root.resolve()
    print(f"Restoring {zip_path.name} -> {resolved_root}")
    restored = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = _strip_backup_wrapper(info.filename)
            file_target = (resolved_root / rel).resolve()
            if not file_target.is_relative_to(resolved_root):
                print(f"  Skipped (path traversal): {info.filename}")
                continue
            file_target.parent.mkdir(parents=True, exist_ok=True)
            file_target.write_bytes(zf.read(info.filename))
            print(f"  Restored: {rel}")
            restored += 1

    print(f"\nRestored {restored} file(s), removed {deleted} extra file(s).")
    return 0


# Aliases: cmd_restore = cmd_recover, cmd_backups extracted from list_backups mode.

def cmd_restore(
    target: str | None = None,
    framework: str | None = None,
    name: str | None = None,
    local_dir=None,
) -> int:
    """Restore agent files from a backup zip (alias for cmd_recover without list mode)."""
    return cmd_recover(target=target, framework=framework, name=name, local_dir=local_dir, list_backups=False)


def cmd_backups(
    framework: str | None = None,
    name: str | None = None,
    local_dir=None,
) -> int:
    """List available backups."""
    return cmd_recover(target=None, framework=framework, name=name, local_dir=local_dir, list_backups=True)


def cmd_list(
    owner: str | None = None,
    page_number: int = 1,
    page_size: int = 10,
    *,
    endpoint: str | None = None,
    token: str | None = None,
) -> int:
    """List remote agent repositories."""
    if not endpoint:
        return _fail("not logged in. Provide endpoint.")
    if not token:
        token = ''

    client = AgentApi(endpoint=endpoint, token=token)
    try:
        result = client.list_agents(owner=owner, page_number=page_number, page_size=page_size)
    except APIError as e:
        return _fail(api_error_message(e, "list"))
    except Exception as e:
        return _fail(f"list failed: {e}")

    items = result.get("items") or []
    total = result.get("total_count", len(items))

    if not items:
        print("(no agent repositories found)")
        return 0

    headers = ['repo_id', 'framework', 'visibility', 'updated']
    rows = []
    for item in items:
        owner_name = item.get('Path') or item.get('path') or ''
        repo_name = item.get('Name') or item.get('name') or ''
        repo_id = f'{owner_name}/{repo_name}' if owner_name else repo_name
        fw = item.get('Framework') or item.get('framework') or '-'
        vis = item.get('Visibility') or item.get('visibility') or '-'
        updated = item.get('LastUpdatedDate') or item.get('last_updated_date') or '-'
        if isinstance(updated, str) and 'T' in updated:
            updated = updated.split('T')[0]
        rows.append((repo_id, fw, vis, updated))

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    fmt = '  '.join(f'{{:<{w}}}' for w in col_widths)
    print(fmt.format(*headers))
    print(fmt.format(*['-' * w for w in col_widths]))
    for row in rows:
        print(fmt.format(*[str(v) for v in row]))

    print(f'\npage {page_number} / total {total} (page_size={page_size})')
    return 0
