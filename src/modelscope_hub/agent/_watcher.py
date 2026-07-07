# Copyright (c) Alibaba, Inc. and its affiliates.
"""File watcher (polling) and daemon management for agent sync."""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from logging.handlers import RotatingFileHandler

from ..utils.logger import get_logger
from ._cache import load_sync_state, log_file, pid_file, save_sync_state, stop_file
from ..errors import APIError
from ._sync import (
    backup_local,
    detect_local_changes,
    pull_incremental,
    push_incremental,
    push_resources,
)

__all__ = ["watch_loop", "daemonize", "stop_daemon"]

_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    _logger = get_logger("agent.watch")
    _logger.setLevel(logging.INFO)
    fh = RotatingFileHandler(
        str(log_file()), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    _logger.addHandler(fh)
    return _logger


def watch_loop(spec, client, username: str, repo: str, framework: str, interval: int = 120, *, push_only: bool = True):
    """Sync loop: push local changes, optionally pull remote changes.

    Args:
        repo: Remote repository name (used as the API path component).
        push_only: True (default) = only pushes, never modifies local files.
                   False = full bidirectional sync (remote wins on conflict).
    """
    logger = _get_logger()

    # After double-fork, the parent's requests.Session connection pool holds
    # stale file descriptors that cause EBADF on new connections.  Rebuild the
    # client so the daemon starts with a fresh session.
    from ._api import AgentApi
    client = AgentApi(endpoint=client.server, token=client.token, timeout=client.timeout)

    logger.info("Watch started for %s/%s (root=%s, interval=%ds, push_only=%s)",
                username, repo, spec.workspace_root, interval, push_only)

    state = load_sync_state(repo)
    running = True
    stop_event = threading.Event()
    sf = stop_file()

    def _handle_term(signum, frame):
        nonlocal running
        running = False
        stop_event.set()

    if threading.current_thread() is threading.main_thread():
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _handle_term)
        signal.signal(signal.SIGINT, _handle_term)

    sf.unlink(missing_ok=True)

    while running:
        elapsed = 0
        poll_interval = min(interval, 5)
        while elapsed < interval and running:
            stop_event.wait(timeout=poll_interval)
            if stop_event.is_set():
                running = False
                break
            if sf.exists():
                running = False
                break
            elapsed += poll_interval
        if not running:
            break

        # ---- Fetch remote file list ----
        try:
            remote_files = client.list_repo_files_detail(username, repo)
        except APIError as e:
            if e.status_code in (404, 500):
                remote_files = []
            else:
                logger.error("Failed to list remote files: %s", e)
                continue

        # ---- Collect local resources & detect changes ----
        local_resources = spec.collect_bytes()
        scope = set(local_resources.keys()) | set(state.get("remote_files", {}).keys())
        remote_sha_map = {f.path: f.sha256 for f in remote_files if f.path in scope}

        remote_changed = (
            remote_sha_map != state.get("remote_files", {})
        )
        local_changed = bool(detect_local_changes(local_resources, state["remote_files"]))

        # ---- Sync decision ----
        did_sync = False
        try:
            did_sync = _sync_action(
                push_only, remote_changed, local_changed,
                client, username, repo, framework, spec,
                remote_files, local_resources, logger,
                state,
            )
        except Exception as exc:
            logger.error("Sync failed (will retry): %s", exc)

        # ---- Update baseline on successful sync ----
        if did_sync:
            if not push_only:
                local_resources = spec.collect_bytes()
            _refresh_baseline(client, username, repo, local_resources, state, logger)
            save_sync_state(repo, state["last_commit_date"], state["remote_files"])

    logger.info("Watch stopped (signal received).")
    pf = pid_file()
    if pf.exists():
        try:
            if pf.read_text(encoding="utf-8").strip() == str(os.getpid()):
                pf.unlink(missing_ok=True)
        except Exception:
            pass
    sf.unlink(missing_ok=True)


def _push_local(client, username, name, framework, local_resources, state, logger, *, remote_paths=None, remote_lfs_paths=None) -> bool:
    """Push local changes: full upload on first time, incremental thereafter."""
    if not local_resources:
        logger.debug("No local resources to push -- skipping.")
        return False
    if not state.get("remote_files"):
        push_resources(client, username, name, framework, local_resources)
        logger.info("Pushed local changes (full upload -- first time).")
        return True
    else:
        changed = detect_local_changes(local_resources, state["remote_files"])
        if changed:
            # Filter stale DELETEs: only delete files that actually exist on
            # the remote.  The baseline may be stale if the remote was
            # modified outside watch (e.g. via upload or manual deletion).
            if remote_paths is not None:
                stale = {
                    p for p, c in changed.items()
                    if c is None and p not in remote_paths
                }
                for p in sorted(stale):
                    logger.warning("  SKIP DELETE: %s (not on remote, stale baseline)", p)
                    del changed[p]
            if not changed:
                logger.info("No real changes to push after filtering stale deletes.")
                return False
            # Use actual remote paths (not stale baseline) for CREATE vs UPDATE.
            actual = remote_paths if remote_paths is not None else set(state["remote_files"].keys())
            push_incremental(client, username, name, changed, actual,
                             remote_lfs_paths=remote_lfs_paths)
            logger.info("Pushed local changes (incremental commit).")
            return True
        return False


def _sync_action(
    push_only, remote_changed, local_changed,
    client, username, name, framework, spec,
    remote_files, local_resources, logger,
    state,
) -> bool:
    """Execute the appropriate sync action. Returns True if something changed."""
    # Backup naming convention: {framework}_{agent_name} so that
    # ``cmd_recover --framework`` can filter watch-created backups.
    backup_label = f"{framework}_{spec.agent_name}"
    remote_paths = {f.path for f in remote_files}
    remote_lfs_paths = {f.path for f in remote_files if getattr(f, 'is_lfs', False)}

    if push_only:
        if not local_changed:
            return False
        return _push_local(client, username, name, framework, local_resources, state, logger,
                           remote_paths=remote_paths, remote_lfs_paths=remote_lfs_paths)

    if remote_changed and local_changed:
        backup_path = backup_local(spec, backup_label)
        pull_incremental(client, username, name, spec, remote_files, local_resources)
        logger.warning("Conflict: remote wins. Local backup: %s", backup_path)
    elif remote_changed:
        backup_path = backup_local(spec, backup_label)
        pull_incremental(client, username, name, spec, remote_files, local_resources)
        logger.info("Pulled remote changes (backup: %s).", backup_path)
    elif local_changed:
        _push_local(client, username, name, framework, local_resources, state, logger,
                    remote_paths=remote_paths, remote_lfs_paths=remote_lfs_paths)
    else:
        return False
    return True


def _refresh_baseline(client, username: str, name: str, local_resources: dict, state: dict, logger) -> None:
    """Re-fetch remote file list and update state in-place."""
    managed = set(local_resources.keys())
    for attempt in range(3):
        try:
            fresh = client.list_repo_files_detail(username, name)
            state["last_commit_date"] = max((f.committed_date for f in fresh), default=0)
            state["remote_files"] = {f.path: f.sha256 for f in fresh if f.path in managed}
            return
        except APIError as e:
            if e.status_code == 500 and attempt < 2:
                time.sleep(3)
                continue
            logger.error("Failed to refresh baseline: %s", e)
            return
        except Exception as exc:
            logger.error("Failed to refresh baseline: %s", exc)
            return


def daemonize(target, *args, **kwargs):
    """Launch *target* as a background process.

    Unix: classic double-fork.
    Windows: subprocess.Popen with DETACHED_PROCESS.
    """
    if hasattr(os, "fork"):
        _daemonize_unix(target, *args, **kwargs)
    else:
        _daemonize_windows(target, *args, **kwargs)


def _daemonize_unix(target, *args, **kwargs):
    """Double-fork daemon (Unix only)."""
    pf = pid_file()

    pid = os.fork()
    if pid > 0:
        return

    os.setsid()

    pid = os.fork()
    if pid > 0:
        os._exit(0)

    pf.write_text(str(os.getpid()), encoding="utf-8")

    sys.stdout.flush()
    sys.stderr.flush()
    with open(os.devnull, "r") as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())
    log_fd = os.open(str(log_file()), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, sys.stdout.fileno())
    os.dup2(log_fd, sys.stderr.fileno())
    os.close(log_fd)

    try:
        target(*args, **kwargs)
    finally:
        try:
            if pf.exists() and pf.read_text(encoding="utf-8").strip() == str(os.getpid()):
                pf.unlink(missing_ok=True)
        except Exception:
            pass
        os._exit(0)


def _daemonize_windows(target, *args, **kwargs):
    """Spawn a detached background process (Windows)."""
    import json
    import tempfile

    spec_obj = args[0] if len(args) > 0 else None
    client_obj = args[1] if len(args) > 1 else None
    payload = {
        "username": args[2] if len(args) > 2 else kwargs.get("username", ""),
        "repo": args[3] if len(args) > 3 else kwargs.get("repo", ""),
        "framework": args[4] if len(args) > 4 else kwargs.get("framework", ""),
        "interval": args[5] if len(args) > 5 else kwargs.get("interval", 120),
        "push_only": kwargs.get("push_only", True),
        "local_name": getattr(spec_obj, "agent_name", "") if spec_obj else "",
        "server": getattr(client_obj, "server", "") if client_obj else "",
        "token": getattr(client_obj, "token", "") if client_obj else "",
    }
    fd, param_path = tempfile.mkstemp(suffix=".json", prefix="ms_agent_watch_")
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f)

    CREATE_NO_WINDOW = 0x08000000
    DETACHED_PROCESS = 0x00000008
    proc = subprocess.Popen(
        [sys.executable, "-m", "modelscope_hub.agent._watcher", "_daemon", param_path],
        creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW,
        close_fds=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    pf = pid_file()
    pf.write_text(str(proc.pid), encoding="utf-8")


_DEFAULT_WATCH_PATTERNS = [
    "agent watch",
    "ms agent watch",
    "modelscope agent watch",
]


def stop_daemon(extra_patterns: list[str] | None = None) -> bool:
    """Stop ALL running watch daemon processes (cross-platform).

    Primary mechanism: write a stop-file that the watch loop polls.
    Secondary: send SIGTERM (Unix) or taskkill (Windows) as a backup.
    """
    stopped = False
    pf = pid_file()
    sf = stop_file()

    sf.write_text("stop", encoding="utf-8")

    tracked_pid = None
    if pf.exists():
        try:
            tracked_pid = int(pf.read_text().strip())
            if hasattr(os, "fork"):
                os.kill(tracked_pid, signal.SIGTERM)
            stopped = True
        except (ValueError, OSError, ProcessLookupError):
            tracked_pid = None

    if hasattr(os, "fork"):
        my_pid = os.getpid()
        for found_pid in _find_watch_pids(extra_patterns):
            if found_pid in (my_pid, tracked_pid):
                continue
            try:
                os.kill(found_pid, signal.SIGTERM)
                stopped = True
            except (ProcessLookupError, PermissionError):
                pass
    else:
        for found_pid in _find_watch_pids_windows(extra_patterns):
            if found_pid == tracked_pid:
                continue
            _terminate_pid_windows(found_pid)
            stopped = True

    if stopped or tracked_pid:
        _wait_for_exit(tracked_pid, timeout=8)

    if tracked_pid and _is_alive(tracked_pid):
        _force_kill(tracked_pid)

    pf.unlink(missing_ok=True)
    sf.unlink(missing_ok=True)

    return stopped or tracked_pid is not None


def _find_watch_pids(extra_patterns: list[str] | None = None) -> list[int]:
    """Find PIDs of running watch daemon processes via pgrep (Unix only)."""
    patterns = list(dict.fromkeys(_DEFAULT_WATCH_PATTERNS + (extra_patterns or [])))
    pids: set = set()
    for pattern in patterns:
        try:
            result = subprocess.run(
                ["pgrep", "-f", "--", pattern],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                for p in result.stdout.strip().split("\n"):
                    if p.strip().isdigit():
                        pids.add(int(p.strip()))
        except (OSError, subprocess.TimeoutExpired, ValueError):
            pass
    return list(pids)


def _find_watch_pids_windows(extra_patterns: list[str] | None = None) -> list[int]:
    """Find PIDs of running watch daemon processes on Windows via wmic."""
    patterns = list(dict.fromkeys(_DEFAULT_WATCH_PATTERNS + (extra_patterns or [])))
    pids: set = set()
    try:
        result = subprocess.run(
            ["wmic", "process", "where", "name like '%python%'",
             "get", "processid,commandline"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        for line in result.stdout.splitlines():
            line_lower = line.lower()
            for pattern in patterns:
                if pattern.lower() in line_lower:
                    parts = line.strip().split()
                    if parts and parts[-1].isdigit():
                        pids.add(int(parts[-1]))
                    break
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return list(pids)


def _terminate_pid_windows(pid: int) -> None:
    """Terminate a process on Windows using taskkill."""
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F"],
            capture_output=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def _is_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _wait_for_exit(pid: int | None, timeout: int = 8) -> None:
    """Wait up to *timeout* seconds for a process to exit."""
    if pid is None:
        time.sleep(2)
        return
    for _ in range(timeout * 2):
        if not _is_alive(pid):
            return
        time.sleep(0.5)


def _force_kill(pid: int) -> None:
    """Force-kill a process (SIGKILL on Unix, taskkill /F on Windows)."""
    if hasattr(os, "fork"):
        try:
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except (ProcessLookupError, PermissionError, OSError):
            pass
    else:
        _terminate_pid_windows(pid)


# ---------------------------------------------------------------------------
# Windows daemon entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    if len(sys.argv) >= 3 and sys.argv[1] == "_daemon":
        param_path = sys.argv[2]
        with open(param_path, "r") as _f:
            _params = json.load(_f)
        os.unlink(param_path)

        from ._api import AgentApi
        from ._workspace import FRAMEWORK_REGISTRY
        from . import frameworks as _  # noqa: F401 — trigger registration

        _fw = _params["framework"]
        _spec_cls = FRAMEWORK_REGISTRY[_fw]
        _spec = _spec_cls(agent_name=_params.get("local_name", "all"))
        _client = AgentApi(endpoint=_params["server"], token=_params["token"])

        _pf = pid_file()
        _pf.write_text(str(os.getpid()), encoding="utf-8")

        watch_loop(
            _spec, _client,
            username=_params["username"],
            repo=_params["repo"],
            framework=_fw,
            interval=_params.get("interval", 120),
            push_only=_params.get("push_only", True),
        )
