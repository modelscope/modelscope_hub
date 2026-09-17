# Copyright (c) Alibaba, Inc. and its affiliates.
"""Locate, fetch, verify and run an agent plugin.

``ms agent install`` resolves which plugin to use, downloads it from a model
repository, verifies it, and hands the agent id to the plugin's entry point. No
framework knowledge lives here: how an agent is registered and what its workspace
looks like are the plugin's decisions. The one exception is the destination
directory, which this module resolves for a plugin that only transports bytes --
it cannot know where such a plugin should write, and the plugin deliberately has
no default of its own.

Integrity comes from ``plugin.json``'s ``content_sha256``, not from the hub's own
file listing -- that listing has been observed reporting a git blob SHA-1 in a
``sha256`` field.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import constants
from ..errors import InvalidParameter, NotSupportedError
from ..utils.file_utils import get_cache_dir

MANIFEST_NAME = "plugin.json"

#: Negotiated rather than hard-coded, so a plugin growing a richer entry point
#: does not require re-releasing the hub. Order is preference: a plugin that
#: implements ``install`` owns placement, registration and completion, so it
#: wins. ``fetch_raw`` is a transport that writes the repository's bytes into a
#: directory the caller names and never touches a framework workspace, which is
#: what keeps a user's own credentials intact. ``download`` is the 0.1.x name
#: for an operation that did install into the workspace.
ENTRY_OPERATIONS: tuple[str, ...] = ("install", "fetch_raw", "download")

#: Staging root for a fetch-only plugin, relative to ``MODELSCOPE_CACHE``.
#: Matches the plugin's own ``staging_dir()`` so one convention covers both
#: sides and there are not two places agent files can land.
AGENT_STAGING_SUBDIR: tuple[str, ...] = ("agent", "agent-staging")


@dataclass(frozen=True, slots=True)
class PluginSpec:
    """A plugin that has been downloaded and verified, but not yet imported."""

    repo_id: str
    owner: str
    name: str
    revision: str
    directory: Path
    manifest: dict[str, Any]
    entry_module: str

    @property
    def version(self) -> str:
        return str(self.manifest.get("version", "unknown"))

    def describe(self) -> str:
        frameworks = self.manifest.get("frameworks") or []
        operations = self.manifest.get("api") or []
        digest = _manifest_digest(self.manifest)
        return (
            f"  plugin     : {self.repo_id}\n"
            f"  revision   : {self.revision}\n"
            f"  version    : {self.version}\n"
            f"  entry      : {self.entry_module}\n"
            f"  frameworks : {', '.join(map(str, frameworks)) or '-'}\n"
            f"  operations : {', '.join(map(str, operations)) or '-'}\n"
            f"  directory  : {self.directory}\n"
            f"  manifest   : {len(self.manifest.get('content_sha256') or {})} file(s), "
            f"sha256 {digest}"
        )


@dataclass(frozen=True, slots=True)
class InstallOutcome:
    ok: bool
    error: str | None = None
    operation: str | None = None
    plugin: PluginSpec | None = None
    result: Any = None
    exit_code: int = 0


def _manifest_digest(manifest: dict[str, Any]) -> str:
    entries = manifest.get("content_sha256") or {}
    if not isinstance(entries, dict) or not entries:
        return "unavailable"
    blob = "\n".join(f"{k}:{entries[k]}" for k in sorted(entries))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _split_repo_id(repo_id: str, *, label: str) -> tuple[str, str]:
    """Split ``owner/name`` or raise, rejecting empty halves.

    ``"/" in repo_id`` is not enough: ``/name`` and ``owner/`` both contain a
    slash but name no repository, and letting either through costs a real request
    for somebody else's path.
    """
    owner, _, name = repo_id.partition("/")
    if not owner or not name:
        raise InvalidParameter(f"{label} {repo_id!r} must be in 'owner/name' form.")
    return owner, name


def resolve_plugin_repo(explicit: str | None = None) -> str:
    """Return the plugin repository id, or raise if none was configured.

    Resolution is the argument, then
    :data:`~modelscope_hub.constants.ENV_AGENT_PLUGIN_REPO`, and stops there.
    There is deliberately no built-in default owner: who publishes the plugin is a
    deployment decision, and a silent fallback would let a typo install from
    somewhere nobody chose.
    """
    repo_id = (explicit or "").strip()
    if not repo_id:
        repo_id = (os.environ.get(constants.ENV_AGENT_PLUGIN_REPO) or "").strip()
    if not repo_id:
        error = InvalidParameter(
            "no agent plugin repository configured. Pass --plugin-repo owner/name, "
            f"or set {constants.ENV_AGENT_PLUGIN_REPO}=owner/name."
        )
        error.suggestion = (
            "The plugin is published as a ModelScope model repository. Its owner is a "
            "deployment choice, so modelscope-hub does not assume one."
        )
        raise error
    _split_repo_id(repo_id, label="agent plugin repository")
    return repo_id


def assert_trusted_owner(repo_id: str) -> tuple[str, str]:
    """Split *repo_id* and require its owner on the allow-list.

    Comparison is case-sensitive: owners are identifiers, so normalising case
    would let ``mushenl`` pass a list that only trusts ``mushenL``.
    """
    owner, name = _split_repo_id(repo_id, label="agent plugin repository")
    trusted = constants.AGENT_PLUGIN_TRUSTED_OWNERS
    if owner not in trusted:
        error = InvalidParameter(
            f"owner {owner!r} is not allowed to provide the agent plugin. "
            f"Trusted owners: {', '.join(sorted(trusted)) or '(none)'}."
        )
        error.suggestion = (
            f"Extend the allow-list with {constants.ENV_AGENT_PLUGIN_TRUSTED_OWNERS}"
            "=owner1,owner2 (comma-separated, case-sensitive), then retry."
        )
        raise error
    return owner, name


def fetch_plugin(
    repo_id: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    endpoint: str | None = None,
    cache_dir: str | None = None,
) -> Path:
    """Download the plugin package and return its directory.

    Transfer executes nothing, so fetching an untrusted plugin is safe; only the
    import is gated, by :func:`require_trust`.
    """
    from ..compat import snapshot_download

    rev = revision or constants.DEFAULT_AGENT_PLUGIN_REVISION
    try:
        directory = snapshot_download(
            repo_id,
            repo_type="model",
            revision=rev,
            token=token,
            endpoint=endpoint,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        # ``snapshot_download`` re-raises hub errors as
        # ``requests.exceptions.HTTPError``, so the original type is not a
        # reliable discriminator; keep the cause chain instead.
        raise NotSupportedError(f"failed to download agent plugin {repo_id}@{rev}: {exc}") from exc
    path = Path(directory)
    if not path.is_dir():
        raise NotSupportedError(f"agent plugin {repo_id}@{rev} did not resolve to a directory: {path}")
    return path


#: Present in a downloaded plugin directory but not plugin content, so their
#: absence from ``content_sha256`` is expected. ``.gitattributes`` is injected by
#: the hub for LFS tracking and therefore never appears in an author's manifest.
NOT_PLUGIN_CONTENT: frozenset[str] = frozenset({MANIFEST_NAME, ".gitattributes"})


def verify_manifest(directory: Path, repo_id: str) -> dict[str, Any]:
    """Check the downloaded package against its own ``plugin.json``.

    Strict on purpose: this is what makes the subsequent import something other
    than unconditional code execution.
    """
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        raise NotSupportedError(f"{repo_id} is not an agent plugin: no {MANIFEST_NAME} at the repository root.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NotSupportedError(f"{repo_id}: cannot read {MANIFEST_NAME}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise NotSupportedError(f"{repo_id}: {MANIFEST_NAME} must be a JSON object.")

    entry_module = manifest.get("entry_module")
    if not isinstance(entry_module, str) or not entry_module:
        raise NotSupportedError(f"{repo_id}: {MANIFEST_NAME} has no 'entry_module'; cannot know what to import.")

    recorded = manifest.get("content_sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise NotSupportedError(
            f"{repo_id}: {MANIFEST_NAME} has no 'content_sha256', so the plugin's integrity "
            "cannot be verified. Refusing to load it."
        )

    missing, mismatched, unexpected = [], [], []
    for rel, expected in sorted(recorded.items()):
        target = directory / rel
        if not target.is_file():
            missing.append(rel)
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            mismatched.append(rel)
    recorded_set = set(recorded)
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        relative = path.relative_to(directory)
        rel = relative.as_posix()
        if rel in NOT_PLUGIN_CONTENT or "__pycache__" in relative.parts:
            continue
        if rel not in recorded_set:
            unexpected.append(rel)

    problems = []
    if missing:
        problems.append(f"missing {len(missing)} file(s): {', '.join(missing[:5])}")
    if mismatched:
        problems.append(f"sha256 mismatch for {len(mismatched)} file(s): {', '.join(mismatched[:5])}")
    if unexpected:
        problems.append(f"not listed in the manifest: {', '.join(unexpected[:5])}")
    if problems:
        raise NotSupportedError(f"{repo_id}: plugin integrity check failed -- " + "; ".join(problems))
    return manifest


def require_trust(spec: PluginSpec, *, trust_remote_code: bool) -> None:
    """Refuse to import the plugin unless execution was opted into.

    The refusal lists what *would* run so the decision is informed. The opt-in is
    a flag or an environment variable and is never persisted -- "allow this code
    to run" is not a preference worth remembering on the user's behalf.
    """
    if trust_remote_code or constants.AGENT_TRUST_REMOTE_CODE:
        return
    raise NotSupportedError(
        "refusing to execute plugin code without an explicit opt-in. The plugin "
        "resolved to:\n" + spec.describe() + "\n\n"
        "Re-run with --trust-remote-code to import and run it, or set "
        f"{constants.ENV_AGENT_TRUST_REMOTE_CODE}=1."
    )


def load_plugin(spec: PluginSpec) -> Any:
    """Import the plugin's entry module from its downloaded directory.

    The directory goes at the *front* of ``sys.path`` so the fetched revision
    wins over any same-named installed distribution.
    """
    root = str(spec.directory)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        return importlib.import_module(spec.entry_module)
    except ImportError:
        if root in sys.path:
            sys.path.remove(root)
        raise


def select_operation(module: Any) -> tuple[str, Any]:
    """Pick the entry operation the plugin actually supports.

    ``capabilities()`` is authoritative when present, so a plugin that ships a
    name without implementing it is not selected.
    """
    declared = None
    capabilities = getattr(module, "capabilities", None)
    if callable(capabilities):
        try:
            payload = capabilities()
            declared = set((payload or {}).get("operations") or ())
        except Exception:
            declared = None

    for name in ENTRY_OPERATIONS:
        func = getattr(module, name, None)
        if not callable(func):
            continue
        if declared is not None and name not in declared:
            continue
        return name, func

    available = ", ".join(sorted(declared)) if declared else "none reported"
    raise NotSupportedError(
        f"plugin {module.__name__} exposes none of {', '.join(ENTRY_OPERATIONS)}. Its capabilities are: {available}."
    )


def _accepted_kwargs(func: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    """Narrow *candidates* to what *func* accepts.

    A plugin's entry signature is its own contract and may grow keywords the hub
    knows nothing about; filtering keeps older hubs working instead of raising
    ``TypeError``. A function taking ``**kwargs`` gets everything.
    """
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return dict(candidates)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(candidates)
    return {key: value for key, value in candidates.items() if key in parameters}


def default_staging_dir(repo: str) -> Path:
    """Where a fetch-only plugin's files land when the caller named no directory.

    The staging directory itself is not created -- the plugin makes it when it
    writes, so an operation that ignores ``dest`` leaves no empty directory
    behind. Resolving the path does create the SDK cache root, as any download
    would. The repository id contains a slash and so is flattened; the timestamp
    keeps repeated fetches of one repository apart, to one-second resolution.
    """
    slug = repo.replace("/", "--")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return get_cache_dir().joinpath(*AGENT_STAGING_SUBDIR, f"{slug}-{stamp}")


def install_agent(
    repo: str,
    *,
    name: str | None = None,
    framework: str | None = None,
    local_dir: str | None = None,
    dry_run: bool = False,
    yes: bool = False,
    force: bool = False,
    quiet: bool = False,
    plugin_repo: str | None = None,
    plugin_revision: str | None = None,
    trust_remote_code: bool = False,
    endpoint: str | None = None,
    token: str | None = None,
    cache_dir: str | None = None,
) -> InstallOutcome:
    """Fetch or install *repo*'s agent through its framework plugin.

    Which of the two happens is the plugin's answer, not this function's: the
    entry operation is negotiated in :func:`select_operation`, so a plugin that
    installs into the workspace installs, and one that only transports bytes
    stages them in ``local_dir`` (or :func:`default_staging_dir`) for the install
    layer to place.

    *repo* is passed through uninterpreted beyond requiring ``owner/name``.
    Plugin failures come back as data (``ok`` False); the three gates
    (:func:`resolve_plugin_repo`, :func:`assert_trusted_owner`,
    :func:`require_trust`) raise instead, so the CLI can map a misconfigured
    command line to exit 2 and keep it distinct from a failed install.
    """
    if not repo or not repo.strip():
        raise InvalidParameter("--repo is required, in 'owner/name' form.")
    repo = repo.strip()
    _split_repo_id(repo, label="agent repository")

    plugin_repo_id = resolve_plugin_repo(plugin_repo)
    owner, plugin_name = assert_trusted_owner(plugin_repo_id)

    directory = fetch_plugin(
        plugin_repo_id,
        revision=plugin_revision,
        token=token,
        endpoint=endpoint,
        cache_dir=cache_dir,
    )
    manifest = verify_manifest(directory, plugin_repo_id)
    spec = PluginSpec(
        repo_id=plugin_repo_id,
        owner=owner,
        name=plugin_name,
        revision=plugin_revision or constants.DEFAULT_AGENT_PLUGIN_REVISION,
        directory=directory,
        manifest=manifest,
        entry_module=str(manifest["entry_module"]),
    )
    require_trust(spec, trust_remote_code=trust_remote_code)

    try:
        module = load_plugin(spec)
        operation, func = select_operation(module)
    except NotSupportedError:
        raise
    except Exception as exc:
        return InstallOutcome(
            ok=False,
            error=f"failed to load plugin {plugin_repo_id}: {exc.__class__.__name__}: {exc}",
            plugin=spec,
            exit_code=1,
        )

    candidates: dict[str, Any] = {
        "repo": repo,
        "name": name,
        "framework": framework,
        "source_framework": framework,
        "local_dir": local_dir,
        # A fetch-only plugin writes where it is told and has no default, so the
        # destination is always resolved here: the caller's --local-dir, else a
        # staging directory. Operations that do not declare ``dest`` never see it.
        "dest": local_dir or str(default_staging_dir(repo)),
        "dry_run": dry_run,
        "yes": yes,
        "force": force,
        "quiet": quiet,
        "endpoint": endpoint,
        "token": token,
    }
    # Unset optionals are dropped so the plugin applies its own defaults; a False
    # boolean is kept because that is a decision the caller made.
    provided = {key: value for key, value in candidates.items() if value is not None}
    try:
        result = func(**_accepted_kwargs(func, provided))
    except Exception as exc:
        return InstallOutcome(
            ok=False,
            error=f"plugin {operation}() failed: {exc.__class__.__name__}: {exc}",
            operation=operation,
            plugin=spec,
            exit_code=1,
        )

    ok = bool(getattr(result, "ok", True))
    if ok:
        return InstallOutcome(ok=True, operation=operation, plugin=spec, result=result)

    error = getattr(result, "error", None) or f"plugin {operation}() reported failure"
    try:
        exit_code = int(getattr(result, "exit_code", 0) or 0)
    except (TypeError, ValueError):
        exit_code = 0
    return InstallOutcome(
        ok=False,
        error=error,
        operation=operation,
        plugin=spec,
        result=result,
        exit_code=exit_code or 1,
    )
