# Copyright (c) Alibaba, Inc. and its affiliates.
"""Locate, fetch, verify and run an agent plugin.

``ms agent install`` resolves which plugin to use, downloads it from a model
repository, verifies it, and hands the agent id to the plugin's entry point. No
framework knowledge lives here: how an agent is registered and what its workspace
looks like are the plugin's decisions. The one exception is the destination
directory, which this module resolves for a plugin that only transports bytes --
it cannot know where such a plugin should write, and the plugin deliberately has
no default of its own.

Trust model
-----------
The **owner allow-list is the authorisation**. It is a compile-time constant
naming only official organisations, and nothing lets a caller point this command
at a plugin whose owner is not on it, so by the time a package has been fetched
the decision to run it was already made by whoever shipped this release. There is
consequently no per-invocation opt-in: an allow-listed plugin is always downloaded
*and* executed, and there is no inspect-only mode. A ``--trust-remote-code`` style
flag is deferred to whichever release supports third-party plugins; exposing one
now would imply a choice the allow-list has already made.

Integrity comes from ``plugin.json``'s ``content_sha256``, not from the hub's own
file listing -- that listing has been observed reporting a git blob SHA-1 in a
``sha256`` field. Be precise about what that buys: it proves the bytes on disk are
the bytes the manifest described, and it gives the audit line in
:func:`log_execution` a stable fingerprint. It is **not** authenticity. The
manifest ships inside the same unsigned repository as the code it describes, so
whoever controls the repository controls the hashes and can make anything verify.
The allow-list is what vouches for the plugin's origin; nothing here vouches for
its contents beyond "unchanged since it was listed". Signing would change that and
is not done yet.

Plugin package contract
-----------------------
Maintainer-facing record of the format; it is deliberately not in the README,
which documents only the supported path of installing an official plugin.

A plugin is a **model** repository (``snapshot_download`` rejects
``repo_type='agent'``) with ``plugin.json`` at its root beside an importable
package or module named by ``entry_module``.

``plugin.json`` -- two fields are required, the rest are display only and never
validated:

* ``entry_module`` (str) -- imported from the download root via
  ``sys.path.insert(0, root)``, so relative imports inside a package work.
* ``content_sha256`` (dict) -- sha256 of every file, keyed by posix path relative
  to the root. Checked in both directions: missing, mismatched and unlisted files
  all fail. Exempt: ``plugin.json`` itself (it cannot hash itself),
  ``.gitattributes`` (the hub injects it) and ``__pycache__``. Keys are validated
  as paths before use, since they are attacker-controlled.
* ``version``, ``frameworks``, ``api``, ``roadmap`` -- feed :meth:`PluginSpec.describe`
  and :meth:`PluginSpec.scope`, nothing else.

The entry module must expose at least one of ``install``, ``fetch_raw``,
``download``, tried in that order. ``capabilities()`` returning
``{"operations": [...], "frameworks": [...], "planned": {...}}`` is authoritative
when present, so a name that is shipped but not implemented is skipped rather than
selected; when it is absent, selection falls back to presence, and when it raises
that is an error rather than an empty declaration.

The chosen operation is called with keyword arguments narrowed to its signature,
from: ``repo``, ``name``, ``framework``, ``source_framework``, ``local_dir``,
``dest``, ``dry_run``, ``yes``, ``force``, ``quiet``, ``endpoint``, ``token``.
Unset optionals are dropped so the plugin's own defaults apply; ``False`` booleans
are kept; ``dest`` is always resolved. Its return value must carry ``ok`` --
required, not defaulted, because it is the only signal deciding whether the user
is told the agent was installed -- plus ``error`` and ``exit_code`` on failure and
``files_written`` / ``root`` for the success message. Raising is also handled.

Two constraints follow from how loading works. The package must use **relative
imports** internally, because it is registered under a directory-scoped alias
rather than its own name so two plugins cannot be served each other's cached
code. And it must not assume its directory stays on ``sys.path`` after the
operation returns: the entry is scoped to the call, since a directory parked at
``sys.path[0]`` lets any file it ships shadow the standard library. Cleaning up
its own staging directory is the plugin's job, not this module's.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import json
import logging
import os
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from .. import constants
from ..api import HubApi
from ..errors import InvalidParameter, NotSupportedError
from ..utils.file_utils import compute_hash, get_cache_dir

logger = logging.getLogger("modelscope_hub.agent")

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
            f"  planned    : {self._planned()}\n"
            f"  directory  : {self.directory}\n"
            f"  manifest   : {len(self.manifest.get('content_sha256') or {})} file(s), "
            f"sha256 {digest}"
        )

    def scope(self) -> str:
        """One line naming what this build covers, for the success path.

        A command that exits 0 otherwise tells a user nothing about which
        frameworks it handled or which operations this plugin version actually
        implements, and both decide whether the result is what they wanted.
        """
        frameworks = ", ".join(map(str, self.manifest.get("frameworks") or [])) or "-"
        operations = ", ".join(map(str, self.manifest.get("api") or [])) or "-"
        return f"frameworks {frameworks} | operations {operations} | planned {self._planned()}"

    def _planned(self) -> str:
        """Operations the manifest declares as not yet implemented, and when."""
        roadmap = self.manifest.get("roadmap") or {}
        if not isinstance(roadmap, dict):
            return "-"
        return ", ".join(f"{name} ({when})" for name, when in sorted(roadmap.items())) or "-"


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


def resolve_plugin_repo(explicit: str | None = None) -> str:
    """Return the plugin repository id: argument, then environment, then default.

    The default is :data:`~modelscope_hub.constants.DEFAULT_AGENT_PLUGIN_REPO`,
    the plugin published under the ModelScope organisation. It is a default and
    not a hard-coded call site because who publishes the plugin is a deployment
    decision -- an override is one flag or one environment variable away, and the
    owner allow-list applies to whichever id wins.
    """
    repo_id = (explicit or "").strip()
    if not repo_id:
        repo_id = (os.environ.get(constants.ENV_AGENT_PLUGIN_REPO) or "").strip()
    if not repo_id:
        repo_id = constants.DEFAULT_AGENT_PLUGIN_REPO
    HubApi._parse_repo_id(repo_id)
    return repo_id


def assert_trusted_owner(repo_id: str) -> tuple[str, str]:
    """Split *repo_id* and require its owner on the allow-list.

    Matching is case-insensitive because that is how the registry treats
    identity: it resolves ``ModelScope/x`` and ``modelscope/x`` to the same
    repository and normalises the owner, so two owners differing only in case
    cannot both exist. An exact comparison would therefore not stop a look-alike
    account -- it would only reject the casing somebody copied from the website.
    """
    owner, name = HubApi._parse_repo_id(repo_id)
    trusted = constants.AGENT_PLUGIN_TRUSTED_OWNERS
    if owner.casefold() not in {entry.casefold() for entry in trusted}:
        error = InvalidParameter(
            f"owner {owner!r} is not allowed to provide the agent plugin. "
            f"Trusted owners: {', '.join(sorted(trusted)) or '(none)'}."
        )
        error.suggestion = (
            "The allow-list is a compile-time constant "
            "(modelscope_hub.constants.AGENT_PLUGIN_TRUSTED_OWNERS), not an "
            "environment variable: it is the trust anchor for a command that runs "
            "downloaded code, so widening it is a reviewed code change."
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

    Transfer executes nothing, and the owner gate in :func:`assert_trusted_owner`
    has already run by the time this is reached -- a package outside the
    allow-list is refused without touching the network.
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

    Strict on purpose, and worth being clear about what strictness buys: it proves
    the files on disk are the files the manifest described, and it makes the
    digest in the :func:`log_execution` audit line mean something, so what was
    recorded as about to run and what actually got imported cannot diverge. It
    does not prove anything about authorship -- the manifest is unsigned and ships
    beside the code it describes, so a repository's owner can make any content
    verify. That is the owner allow-list's job.
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

    missing, mismatched, unexpected, escaped = [], [], [], []
    directory_resolved = directory.resolve()
    for rel, expected in sorted(recorded.items()):
        # Manifest keys are attacker-controlled. ``directory / "/etc/passwd"``
        # discards the directory entirely, so an absolute or parent-climbing key
        # would have this loader read -- and hash -- a file outside the package.
        # Nothing is returned to the caller, so it is not a disclosure, but it is
        # a read the manifest has no business requesting.
        parts = PurePosixPath(rel).parts
        if PurePosixPath(rel).is_absolute() or ".." in parts:
            escaped.append(rel)
            continue
        target = directory / rel
        if not target.resolve().is_relative_to(directory_resolved):
            escaped.append(rel)
            continue
        if not target.is_file():
            missing.append(rel)
            continue
        actual = compute_hash(target)
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
    if escaped:
        problems.append(f"{len(escaped)} manifest key(s) point outside the package: {', '.join(escaped[:5])}")
    if missing:
        problems.append(f"missing {len(missing)} file(s): {', '.join(missing[:5])}")
    if mismatched:
        problems.append(f"sha256 mismatch for {len(mismatched)} file(s): {', '.join(mismatched[:5])}")
    if unexpected:
        problems.append(f"not listed in the manifest: {', '.join(unexpected[:5])}")
    if problems:
        raise NotSupportedError(f"{repo_id}: plugin integrity check failed -- " + "; ".join(problems))
    return manifest


def log_execution(spec: PluginSpec) -> None:
    """Record which build is about to be imported, before it is imported.

    There is no per-invocation opt-in to wait for any more. The owner allow-list
    is the authorisation: it is compile-time, it names only official
    organisations, and nothing lets a user point this command at a plugin whose
    owner is not on it. Executing downloaded code still deserves an audit line
    naming the exact build, emitted *before* the import so that a crash during it
    leaves a trace of what was being loaded.

    A per-invocation opt-in (``--trust-remote-code``) is deliberately deferred:
    it belongs with third-party plugins, which this release does not support, and
    exposing it now would imply a choice the allow-list has already made.
    Reintroducing it means gating here again and refusing instead of logging.
    """
    logger.info("Executing agent plugin:\n%s", spec.describe())


def _module_alias(spec: PluginSpec) -> str:
    """A ``sys.modules`` name unique to the code being loaded.

    ``importlib.import_module(entry_module)`` goes through the global cache, so
    loading two plugins whose entry modules share a name in one process would
    silently run the first one's code for the second.

    The discriminator has to be the *directory*, not the repository and revision:
    those do not identify the bytes. Two checkouts of one repository at one
    revision -- a re-download into a different cache, a development tree beside a
    published one -- are different code with the same identity, and aliasing on
    the pair collides. The entry module name stays in the alias so a traceback is
    still readable.
    """
    resolved = Path(spec.directory).resolve()
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
    safe = re.sub(r"[^A-Za-z0-9_]", "_", spec.entry_module)
    return f"_ms_agent_plugin_{safe}_{digest}"


@contextmanager
def plugin_syspath(directory: Path) -> Iterator[None]:
    """Keep *directory* importable for the duration of the block, then undo it.

    Scoped rather than permanent on purpose. A plugin directory parked at
    ``sys.path[0]`` lets any module it ships shadow the standard library or a
    dependency for the rest of the process, and shipping one is not even a rule
    violation -- every file has to be listed in ``content_sha256``, so a
    ``json.py`` or ``requests.py`` passes the integrity check like anything else.
    A one-shot CLI barely notices; a long-lived process calling
    :func:`install_agent` would stay poisoned.

    It cannot be narrowed to the import alone: plugins import their own sibling
    packages lazily, at call time, so the directory has to stay reachable until
    the operation returns.
    """
    root = str(directory)
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(root)
            except ValueError:
                pass


def load_plugin(spec: PluginSpec) -> Any:
    """Import the plugin's entry module from its downloaded directory.

    The directory goes at the *front* of ``sys.path`` so the fetched revision
    wins over any same-named installed distribution, and the module is registered
    under :func:`_module_alias` rather than its own name so a second plugin cannot
    be served the first one's cached code. Callers that will also *invoke* the
    plugin should hold :func:`plugin_syspath` open for the whole operation.

    Residual limitation, not solved here: a plugin's sibling top-level packages
    (``agent_hub_core`` beside ``agent_hub_plugin``, say) are imported by their own
    names and still land in the global cache, so two plugins shipping different
    copies of one would collide. Isolating that needs a subprocess, which in turn
    needs a serialisable result contract.
    """
    root = str(spec.directory)
    if root not in sys.path:
        sys.path.insert(0, root)

    alias = _module_alias(spec)
    cached = sys.modules.get(alias)
    if cached is not None:
        return cached

    package_init = Path(spec.directory) / spec.entry_module / "__init__.py"
    single_file = Path(spec.directory) / f"{spec.entry_module}.py"
    if package_init.is_file():
        target, search_locations = package_init, [str(package_init.parent)]
    elif single_file.is_file():
        target, search_locations = single_file, None
    else:
        if root in sys.path:
            sys.path.remove(root)
        raise ImportError(f"entry module {spec.entry_module!r} is neither a package nor a module in {root}")

    module_spec = importlib.util.spec_from_file_location(alias, target, submodule_search_locations=search_locations)
    if module_spec is None or module_spec.loader is None:
        if root in sys.path:
            sys.path.remove(root)
        raise ImportError(f"cannot build an import spec for {target}")

    module = importlib.util.module_from_spec(module_spec)
    # Registered before executing so the plugin's own relative and recursive
    # imports resolve to this module rather than re-entering it.
    sys.modules[alias] = module
    try:
        module_spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(alias, None)
        if root in sys.path:
            sys.path.remove(root)
        raise
    return module


def select_operation(module: Any) -> tuple[str, Any]:
    """Pick the entry operation the plugin actually supports.

    ``capabilities()`` is authoritative when present, so a plugin that ships a
    name without implementing it is not selected. When it is *absent* selection
    falls back to presence, which is what lets a minimal plugin work. When it is
    present but fails, that is a broken plugin rather than an undeclaring one, so
    it is an error: silently falling back to presence would pick whatever name
    happens to exist, including a placeholder the plugin chose not to declare.
    """
    declared = None
    capabilities = getattr(module, "capabilities", None)
    if callable(capabilities):
        try:
            payload = capabilities()
        except Exception as exc:
            raise NotSupportedError(
                f"plugin {module.__name__}.capabilities() raised "
                f"{exc.__class__.__name__}: {exc}. Without it there is no way to tell "
                "which operations the plugin implements, so refusing to guess."
            ) from exc
        operations = payload.get("operations") if isinstance(payload, dict) else None
        if operations is None:
            raise NotSupportedError(
                f"plugin {module.__name__}.capabilities() returned no 'operations' "
                f"(got {type(payload).__name__}); cannot tell which entry points it implements."
            )
        declared = set(operations)

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
    endpoint: str | None = None,
    token: str | None = None,
    cache_dir: str | None = None,
) -> InstallOutcome:
    """Fetch or install *repo*'s agent through its framework plugin.

    An allow-listed plugin is always downloaded **and executed** -- there is no
    inspect-only mode. Authorisation is the compile-time owner allow-list, not a
    per-invocation opt-in, so by the time this function is past
    :func:`assert_trusted_owner` the decision has already been made by whoever
    shipped this package.

    Which of fetch or install happens is the plugin's answer, not this function's:
    the entry operation is negotiated in :func:`select_operation`, so a plugin
    that installs into the workspace installs, and one that only transports bytes
    stages them in ``local_dir`` (or :func:`default_staging_dir`) for the install
    layer to place.

    *repo* is passed through uninterpreted beyond requiring ``owner/name``.
    Plugin failures come back as data (``ok`` False); the two gates
    (:func:`resolve_plugin_repo`, :func:`assert_trusted_owner`) raise instead, so
    the CLI can map a misconfigured command line to exit 2 and keep it distinct
    from a failed install.
    """
    if not repo or not repo.strip():
        raise InvalidParameter("--repo is required, in 'owner/name' form.")
    repo = repo.strip()
    HubApi._parse_repo_id(repo)

    plugin_repo_id = resolve_plugin_repo(plugin_repo)
    owner, plugin_name = assert_trusted_owner(plugin_repo_id)

    try:
        directory = fetch_plugin(
            plugin_repo_id,
            revision=plugin_revision,
            token=token,
            endpoint=endpoint,
            cache_dir=cache_dir,
        )
    except NotSupportedError as exc:
        if plugin_repo_id == constants.DEFAULT_AGENT_PLUGIN_REPO:
            # The user never named this repository, so a bare download error
            # leaves them nothing to act on.
            exc.suggestion = (
                f"{plugin_repo_id} is the built-in default. If it is not published yet, "
                f"or you built your own, pass --plugin-repo owner/name (or set "
                f"{constants.ENV_AGENT_PLUGIN_REPO}). Its owner must be one of "
                f"{', '.join(sorted(constants.AGENT_PLUGIN_TRUSTED_OWNERS))}."
            )
        raise
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
    log_execution(spec)

    # The plugin directory is importable for exactly as long as the plugin runs,
    # not for the rest of the process -- see :func:`plugin_syspath`.
    with plugin_syspath(spec.directory):
        try:
            module = load_plugin(spec)
            operation, func = select_operation(module)
        except NotSupportedError as exc:
            # Imported but nothing was runnable. Say that the download succeeded
            # and that no agent work happened, or the message reads like a
            # network failure and sends the user off checking the wrong thing.
            raise NotSupportedError(
                f"plugin {plugin_repo_id}@{spec.revision} downloaded and verified, but was "
                f"NOT executed -- no agent was fetched or installed. {exc}"
            ) from exc
        except Exception as exc:
            return InstallOutcome(
                ok=False,
                error=(
                    f"plugin {plugin_repo_id}@{spec.revision} downloaded and verified, but could "
                    f"not be imported, so it was NOT executed and no agent was fetched or "
                    f"installed: {exc.__class__.__name__}: {exc}"
                ),
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

    # ``ok`` is required, not defaulted. It is the only signal deciding whether
    # the user is told the agent was installed, so defaulting it to True let a
    # plugin returning None, a bare string or an empty dict report success. Every
    # other gate here fails closed; this one has to as well.
    if not hasattr(result, "ok"):
        return InstallOutcome(
            ok=False,
            error=(
                f"plugin {operation}() returned {type(result).__name__} with no 'ok' attribute. "
                "An entry operation must return a result carrying at least 'ok', plus 'error' "
                "and 'exit_code' on failure. Refusing to report an unverifiable install as success."
            ),
            operation=operation,
            plugin=spec,
            result=result,
            exit_code=1,
        )

    if bool(result.ok):
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
