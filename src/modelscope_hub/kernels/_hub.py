"""ModelScope-backed stand-in for the small ``HfApi`` surface used by ``kernels``.

The upstream ``kernels`` library routes every Hub interaction through
``kernels.hf_hub._get_hf_api()``. The patched factory (see
:mod:`modelscope_hub.kernels._patch`) returns :class:`MsKernelApi`: a
duck-typed API client that forwards the handful of methods ``kernels``
actually calls onto :class:`modelscope_hub.HubApi`, mapping HF kernel
repositories onto ModelScope model repositories.

Method-to-method mapping:

- ``snapshot_download``      -> :meth:`modelscope_hub.HubApi.download_repo`
- ``hf_hub_download``        -> :meth:`modelscope_hub.HubApi.download_file`
- ``list_repo_refs``         -> :meth:`modelscope_hub.legacy.list_revisions_detail`
- ``list_repo_tree``         -> :meth:`modelscope_hub.HubApi.list_repo_files`
- ``get_organization_overview`` -> local trusted-publisher allowlist

``huggingface_hub`` is imported lazily inside the methods that need its
dataclasses/exceptions: it arrives as a transitive dependency of ``kernels``
and this module must stay importable without it.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from types import SimpleNamespace

import modelscope_hub
from modelscope_hub.constants import ENV_KERNELS_TRUSTED_PUBLISHERS

#: Publisher organizations trusted when ``trust_remote_code=False``. ModelScope
#: has no equivalent of the Hub's ``trustedKernelPublisher`` flag, so the
#: default trust decision falls back to this allowlist (extend it via the
#: ``MS_KERNELS_TRUSTED_PUBLISHERS`` environment variable, comma-separated).
TRUSTED_KERNEL_PUBLISHERS: frozenset[str] = frozenset({"kernels-community"})

# Kernel-version refs (``v1``, ``v2``, ...) as created by the kernels Hub.
_VERSION_REF_RE = re.compile(r"v(\d+)")

# Full git commit hashes (SHA-1); lockfiles pin revisions this way.
_COMMIT_HASH_RE = re.compile(r"[0-9a-f]{40}", re.IGNORECASE)

_LOCKFILE_COMMIT_HINT = (
    "Revision {revision!r} looks like a commit hash. Commit hashes are "
    "hub-specific: a kernel lockfile created against the Hugging Face Hub "
    "cannot resolve on ModelScope. Re-create the lockfile in the ModelScope "
    "environment (get_locked_kernel / install_kernel_all_variants from "
    "modelscope_hub.kernels, or the kernels CLI under patch_kernels_hub) so "
    "it records ModelScope commits."
)


def _note_commit_hash_hint(exc: BaseException, revision: str | None) -> None:
    """Append a lockfile hint when a full commit-hash revision is not found.

    PEP 678 notes require Python 3.11+; on older interpreters the hint is
    skipped silently (the original exception propagates unchanged either way).
    """
    if revision is None or not _COMMIT_HASH_RE.fullmatch(str(revision)):
        return
    add_note = getattr(exc, "add_note", None)
    if add_note is not None:
        add_note(_LOCKFILE_COMMIT_HINT.format(revision=revision))


def _ms_revision(revision: str | None) -> str:
    """Translate an HF revision string into one ModelScope accepts."""
    return "master" if revision in (None, "main") else revision


#: Synthetic OID -> ModelScope branch/tag name, filled by ``list_repo_refs``.
#: ModelScope's revision API does not expose git commit hashes, but kernels
#: 0.18 validates ``Oid.from_str(target_commit)`` (40/64 hex digits) and both
#: kernels 0.17 and 0.18 use ``target_commit`` as the download revision. The
#: shim therefore hands out a deterministic synthetic OID per (repo, ref) and
#: maps it back to the real ref name at every download/list entry point.
_synthetic_oids: dict[str, str] = {}


def _synthetic_oid(repo_id: str, ref_name: str) -> str:
    """A stable 40-hex stand-in for the commit hash ModelScope does not expose."""
    oid = hashlib.sha1(f"{repo_id}:{ref_name}".encode()).hexdigest()
    _synthetic_oids.setdefault(oid, ref_name)
    return oid


def _resolve_revision(revision: str | None) -> str | None:
    """Map a synthetic OID back to its ModelScope branch/tag name."""
    if revision is None:
        return None
    return _synthetic_oids.get(str(revision), revision)


def _ms_repo_type(repo_type: str | None) -> str:
    """Map HF repo types onto ModelScope repo types (kernel repos are models)."""
    return "model" if repo_type in (None, "kernel", "model") else repo_type


def _trusted_publishers() -> frozenset[str]:
    publishers = set(TRUSTED_KERNEL_PUBLISHERS)
    extra = os.environ.get(ENV_KERNELS_TRUSTED_PUBLISHERS, "")
    publishers.update(part.strip() for part in extra.split(",") if part.strip())
    return frozenset(publishers)


def _user_agent(extra: str | dict | None) -> str | dict:
    """Telemetry info appended to the SDK user agent on every download."""
    if extra:
        return extra
    return f"ms-kernels; modelscope_hub/{modelscope_hub.__version__}"


class MsKernelApi:
    """Minimal ``HfApi`` look-alike that forwards to ModelScope.

    ``hub_api`` exists for dependency injection (tests); production callers get
    a lazily constructed :class:`modelscope_hub.HubApi`. Only the methods that
    ``kernels`` calls are implemented.
    """

    def __init__(
        self,
        hub_api: modelscope_hub.HubApi | None = None,
        user_agent: str | dict | None = None,
    ) -> None:
        self._hub_api_instance = hub_api
        self._user_agent = user_agent

    def _api(self) -> modelscope_hub.HubApi:
        if self._hub_api_instance is None:
            self._hub_api_instance = modelscope_hub.HubApi()
        return self._hub_api_instance

    # ------------------------------------------------------------------
    # Downloads
    # ------------------------------------------------------------------
    def snapshot_download(
        self,
        repo_id: str,
        *,
        repo_type: str | None = None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        cache_dir: str | Path | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,  # HF-signature compatibility
    ) -> Path:
        """Download a repository snapshot, mirroring ``HfApi.snapshot_download``."""
        from huggingface_hub.errors import LocalEntryNotFoundError

        ms_revision = _ms_revision(_resolve_revision(revision))
        try:
            return self._api().download_repo(
                repo_id,
                _ms_repo_type(repo_type),
                revision=ms_revision,
                cache_dir=Path(cache_dir) if cache_dir else None,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                local_files_only=local_files_only,
                user_agent=_user_agent(self._user_agent),
            )
        except modelscope_hub.CacheNotFound as e:
            raise LocalEntryNotFoundError(
                f"Cannot find a local snapshot for {repo_id} (revision: {ms_revision}). "
                "The kernel must be downloaded while online first."
            ) from e
        except modelscope_hub.NotExistError as e:
            _note_commit_hash_hint(e, revision)
            raise

    def hf_hub_download(
        self,
        repo_id: str | None = None,
        *,
        repo_type: str | None = None,
        filename: str | None = None,
        cache_dir: str | Path | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,  # HF-signature compatibility
    ) -> Path:
        """Download a single repository file, mirroring ``HfApi.hf_hub_download``."""
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

        if repo_id is None or filename is None:
            raise ValueError("Both `repo_id` and `filename` must be provided.")
        ms_revision = _ms_revision(_resolve_revision(revision))
        try:
            return self._api().download_file(
                repo_id,
                _ms_repo_type(repo_type),
                file_path=filename,
                revision=ms_revision,
                cache_dir=Path(cache_dir) if cache_dir else None,
                local_files_only=local_files_only,
                user_agent=_user_agent(self._user_agent),
            )
        except modelscope_hub.CacheNotFound as e:
            raise LocalEntryNotFoundError(
                f"Cannot find the local cache entry for {repo_id}/{filename} "
                f"(revision: {ms_revision}). The file must be downloaded while online first."
            ) from e
        except modelscope_hub.NotExistError as e:
            _note_commit_hash_hint(e, revision)
            raise EntryNotFoundError(f"File {filename} does not exist in {repo_id} (revision: {ms_revision}).") from e

    # ------------------------------------------------------------------
    # Repository metadata
    # ------------------------------------------------------------------
    def list_repo_refs(self, repo_id: str | None = None, *, repo_type: str | None = None, **kwargs):
        """List repository branches and tags, mirroring ``HfApi.list_repo_refs``.

        ModelScope's revision API does not expose git commit hashes, so every
        ref's ``target_commit`` is a deterministic synthetic OID for the ref
        name (see :func:`_synthetic_oid`); the download entry points map it
        back to the real name, so both kernels 0.17 (which downloads by
        ``target_commit`` verbatim) and kernels 0.18 (which validates it with
        ``Oid.from_str`` before fetching) work unchanged. When the repository
        carries no ``v<N>`` refs at all (typical for a plain mirror with only
        a ``master`` branch), ``master`` is exposed as kernel version ``v1``
        so that ``get_kernel(..., version=1)`` works without the mirror having
        to create version branches.
        """
        from huggingface_hub.hf_api import GitRefInfo

        if repo_id is None:
            raise ValueError("`repo_id` must be provided.")
        branches, tags = self._api().legacy.list_revisions_detail(repo_id, _ms_repo_type(repo_type))
        branch_refs = [
            GitRefInfo(
                name=ref["Revision"],
                ref=f"refs/heads/{ref['Revision']}",
                target_commit=_synthetic_oid(repo_id, ref["Revision"]),
            )
            for ref in branches or []
        ]
        tag_refs = [
            GitRefInfo(
                name=ref["Revision"],
                ref=f"refs/tags/{ref['Revision']}",
                target_commit=_synthetic_oid(repo_id, ref["Revision"]),
            )
            for ref in tags or []
        ]
        if not any(_VERSION_REF_RE.fullmatch(ref.name) for ref in branch_refs + tag_refs):
            branch_refs.append(
                GitRefInfo(name="v1", ref="refs/heads/v1", target_commit=_synthetic_oid(repo_id, "master"))
            )
        return SimpleNamespace(branches=branch_refs, tags=tag_refs, converts=[])

    def list_repo_tree(
        self,
        repo_id: str | None = None,
        *,
        path_in_repo: str | None = None,
        repo_type: str | None = None,
        revision: str | None = None,
        recursive: bool = False,  # ModelScope always lists recursively
        **kwargs,  # HF-signature compatibility
    ):
        """List the direct children of a repository folder, mirroring ``HfApi.list_repo_tree``.

        ModelScope cannot list the direct children of a subfolder
        non-recursively, so the full recursive tree is fetched and the direct
        subfolders derived from it — the only thing ``kernels`` consumes.
        """
        from huggingface_hub.hf_api import RepoFolder

        if repo_id is None:
            raise ValueError("`repo_id` must be provided.")
        root = path_in_repo or ""
        prefix = f"{root}/" if root else ""
        infos = self._api().list_repo_files(
            repo_id,
            _ms_repo_type(repo_type),
            revision=_ms_revision(_resolve_revision(revision)),
            recursive=True,
        )
        folders: set[str] = set()
        for info in infos:
            path = info.path
            if prefix and not path.startswith(prefix):
                continue
            rest = path[len(prefix) :]
            if not rest:
                continue
            first, _, remainder = rest.partition("/")
            if remainder or (info.is_dir and not remainder):
                # Either the parent of a nested entry, or an explicit
                # tree entry at direct-child depth.
                folders.add(first)
        return [
            RepoFolder(path=f"{root}/{name}" if root else name, oid="", last_commit=None) for name in sorted(folders)
        ]

    def file_exists(
        self,
        repo_id: str | None = None,
        filename: str | None = None,
        *,
        repo_type: str | None = None,
        revision: str | None = None,
        **kwargs,  # HF-signature compatibility
    ) -> bool:
        """Check whether ``filename`` exists in the repository."""
        if repo_id is None or filename is None:
            raise ValueError("Both `repo_id` and `filename` must be provided.")
        infos = self._api().list_repo_files(
            repo_id,
            _ms_repo_type(repo_type),
            revision=_ms_revision(_resolve_revision(revision)),
            recursive=True,
        )
        return any(info.path == filename for info in infos)

    # ------------------------------------------------------------------
    # Publisher trust
    # ------------------------------------------------------------------
    def get_organization_overview(self, organization: str):
        """Stand-in for the Hub's organization lookup.

        There is no ``trustedKernelPublisher`` equivalent on ModelScope, so
        trust is decided against :data:`TRUSTED_KERNEL_PUBLISHERS` (plus the
        ``MS_KERNELS_TRUSTED_PUBLISHERS`` environment variable). Untrusted
        publishers cause kernels' usual ``ValueError`` with instructions to
        pass ``trust_remote_code=True``.
        """
        return SimpleNamespace(trustedKernelPublisher=organization in _trusted_publishers())
