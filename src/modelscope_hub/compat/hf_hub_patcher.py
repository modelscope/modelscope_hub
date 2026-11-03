"""Monkey-patching of the ``huggingface_hub`` module onto the ModelScope Hub.

Adapted from ``modelscope/utils/hf_util/patcher.py`` (Apache-2.0, Alibaba),
re-targeted from the legacy ``modelscope`` SDK onto ``modelscope_hub``.
The transformers/peft/diffusers patches stay with the ``modelscope`` package;
this module must stay importable without transformers. ``huggingface_hub``
itself is *not* a declared dependency of this SDK — the patch entry points
check for it at call time and raise an actionable ``ImportError`` when it is
not installed.

:func:`patch_hf_hub` replaces the ``huggingface_hub`` module-level functions
that HF-ecosystem code calls — ``hf_hub_download``, ``file_exists``,
``whoami``, ``create_repo``, ``upload_file``, ``upload_folder``,
``create_commit``, and the ``repocard`` helpers — with ModelScope-backed
implementations. Patch markers reuse the historical attribute names
(``hf_api._hf_hub_download_origin`` etc.), so a legacy ``modelscope``
patcher and this module treat each other's patches as already applied.

Compatibility contract (kept identical to the legacy patcher):

- ``whoami`` returns ``{'name': ...}``;
- ``create_repo`` returns a ``RepoUrl``-shaped str (``HfRepoUrl``);
- ``upload_folder`` / ``upload_file`` / ``create_commit`` return a commit
  info with ``commit_url`` shaped like ``{endpoint}/models/{repo_id}/files``;
- the default revision is ``master`` (HF's ``main`` is normalised away);
- ``create_commit`` only supports ``Add`` operations.

One deliberate behaviour change: uploads go through the ModelScope commit
API (resumable, LFS-routed) instead of a local git clone + push.

Version support: verified against ``huggingface_hub`` 0.19.4 (source-level
symbol audit) and 1.32 (full test suite); ``huggingface_hub>=0.19`` is the
documented minimum. Note that ``kernels>=0.17`` itself requires
``huggingface_hub>=1.10``, so a ``kernels`` install resolves to a 1.x hub.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import modelscope_hub
from modelscope_hub import compat
from modelscope_hub.constants import DEFAULT_MODELSCOPE_DATA_ENDPOINT
from modelscope_hub.utils.logger import get_logger

logger = get_logger()

__all__ = ["patch_hf_hub", "unpatch_hf_hub", "hf_hub_patch_scope"]


def _require_hf_hub():
    """Import ``huggingface_hub`` or raise an actionable ``ImportError``.

    ``huggingface_hub`` is deliberately not a declared dependency of this
    SDK; the huggingface_hub patches simply cannot work without it, so the
    entry points fail with installation instructions at call time.
    """
    try:
        import huggingface_hub
    except ImportError as e:
        raise ImportError(
            "The `huggingface_hub` package is required to redirect huggingface_hub "
            "calls to ModelScope, but it is not installed. "
            "Install it with: pip install huggingface_hub"
        ) from e
    return huggingface_hub


def _ms_revision(revision: str | None) -> str:
    """Translate an HF revision string into one ModelScope accepts."""
    return "master" if revision in (None, "main") else revision


def _token(token) -> str | None:
    """HF tokens may be bool (``True``/``False``) or strings; keep strings only."""
    return token if isinstance(token, str) else None


def _hub_api(token=None) -> modelscope_hub.HubApi:
    """A ``HubApi`` for this call; ``token=None`` falls back to stored credentials."""
    return modelscope_hub.HubApi(token=_token(token))


@dataclass
class HfCommitInfo:
    """Return shape of the ``upload_*`` / ``create_commit`` patches.

    Mirrors the legacy ``modelscope.utils.repo_utils.CommitInfo`` fields that
    callers (transformers ``push_to_hub``, diffusers) actually consume.
    """

    commit_url: str
    commit_message: str | None
    commit_description: str | None
    oid: str | None


class HfRepoUrl(str):
    """Lenient ``RepoUrl`` replacement returned by the ``create_repo`` patch.

    The legacy patcher returned the ``modelscope.utils.repo_utils.RepoUrl``
    str subclass (url plus ``repo_type``/``repo_id`` attributes), not HF's
    own class; recent ``huggingface_hub`` releases additionally reject
    non-Hugging Face hosts in their ``RepoUrl`` constructor. This subclass
    keeps the legacy duck-typing shape (``str``, ``.repo_id``, ``.repo_type``).
    """

    def __new__(cls, url: str, repo_type: str = "model", repo_id: str | None = None):
        obj = super().__new__(cls, url)
        obj.repo_type = repo_type  # type: ignore[attr-defined]
        obj.repo_id = repo_id  # type: ignore[attr-defined]
        return obj


def _commit_url(repo_id: str) -> str:
    return f"{DEFAULT_MODELSCOPE_DATA_ENDPOINT}/models/{repo_id}/files"


# ---------------------------------------------------------------------------
# Download / metadata patches
# ---------------------------------------------------------------------------
def _hf_hub_download(
    repo_id: str,
    filename: str,
    *,
    subfolder: str | None = None,
    repo_type: str | None = None,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    token=None,
    local_files_only: bool = False,
    **kwargs,
) -> str:
    """ModelScope-backed ``huggingface_hub.hf_hub_download``."""
    if kwargs:
        logger.warning(
            "The passed in %s are not used by modelscope_hub.",
            ", ".join(sorted(kwargs)),
        )
    if repo_type not in (None, "model", "dataset"):
        raise ValueError(f"repo_type={repo_type} is not supported in ModelScope")
    file_path = os.path.join(subfolder, filename) if subfolder else filename
    if repo_type == "dataset":
        return compat.dataset_file_download(
            repo_id,
            file_path,
            revision=_ms_revision(revision),
            cache_dir=str(cache_dir) if cache_dir else None,
            local_dir=str(local_dir) if local_dir else None,
            token=_token(token),
            local_files_only=local_files_only,
        )
    return compat.model_file_download(
        repo_id,
        file_path,
        revision=_ms_revision(revision),
        cache_dir=str(cache_dir) if cache_dir else None,
        local_dir=str(local_dir) if local_dir else None,
        token=_token(token),
        local_files_only=local_files_only,
    )


def _file_exists(
    repo_id: str,
    filename: str,
    *,
    repo_type: str | None = None,
    revision: str | None = None,
    token=None,
    **kwargs,
) -> bool:
    """ModelScope-backed ``huggingface_hub.file_exists``."""
    if repo_type is not None:
        logger.warning("repo_type is ignored by modelscope_hub; only model repos can be queried.")
    infos = _hub_api(token).list_repo_files(repo_id, "model", revision=_ms_revision(revision))
    return any(info.path == filename for info in infos)


def _whoami(token=None) -> dict:
    """ModelScope-backed ``huggingface_hub.whoami``."""
    user = _hub_api(token).whoami()
    return {"name": getattr(user, "username", None) or "unknown"}


# ---------------------------------------------------------------------------
# Repository / upload patches
# ---------------------------------------------------------------------------
def _create_repo(repo_id: str, *, token=None, private: bool = False, **kwargs) -> HfRepoUrl:
    """ModelScope-backed ``huggingface_hub.create_repo``; returns a ``RepoUrl``-shaped str."""
    compat.LegacyHubApi(token=_token(token)).create_repo(
        repo_id,
        visibility="private" if private else "public",
        **kwargs,
    )
    return HfRepoUrl(
        f"{DEFAULT_MODELSCOPE_DATA_ENDPOINT}/models/{repo_id}",
        repo_type="model",
        repo_id=repo_id,
    )


def _upload_file(
    *,
    path_or_fileobj,
    path_in_repo: str,
    repo_id: str,
    token=None,
    repo_type: str | None = "model",
    revision: str | None = "master",
    commit_message: str | None = None,
    commit_description: str | None = None,
    **kwargs,
) -> None:
    """ModelScope-backed ``huggingface_hub.upload_file`` (commit-API upload)."""
    if repo_type not in (None, "model"):
        raise ValueError(f"repo_type={repo_type} is not supported in ModelScope")
    _hub_api(token).upload_file(
        repo_id,
        "model",
        path_or_fileobj,
        path_in_repo,
        commit_message=commit_message,
        commit_description=commit_description,
        revision=_ms_revision(revision),
    )


def _upload_folder(
    *,
    repo_id: str,
    folder_path: str | Path,
    path_in_repo: str | None = None,
    commit_message: str | None = None,
    commit_description: str | None = None,
    token=None,
    repo_type: str | None = "model",
    revision: str | None = "master",
    ignore_patterns=None,
    allow_patterns=None,
    **kwargs,
) -> HfCommitInfo:
    """ModelScope-backed ``huggingface_hub.upload_folder`` (commit-API upload)."""
    if repo_type not in (None, "model"):
        raise ValueError(f"repo_type={repo_type} is not supported in ModelScope")
    _hub_api(token).upload_folder(
        repo_id,
        "model",
        folder_path,
        path_in_repo=path_in_repo or "",
        commit_message=commit_message,
        commit_description=commit_description,
        revision=_ms_revision(revision),
        ignore_patterns=ignore_patterns,
        allow_patterns=allow_patterns,
    )
    return HfCommitInfo(
        commit_url=_commit_url(repo_id),
        commit_message=commit_message,
        commit_description=commit_description,
        oid=None,
    )


def _create_commit(
    repo_id: str,
    operations,
    *,
    commit_message: str,
    commit_description: str | None = None,
    token=None,
    repo_type: str | None = None,
    revision: str | None = "master",
    **kwargs,
) -> None:
    """ModelScope-backed ``huggingface_hub.create_commit`` (Add operations only).

    Each ``CommitOperationAdd`` becomes one commit through the ModelScope
    commit API, so a multi-operation batch produces several commits on the
    Hub, unlike a single git push.
    """
    from huggingface_hub import CommitOperationAdd

    if repo_type not in (None, "model"):
        raise ValueError(f"repo_type={repo_type} is not supported in ModelScope")
    api = _hub_api(token)
    for op in operations:
        if not isinstance(op, CommitOperationAdd):
            raise ValueError("ModelScope create_commit only supports Add operations for now.")
        api.upload_file(
            repo_id,
            "model",
            op.path_or_fileobj,
            op.path_in_repo,
            commit_message=commit_message,
            commit_description=commit_description,
            revision=_ms_revision(revision),
        )


def _repocard_load(cls, repo_id_or_path, repo_type=None, token=None, ignore_metadata_errors=False):
    """ModelScope-backed ``huggingface_hub.repocard.RepoCard.load``."""
    if os.path.exists(repo_id_or_path):
        file_path = repo_id_or_path
    elif repo_type == "dataset":
        file_path = compat.dataset_file_download(repo_id_or_path, "README.md", token=_token(token))
    else:
        file_path = compat.model_file_download(repo_id_or_path, "README.md", token=_token(token))

    with open(file_path, encoding="utf-8") as f:
        repo_card = cls(f.read(), ignore_metadata_errors=ignore_metadata_errors)
        if not hasattr(repo_card.data, "tags"):
            repo_card.data.tags = []
        return repo_card


# ---------------------------------------------------------------------------
# Patch / unpatch
# ---------------------------------------------------------------------------
def patch_hf_hub() -> None:
    """Monkey-patch ``huggingface_hub`` to route Hub calls to ModelScope.

    Idempotent: each function is guarded by its historical origin marker, so
    a legacy ``modelscope`` patcher that has already applied the same
    attribute names is respected instead of being double-patched.

    Raises:
        ImportError: when ``huggingface_hub`` is not installed.
    """
    huggingface_hub = _require_hf_hub()
    from huggingface_hub import hf_api, repocard

    if not hasattr(hf_api, "_hf_hub_download_origin"):
        setattr(hf_api, "_hf_hub_download_origin", huggingface_hub.file_download.hf_hub_download)
        huggingface_hub.hf_hub_download = _hf_hub_download
        huggingface_hub.file_download.hf_hub_download = _hf_hub_download

    if not hasattr(hf_api, "_file_exists_origin"):
        setattr(hf_api, "_file_exists_origin", hf_api.file_exists)
        hf_api.file_exists = _file_exists
        huggingface_hub.file_exists = _file_exists
        huggingface_hub.hf_api.file_exists = _file_exists

    if not hasattr(hf_api, "_whoami_origin"):
        setattr(hf_api, "_whoami_origin", hf_api.whoami)
        hf_api.whoami = _whoami
        huggingface_hub.whoami = _whoami
        huggingface_hub.hf_api.whoami = _whoami

    if not hasattr(hf_api, "_create_repo_origin"):
        setattr(hf_api, "_create_repo_origin", hf_api.create_repo)
        hf_api.create_repo = _create_repo
        huggingface_hub.create_repo = _create_repo
        huggingface_hub.hf_api.create_repo = _create_repo

    if not hasattr(hf_api, "_upload_folder_origin"):
        setattr(hf_api, "_upload_folder_origin", hf_api.upload_folder)
        hf_api.upload_folder = _upload_folder
        huggingface_hub.upload_folder = _upload_folder
        huggingface_hub.hf_api.upload_folder = _upload_folder

    if not hasattr(hf_api, "_upload_file_origin"):
        setattr(hf_api, "_upload_file_origin", hf_api.upload_file)
        hf_api.upload_file = _upload_file
        huggingface_hub.upload_file = _upload_file
        huggingface_hub.hf_api.upload_file = _upload_file
        repocard.upload_file = _upload_file

    if not hasattr(hf_api, "_create_commit_origin"):
        setattr(hf_api, "_create_commit_origin", hf_api.create_commit)
        hf_api.create_commit = _create_commit
        huggingface_hub.create_commit = _create_commit
        huggingface_hub.hf_api.create_commit = _create_commit

    if not hasattr(repocard.RepoCard, "_validate_origin"):
        setattr(repocard.RepoCard, "_validate_origin", repocard.RepoCard.validate)
        repocard.RepoCard.validate = lambda *args, **kwargs: None
        setattr(repocard.RepoCard, "_load_origin", repocard.RepoCard.load)
        repocard.RepoCard.load = MethodType(_repocard_load, repocard.RepoCard)


def unpatch_hf_hub() -> None:
    """Undo :func:`patch_hf_hub` and restore the original huggingface_hub functions.

    Without ``huggingface_hub`` nothing can have been patched, so this is a
    silent no-op rather than an error (patch/unpatch stay symmetric in
    minimal environments).
    """
    try:
        import huggingface_hub
        from huggingface_hub import hf_api, repocard
    except ImportError:
        return

    if hasattr(repocard.RepoCard, "_validate_origin"):
        repocard.RepoCard.validate = repocard.RepoCard._validate_origin
        delattr(repocard.RepoCard, "_validate_origin")
    if hasattr(repocard.RepoCard, "_load_origin"):
        repocard.RepoCard.load = repocard.RepoCard._load_origin
        delattr(repocard.RepoCard, "_load_origin")

    if hasattr(hf_api, "_hf_hub_download_origin"):
        origin = hf_api._hf_hub_download_origin
        huggingface_hub.file_download.hf_hub_download = origin
        huggingface_hub.hf_hub_download = origin
        delattr(hf_api, "_hf_hub_download_origin")

    if hasattr(hf_api, "_file_exists_origin"):
        origin = hf_api._file_exists_origin
        hf_api.file_exists = origin
        huggingface_hub.file_exists = origin
        huggingface_hub.hf_api.file_exists = origin
        delattr(hf_api, "_file_exists_origin")

    if hasattr(hf_api, "_whoami_origin"):
        origin = hf_api._whoami_origin
        hf_api.whoami = origin
        huggingface_hub.whoami = origin
        huggingface_hub.hf_api.whoami = origin
        delattr(hf_api, "_whoami_origin")

    if hasattr(hf_api, "_create_repo_origin"):
        origin = hf_api._create_repo_origin
        hf_api.create_repo = origin
        huggingface_hub.create_repo = origin
        huggingface_hub.hf_api.create_repo = origin
        delattr(hf_api, "_create_repo_origin")

    if hasattr(hf_api, "_upload_folder_origin"):
        origin = hf_api._upload_folder_origin
        hf_api.upload_folder = origin
        huggingface_hub.upload_folder = origin
        huggingface_hub.hf_api.upload_folder = origin
        delattr(hf_api, "_upload_folder_origin")

    if hasattr(hf_api, "_upload_file_origin"):
        origin = hf_api._upload_file_origin
        hf_api.upload_file = origin
        huggingface_hub.upload_file = origin
        huggingface_hub.hf_api.upload_file = origin
        repocard.upload_file = origin
        delattr(hf_api, "_upload_file_origin")

    if hasattr(hf_api, "_create_commit_origin"):
        origin = hf_api._create_commit_origin
        hf_api.create_commit = origin
        huggingface_hub.create_commit = origin
        huggingface_hub.hf_api.create_commit = origin
        delattr(hf_api, "_create_commit_origin")


@contextlib.contextmanager
def hf_hub_patch_scope():
    """Apply :func:`patch_hf_hub` for the duration of the ``with`` block.

    Nesting is safe: an already-applied patch (including one applied by the
    legacy ``modelscope`` patcher) is reused, and only the outermost scope
    restores the original functions.

    Raises:
        ImportError: when ``huggingface_hub`` is not installed.
    """
    _require_hf_hub()
    from huggingface_hub import hf_api

    if hasattr(hf_api, "_hf_hub_download_origin"):
        yield
        return
    patch_hf_hub()
    try:
        yield
    finally:
        unpatch_hf_hub()
