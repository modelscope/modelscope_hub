"""Internal file upload implementation.

Supports single-file and folder uploads with:
- Automatic LFS routing for files > threshold
- SHA256 deduplication via validate_blobs
- Batched commit operations
- Parallel blob uploads
- tqdm progress display
"""

from __future__ import annotations

import base64
import fnmatch
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, IO, Union

from tqdm.auto import tqdm

from .constants import DOWNLOAD_CHUNK_SIZE, UPLOAD_LFS_THRESHOLD, RepoType
from .errors import FileIntegrityError, HubError
from .utils.file_utils import compute_hash, get_file_size
from .utils.logger import get_logger

if TYPE_CHECKING:
    from .config import HubConfig
    from ._legacy_api import LegacyClient

logger = get_logger("upload")

# Type alias for path-or-fileobj inputs
PathOrFileObj = Union[str, Path, bytes, BinaryIO, IO[bytes]]

# Maximum number of operations per commit request
_COMMIT_BATCH_SIZE = 50


def _compute_sha256_from_path(path: Path) -> str:
    """Compute SHA256 hex digest for a file on disk."""
    return compute_hash(path, "sha256")


def _compute_sha256_from_bytes(data: bytes) -> str:
    """Compute SHA256 hex digest for in-memory bytes."""
    return hashlib.sha256(data).hexdigest()


def _read_as_bytes(path_or_fileobj: PathOrFileObj) -> bytes:
    """Read the content as bytes from various input types."""
    if isinstance(path_or_fileobj, bytes):
        return path_or_fileobj
    if isinstance(path_or_fileobj, (str, Path)):
        return Path(path_or_fileobj).read_bytes()
    # File-like object
    pos = path_or_fileobj.tell()
    content = path_or_fileobj.read()
    path_or_fileobj.seek(pos)
    return content


def _matches_patterns(path: str, patterns: list[str] | None) -> bool:
    """Check if a path matches any glob pattern."""
    if not patterns:
        return False
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


class UploadManager:
    """Internal file upload implementation.

    Dependencies are injected via constructor for testability.
    """

    def __init__(self, legacy_client: "LegacyClient", config: "HubConfig") -> None:
        self._client = legacy_client
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def upload_file(
        self,
        repo_id: str,
        repo_type: str,
        path_or_fileobj: PathOrFileObj,
        path_in_repo: str,
        commit_message: str = "Upload file",
        revision: str = "master",
    ) -> dict:
        """Upload a single file to a repository.

        Files larger than :data:`~.constants.UPLOAD_LFS_THRESHOLD` are uploaded
        via the LFS blob path; smaller files use direct commit.

        Parameters
        ----------
        repo_id:
            Repository identifier (``owner/name``).
        repo_type:
            One of the :class:`~.constants.RepoType` values.
        path_or_fileobj:
            Local file path, bytes, or binary file object.
        path_in_repo:
            Destination path within the repository.
        commit_message:
            Commit message for the upload.
        revision:
            Target branch or tag.

        Returns
        -------
        dict
            Commit info from the server.
        """
        # Determine size and sha256
        size = self._get_size(path_or_fileobj)
        sha256 = self._get_sha256(path_or_fileobj)

        if size > UPLOAD_LFS_THRESHOLD:
            # LFS path: validate → upload blob → commit pointer
            return self._upload_lfs(
                repo_id=repo_id,
                repo_type=repo_type,
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                sha256=sha256,
                size=size,
                commit_message=commit_message,
                revision=revision,
            )
        else:
            # Direct commit path: encode content as base64
            return self._upload_direct(
                repo_id=repo_id,
                repo_type=repo_type,
                path_or_fileobj=path_or_fileobj,
                path_in_repo=path_in_repo,
                commit_message=commit_message,
                revision=revision,
            )

    def upload_folder(
        self,
        repo_id: str,
        repo_type: str,
        folder_path: str | Path,
        path_in_repo: str = "",
        commit_message: str = "Upload folder",
        revision: str = "master",
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        max_workers: int = 4,
    ) -> dict:
        """Upload an entire folder to a repository.

        Parameters
        ----------
        repo_id:
            Repository identifier.
        repo_type:
            Repository type.
        folder_path:
            Local folder to upload.
        path_in_repo:
            Destination prefix in the repository.
        commit_message:
            Commit message.
        revision:
            Target branch.
        allow_patterns:
            Only upload files matching these globs.
        ignore_patterns:
            Skip files matching these globs.
        max_workers:
            Number of parallel upload threads for LFS blobs.

        Returns
        -------
        dict
            Final commit info.
        """
        folder = Path(folder_path).resolve()
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")

        # Collect files
        all_files: list[tuple[Path, str]] = []
        for root, _dirs, filenames in os.walk(folder):
            for fname in filenames:
                local_path = Path(root) / fname
                rel_path = local_path.relative_to(folder).as_posix()
                repo_path = f"{path_in_repo}/{rel_path}" if path_in_repo else rel_path

                if allow_patterns and not _matches_patterns(rel_path, allow_patterns):
                    continue
                if ignore_patterns and _matches_patterns(rel_path, ignore_patterns):
                    continue

                all_files.append((local_path, repo_path))

        if not all_files:
            logger.info("No files found to upload in %s", folder)
            return {"message": "No files to upload"}

        logger.info("Uploading %d files from %s", len(all_files), folder)

        # Classify files: LFS vs direct
        lfs_files: list[tuple[Path, str, str, int]] = []  # (path, repo_path, sha256, size)
        direct_files: list[tuple[Path, str]] = []  # (path, repo_path)

        for local_path, repo_path in all_files:
            size = local_path.stat().st_size
            if size > UPLOAD_LFS_THRESHOLD:
                sha256 = _compute_sha256_from_path(local_path)
                lfs_files.append((local_path, repo_path, sha256, size))
            else:
                direct_files.append((local_path, repo_path))

        # Upload LFS blobs in parallel
        if lfs_files:
            self._upload_lfs_blobs_parallel(
                repo_id=repo_id,
                repo_type=repo_type,
                lfs_files=lfs_files,
                max_workers=max_workers,
            )

        # Build commit operations and submit in batches
        operations: list[dict] = []

        # LFS pointer operations
        for local_path, repo_path, sha256, size in lfs_files:
            pointer_content = self._lfs_pointer(sha256, size)
            operations.append({
                "action": "create",
                "file_path": repo_path,
                "content": base64.b64encode(pointer_content.encode()).decode(),
            })

        # Direct file operations
        for local_path, repo_path in direct_files:
            content = local_path.read_bytes()
            operations.append({
                "action": "create",
                "file_path": repo_path,
                "content": base64.b64encode(content).decode(),
            })

        # Batch commits
        last_result: dict = {}
        for i in range(0, len(operations), _COMMIT_BATCH_SIZE):
            batch = operations[i:i + _COMMIT_BATCH_SIZE]
            batch_msg = commit_message if i == 0 else f"{commit_message} (part {i // _COMMIT_BATCH_SIZE + 1})"
            last_result = self._client.create_commit(
                repo_id=repo_id,
                repo_type=repo_type,
                operations=batch,
                commit_message=batch_msg,
                revision=revision,
            )
            logger.info("Committed batch %d/%d", i // _COMMIT_BATCH_SIZE + 1,
                        (len(operations) + _COMMIT_BATCH_SIZE - 1) // _COMMIT_BATCH_SIZE)

        return last_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_size(self, path_or_fileobj: PathOrFileObj) -> int:
        """Get size of a path-or-fileobj input."""
        if isinstance(path_or_fileobj, bytes):
            return len(path_or_fileobj)
        if isinstance(path_or_fileobj, (str, Path)):
            return Path(path_or_fileobj).stat().st_size
        return get_file_size(path_or_fileobj)

    def _get_sha256(self, path_or_fileobj: PathOrFileObj) -> str:
        """Compute SHA256 of a path-or-fileobj input."""
        if isinstance(path_or_fileobj, bytes):
            return _compute_sha256_from_bytes(path_or_fileobj)
        if isinstance(path_or_fileobj, (str, Path)):
            return _compute_sha256_from_path(Path(path_or_fileobj))
        # File-like: read, hash, rewind
        content = _read_as_bytes(path_or_fileobj)
        return _compute_sha256_from_bytes(content)

    def _upload_lfs(
        self,
        repo_id: str,
        repo_type: str,
        path_or_fileobj: PathOrFileObj,
        path_in_repo: str,
        sha256: str,
        size: int,
        commit_message: str,
        revision: str,
    ) -> dict:
        """Upload a large file via the LFS blob path."""
        # Step 1: Validate if blob already exists
        validated = self._client.validate_blobs(
            repo_id=repo_id,
            repo_type=repo_type,
            objects=[{"oid": sha256, "size": size}],
        )

        upload_url = validated.get(sha256)
        if upload_url:
            # Step 2: Upload the blob
            logger.info("Uploading LFS blob %s... (%d bytes)", sha256[:8], size)
            data = self._open_for_upload(path_or_fileobj)
            self._client.upload_blob(upload_url=upload_url, data=data, size=size)
        else:
            logger.info("Blob %s... already exists, skipping upload", sha256[:8])

        # Step 3: Commit the LFS pointer
        pointer_content = self._lfs_pointer(sha256, size)
        operation = {
            "action": "create",
            "file_path": path_in_repo,
            "content": base64.b64encode(pointer_content.encode()).decode(),
        }
        return self._client.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=[operation],
            commit_message=commit_message,
            revision=revision,
        )

    def _upload_direct(
        self,
        repo_id: str,
        repo_type: str,
        path_or_fileobj: PathOrFileObj,
        path_in_repo: str,
        commit_message: str,
        revision: str,
    ) -> dict:
        """Upload a small file directly via commit API."""
        content = _read_as_bytes(path_or_fileobj)
        operation = {
            "action": "create",
            "file_path": path_in_repo,
            "content": base64.b64encode(content).decode(),
        }
        return self._client.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=[operation],
            commit_message=commit_message,
            revision=revision,
        )

    def _upload_lfs_blobs_parallel(
        self,
        repo_id: str,
        repo_type: str,
        lfs_files: list[tuple[Path, str, str, int]],
        max_workers: int,
    ) -> None:
        """Validate and upload LFS blobs in parallel."""
        # Batch validate all blobs
        objects = [{"oid": sha256, "size": size} for _, _, sha256, size in lfs_files]
        validated = self._client.validate_blobs(
            repo_id=repo_id,
            repo_type=repo_type,
            objects=objects,
        )

        # Filter to only blobs that need uploading
        to_upload: list[tuple[Path, str, int]] = []
        for local_path, _repo_path, sha256, size in lfs_files:
            url = validated.get(sha256)
            if url:
                to_upload.append((local_path, url, size))
            else:
                logger.debug("Blob %s... already exists", sha256[:8])

        if not to_upload:
            logger.info("All LFS blobs already exist, no upload needed")
            return

        logger.info("Uploading %d LFS blob(s)", len(to_upload))

        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._upload_single_blob, path, url, size): path
                for path, url, size in to_upload
            }
            with tqdm(total=len(to_upload), desc="Uploading blobs", unit="file") as pbar:
                for future in as_completed(futures):
                    path = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        errors.append(f"{path}: {exc}")
                        logger.error("Failed to upload blob %s: %s", path, exc)
                    finally:
                        pbar.update(1)

        if errors:
            raise HubError(f"{len(errors)} blob upload(s) failed: {errors[0]}")

    def _upload_single_blob(self, local_path: Path, upload_url: str, size: int) -> None:
        """Upload a single blob file to its presigned URL."""
        with open(local_path, "rb") as fh:
            self._client.upload_blob(upload_url=upload_url, data=fh, size=size)

    def _open_for_upload(self, path_or_fileobj: PathOrFileObj) -> BinaryIO | bytes:
        """Return a readable object suitable for upload."""
        if isinstance(path_or_fileobj, bytes):
            return path_or_fileobj
        if isinstance(path_or_fileobj, (str, Path)):
            return open(Path(path_or_fileobj), "rb")
        return path_or_fileobj

    @staticmethod
    def _lfs_pointer(sha256: str, size: int) -> str:
        """Generate a Git LFS pointer file content."""
        return (
            f"version https://git-lfs.github.com/spec/v1\n"
            f"oid sha256:{sha256}\n"
            f"size {size}\n"
        )
