"""Internal file download implementation.

Supports single-file and whole-repo (snapshot) downloads with:
- HTTP Range-based resume
- SHA256 integrity verification
- tqdm progress display
- Parallel downloads via ThreadPoolExecutor
- Local snapshot cache directory management
"""

from __future__ import annotations

import fnmatch
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm.auto import tqdm

from .constants import DOWNLOAD_CHUNK_SIZE
from .errors import FileIntegrityError, NetworkError
from .utils.file_utils import compute_hash, ensure_dir
from .utils.logger import get_logger

if TYPE_CHECKING:
    from .config import HubConfig
    from ._legacy_api import LegacyClient

logger = get_logger("download")


def _matches_patterns(path: str, patterns: list[str] | None) -> bool:
    """Check if path matches any of the glob patterns."""
    if not patterns:
        return False
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


class DownloadManager:
    """Internal file download implementation.

    Dependencies are injected via constructor to keep this class testable.
    """

    def __init__(self, legacy_client: "LegacyClient", config: "HubConfig") -> None:
        self._client = legacy_client
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def download_file(
        self,
        repo_id: str,
        repo_type: str,
        file_path: str,
        revision: str = "master",
        cache_dir: Path | None = None,
        local_dir: Path | None = None,
        force: bool = False,
    ) -> Path:
        """Download a single file from a repository.

        When *local_dir* is provided the file is placed directly under that
        directory (preserving relative path structure).  Otherwise the
        standard cache layout is used::

            {cache_dir}/{type}s/{owner}--{name}/snapshots/{revision}/{file_path}

        Parameters
        ----------
        repo_id:
            Repository identifier (``owner/name``).
        repo_type:
            One of the :class:`~.constants.RepoType` values.
        file_path:
            Path within the repository.
        revision:
            Branch, tag, or commit hash.
        cache_dir:
            Override for the default cache directory.
        local_dir:
            When set, download directly into this directory instead of cache.
        force:
            Re-download even if file exists in cache.

        Returns
        -------
        Path
            Absolute path to the downloaded (or cached) file on disk.
        """
        if local_dir is not None:
            target = Path(local_dir) / file_path
        else:
            root = self._repo_cache_dir(repo_id, repo_type, cache_dir)
            target = root / "snapshots" / revision / file_path

        if not force and target.exists():
            logger.debug("Cache hit: %s", target)
            return target

        ensure_dir(target.parent)
        self._download_with_resume(repo_id, repo_type, file_path, revision, target)
        return target

    def download_repo(
        self,
        repo_id: str,
        repo_type: str,
        revision: str = "master",
        cache_dir: Path | None = None,
        local_dir: Path | None = None,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        max_workers: int = 4,
    ) -> Path:
        """Download an entire repository (snapshot download).

        Parameters
        ----------
        repo_id:
            Repository identifier (``owner/name``).
        repo_type:
            One of the :class:`~.constants.RepoType` values.
        revision:
            Branch, tag, or commit hash.
        cache_dir:
            Override for the default cache directory.
        local_dir:
            When set, download directly into this directory instead of cache.
        allow_patterns:
            Only files matching these globs will be downloaded.
        ignore_patterns:
            Files matching these globs will be skipped.
        max_workers:
            Number of parallel download threads.

        Returns
        -------
        Path
            Absolute path to the snapshot/local directory.
        """
        if local_dir is not None:
            output_dir = ensure_dir(Path(local_dir))
        else:
            root = self._repo_cache_dir(repo_id, repo_type, cache_dir)
            output_dir = ensure_dir(root / "snapshots" / revision)

        files = self._client.list_repo_files(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            recursive=True,
        )

        file_paths: list[str] = []
        for f in files:
            path = f.get("Path") or f.get("path") or f.get("Name") or ""
            ftype = f.get("Type") or f.get("type") or "blob"
            if ftype == "tree":
                continue
            if not path:
                continue
            if allow_patterns and not _matches_patterns(path, allow_patterns):
                continue
            if ignore_patterns and _matches_patterns(path, ignore_patterns):
                continue
            file_paths.append(path)

        if not file_paths:
            logger.info("No files to download for %s@%s", repo_id, revision)
            return output_dir

        logger.info("Downloading %d files from %s@%s", len(file_paths), repo_id, revision)

        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.download_file,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    file_path=fp,
                    revision=revision,
                    cache_dir=cache_dir,
                    local_dir=local_dir,
                ): fp
                for fp in file_paths
            }

            with tqdm(total=len(file_paths), desc="Downloading", unit="file") as pbar:
                for future in as_completed(futures):
                    fp = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        errors.append(f"{fp}: {exc}")
                        logger.error("Failed to download %s: %s", fp, exc)
                    finally:
                        pbar.update(1)

        if errors:
            logger.warning("%d file(s) failed to download", len(errors))

        return output_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _repo_cache_dir(
        self,
        repo_id: str,
        repo_type: str,
        cache_dir: Path | None = None,
    ) -> Path:
        """Compute the cache directory for a given repo."""
        base = cache_dir or self._config.cache_dir
        segment = f"{repo_type}s" if not repo_type.endswith("s") else repo_type
        # Encode repo_id: owner/name → owner--name for filesystem safety
        safe_id = repo_id.replace("/", "--")
        return ensure_dir(base / segment / safe_id)

    def _download_with_resume(
        self,
        repo_id: str,
        repo_type: str,
        file_path: str,
        revision: str,
        target: Path,
    ) -> Path:
        """Download a file with HTTP Range resume support."""
        # Use a temp file for partial downloads
        tmp_path = target.with_suffix(target.suffix + ".incomplete")

        existing_size = 0
        if tmp_path.exists():
            existing_size = tmp_path.stat().st_size

        # Prepare headers for resume
        extra_headers: dict[str, str] = {}
        if existing_size > 0:
            extra_headers["Range"] = f"bytes={existing_size}-"
            logger.debug("Resuming download from byte %d", existing_size)

        try:
            resp = self._client.download_stream(
                repo_id=repo_id,
                repo_type=repo_type,
                file_path=file_path,
                revision=revision,
                headers=extra_headers if extra_headers else None,
            )
        except Exception as exc:
            raise NetworkError(f"Download failed for {file_path}: {exc}") from exc

        # Determine total size
        content_length = resp.headers.get("Content-Length")
        total_size = int(content_length) if content_length else None
        is_resumed = resp.status_code == 206

        if is_resumed and total_size:
            total_size += existing_size

        # Write to temp file
        mode = "ab" if is_resumed else "wb"
        if not is_resumed:
            existing_size = 0

        with tqdm(
            total=total_size,
            initial=existing_size,
            unit="B",
            unit_scale=True,
            desc=Path(file_path).name,
            leave=False,
        ) as pbar:
            with open(tmp_path, mode) as fh:
                for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        pbar.update(len(chunk))

        # Move temp → final
        tmp_path.replace(target)
        logger.debug("Downloaded: %s", target)
        return target

    def verify_file(self, file_path: Path, expected_sha256: str) -> bool:
        """Verify a downloaded file's SHA256 hash.

        Raises :class:`~.errors.FileIntegrityError` on mismatch.
        """
        actual = compute_hash(file_path, "sha256")
        if actual != expected_sha256:
            raise FileIntegrityError(
                f"Hash mismatch for {file_path.name}: "
                f"expected {expected_sha256[:16]}..., got {actual[:16]}..."
            )
        return True
