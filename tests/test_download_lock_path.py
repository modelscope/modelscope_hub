"""Regression tests for download lock filename safety.

Remote repositories can contain perfectly valid path components close to the
common 255-byte filesystem basename limit. The local lock file must therefore
never embed the raw remote path in its own basename; otherwise the lock fails
before the actual target file is even opened.
"""

from __future__ import annotations

from pathlib import Path

from modelscope_hub._download import DownloadManager, _optional_file_lock
from modelscope_hub.api import HubApi
from modelscope_hub.compat.file_download import dataset_file_download

_LONG_IMAGE_NAME = "a" * 215 + ".jpg"
_LONG_FILE_PATH = f"images/{_LONG_IMAGE_NAME}"


def _download_manager() -> DownloadManager:
    return HubApi().downloader


class TestDownloadLockPath:
    def test_file_lock_name_is_fixed_length_for_long_remote_paths(self, tmp_path):
        dm = _download_manager()

        lock_path = dm._lock_path(
            "OmniDocBench/OmniDocBench",
            "dataset",
            cache_dir=tmp_path,
            file_path=_LONG_FILE_PATH,
        )

        assert lock_path.parent == tmp_path / ".lock"
        assert lock_path.name.startswith("file_")
        assert lock_path.name.endswith(".lock")
        assert len(lock_path.name.encode("utf-8")) < 255
        assert _LONG_IMAGE_NAME not in lock_path.name

    def test_repo_lock_name_is_fixed_length_for_long_repo_ids(self, tmp_path):
        dm = _download_manager()
        long_repo_id = f"{'owner' * 30}/{'repo' * 40}"

        lock_path = dm._lock_path(long_repo_id, "model", cache_dir=tmp_path)

        assert lock_path.name.startswith("repo_")
        assert lock_path.name.endswith(".lock")
        assert len(lock_path.name.encode("utf-8")) < 255
        assert "owner" not in lock_path.name

    def test_lock_key_is_deterministic_and_file_specific(self, tmp_path):
        dm = _download_manager()

        first = dm._lock_path("owner/repo", "model", cache_dir=tmp_path, file_path="config.json")
        second = dm._lock_path("owner/repo", "model", cache_dir=tmp_path, file_path="config.json")
        other_file = dm._lock_path("owner/repo", "model", cache_dir=tmp_path, file_path="tokenizer.json")
        repo_lock = dm._lock_path("owner/repo", "model", cache_dir=tmp_path)

        assert first == second
        assert first != other_file
        assert first != repo_lock

    def test_optional_file_lock_can_acquire_hashed_long_path(self, tmp_path):
        dm = _download_manager()
        lock_path = dm._lock_path(
            "OmniDocBench/OmniDocBench",
            "dataset",
            cache_dir=tmp_path,
            file_path=_LONG_FILE_PATH,
        )

        with _optional_file_lock(lock_path):
            assert lock_path.exists()


class TestCompatDownloadUsesSafeLockNames:
    def test_dataset_file_download_accepts_long_file_path(self, tmp_path, monkeypatch):
        """The legacy compat entry point reaches the same safe lock path.

        No network is needed: patch the low-level transfer after the lock has
        been acquired, and write the expected target file directly.
        """

        def fake_download_with_resume(
            self,
            repo_id: str,
            repo_type: str,
            file_path: str,
            revision: str,
            target: Path,
            **kwargs,
        ) -> Path:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"image-bytes")
            return target

        monkeypatch.setattr(DownloadManager, "_download_with_resume", fake_download_with_resume)

        cache_dir = tmp_path / "cache"
        result = dataset_file_download(
            "OmniDocBench/OmniDocBench",
            _LONG_FILE_PATH,
            cache_dir=str(cache_dir),
            local_dir=str(tmp_path / "local"),
            endpoint="https://modelscope.cn",
        )

        assert Path(result).read_bytes() == b"image-bytes"
        lock_files = list((cache_dir / ".lock").glob("*.lock"))
        assert lock_files
        assert all(len(path.name.encode("utf-8")) < 255 for path in lock_files)
