"""Unit tests for legacy (pre-1.38) cache auto-detection.

These are network-free tests for ``DownloadManager._find_legacy_repo_dir``,
which lets ``download_repo`` / ``download_file`` reuse an existing old-SDK
cache instead of re-downloading into the new layout.
"""
from __future__ import annotations

from modelscope_hub.api import HubApi


def _make_download_manager():
    """Build a network-free DownloadManager via the public HubApi facade."""
    return HubApi().downloader


class TestFindLegacyRepoDir:
    def test_detects_modelscope_cache_layout(self, tmp_path):
        # MODELSCOPE_CACHE explicitly set: {base}/models/{owner}/{name___}
        legacy = tmp_path / "models" / "Qwen" / "Qwen3___5-4B"
        legacy.mkdir(parents=True)
        (legacy / "config.json").write_text("{}")

        dm = _make_download_manager()
        found = dm._find_legacy_repo_dir("Qwen/Qwen3.5-4B", "model", tmp_path)
        assert found == legacy

    def test_detects_default_hub_segment_layout(self, tmp_path):
        # Default cache (~/.cache/modelscope/hub): {base}/hub/models/{owner}/{name___}
        legacy = tmp_path / "hub" / "models" / "Qwen" / "Qwen3___5-4B"
        legacy.mkdir(parents=True)
        (legacy / "config.json").write_text("{}")

        dm = _make_download_manager()
        found = dm._find_legacy_repo_dir("Qwen/Qwen3.5-4B", "model", tmp_path)
        assert found == legacy

    def test_multi_dot_name_encoding(self, tmp_path):
        legacy = tmp_path / "models" / "Qwen" / "Qwen2___5-0___5B"
        legacy.mkdir(parents=True)
        (legacy / "config.json").write_text("{}")

        dm = _make_download_manager()
        found = dm._find_legacy_repo_dir("Qwen/Qwen2.5-0.5B", "model", tmp_path)
        assert found == legacy

    def test_clean_cache_returns_none(self, tmp_path):
        dm = _make_download_manager()
        assert dm._find_legacy_repo_dir(
            "Qwen/Qwen3.5-4B", "model", tmp_path) is None

    def test_empty_legacy_dir_returns_none(self, tmp_path):
        legacy = tmp_path / "models" / "Qwen" / "Qwen3___5-4B"
        legacy.mkdir(parents=True)  # exists but empty

        dm = _make_download_manager()
        assert dm._find_legacy_repo_dir(
            "Qwen/Qwen3.5-4B", "model", tmp_path) is None
