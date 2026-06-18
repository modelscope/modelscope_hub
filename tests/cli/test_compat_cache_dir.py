"""Unit tests for compat layer cache_dir path resolution.

Verifies that ``_resolve_legacy_paths`` correctly transforms cache_dir into
the legacy flat layout ``{cache_dir}/{repo_id}/`` via local_dir, and that
the snapshot/file download wrappers propagate the resolved paths to the
underlying HubApi methods.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _resolve_legacy_paths unit tests (pure logic, mock api)
# ---------------------------------------------------------------------------
@pytest.mark.mock_only
class TestResolveLegacyPaths:
    """Direct tests for the path resolution helper."""

    def _make_api(self, default_cache: str = "/default/cache") -> MagicMock:
        api = MagicMock()
        api._config.cache_dir = Path(default_cache)
        return api

    def test_local_dir_specified_passthrough(self):
        """local_dir explicitly set -> original (cache_dir, local_dir) returned unchanged."""
        from modelscope_hub.compat.file_download import _resolve_legacy_paths

        api = self._make_api()
        result = _resolve_legacy_paths("owner/repo", "/some/cache", "/custom/dir", api)
        assert result == ("/some/cache", "/custom/dir")

    def test_local_dir_none_cache_dir_specified(self):
        """local_dir=None, cache_dir given -> (None, '{cache_dir}/{repo_id}')."""
        from modelscope_hub.compat.file_download import _resolve_legacy_paths

        api = self._make_api()
        result = _resolve_legacy_paths("damo/bert", "/tmp/cache", None, api)
        assert result == (None, "/tmp/cache/damo/bert")

    def test_local_dir_none_cache_dir_none_uses_default(self):
        """local_dir=None, cache_dir=None -> uses api._config.cache_dir."""
        from modelscope_hub.compat.file_download import _resolve_legacy_paths

        api = self._make_api("/default/cache")
        result = _resolve_legacy_paths("owner/model", None, None, api)
        assert result == (None, "/default/cache/owner/model")

    def test_repo_id_multi_level_path(self):
        """repo_id with multiple path segments is correctly joined."""
        from modelscope_hub.compat.file_download import _resolve_legacy_paths

        api = self._make_api()
        result = _resolve_legacy_paths("org/sub/name", "/data", None, api)
        assert result == (None, "/data/org/sub/name")

    def test_cache_dir_trailing_slash_normalized(self):
        """Trailing slash in cache_dir is normalized by Path."""
        from modelscope_hub.compat.file_download import _resolve_legacy_paths

        api = self._make_api()
        result = _resolve_legacy_paths("owner/repo", "/tmp/cache/", None, api)
        assert result == (None, "/tmp/cache/owner/repo")


# ---------------------------------------------------------------------------
# snapshot_download integration-level mock tests
# ---------------------------------------------------------------------------
@pytest.mark.mock_only
class TestSnapshotDownloadCacheCompat:
    """Verify path conversion propagates correctly to download_repo."""

    @patch("modelscope_hub.compat.snapshot_download.HubApi")
    def test_cache_dir_only_converts_to_local_dir(self, MockHubApi):
        """cache_dir without local_dir -> download_repo gets local_dir='{cache_dir}/{id}'."""
        from modelscope_hub.compat.snapshot_download import snapshot_download

        mock_api = MagicMock()
        mock_api._config.cache_dir = Path("/default/cache")
        mock_api.download_repo.return_value = "/tmp/cache/damo/bert"
        MockHubApi.return_value = mock_api

        result = snapshot_download(model_id="damo/bert", cache_dir="/tmp/cache")

        mock_api.download_repo.assert_called_once()
        call_kwargs = mock_api.download_repo.call_args[1]
        assert call_kwargs["cache_dir"] is None
        assert call_kwargs["local_dir"] == "/tmp/cache/damo/bert"

    @patch("modelscope_hub.compat.snapshot_download.HubApi")
    def test_local_dir_explicit_not_overridden(self, MockHubApi):
        """Explicit local_dir is passed through without modification."""
        from modelscope_hub.compat.snapshot_download import snapshot_download

        mock_api = MagicMock()
        mock_api._config.cache_dir = Path("/default/cache")
        mock_api.download_repo.return_value = "/custom/dir"
        MockHubApi.return_value = mock_api

        result = snapshot_download(model_id="damo/bert", local_dir="/custom/dir")

        mock_api.download_repo.assert_called_once()
        call_kwargs = mock_api.download_repo.call_args[1]
        assert call_kwargs["local_dir"] == "/custom/dir"

    @patch("modelscope_hub.compat.snapshot_download.HubApi")
    def test_dataset_snapshot_download_path_conversion(self, MockHubApi):
        """dataset_snapshot_download also applies path conversion."""
        from modelscope_hub.compat.snapshot_download import dataset_snapshot_download

        mock_api = MagicMock()
        mock_api._config.cache_dir = Path("/default/cache")
        mock_api.download_repo.return_value = "/data/hub/my_org/dataset1"
        MockHubApi.return_value = mock_api

        result = dataset_snapshot_download(
            dataset_id="my_org/dataset1", cache_dir="/data/hub",
        )

        mock_api.download_repo.assert_called_once()
        call_kwargs = mock_api.download_repo.call_args[1]
        assert call_kwargs["cache_dir"] is None
        assert call_kwargs["local_dir"] == "/data/hub/my_org/dataset1"


# ---------------------------------------------------------------------------
# model_file_download / dataset_file_download integration-level mock tests
# ---------------------------------------------------------------------------
@pytest.mark.mock_only
class TestFileDownloadCacheCompat:
    """Verify path conversion propagates correctly to download_file."""

    @patch("modelscope_hub.compat.file_download.HubApi")
    def test_cache_dir_only_converts_to_local_dir(self, MockHubApi):
        """cache_dir without local_dir -> download_file gets local_dir='{cache_dir}/{id}'."""
        from modelscope_hub.compat.file_download import model_file_download

        mock_api = MagicMock()
        mock_api._config.cache_dir = Path("/default/cache")
        mock_api.download_file.return_value = "/data/hub/qwen/chat/model.bin"
        MockHubApi.return_value = mock_api

        result = model_file_download("qwen/chat", "model.bin", cache_dir="/data/hub")

        mock_api.download_file.assert_called_once()
        call_kwargs = mock_api.download_file.call_args[1]
        assert call_kwargs["cache_dir"] is None
        assert call_kwargs["local_dir"] == "/data/hub/qwen/chat"

    @patch("modelscope_hub.compat.file_download.HubApi")
    def test_local_dir_explicit_passthrough(self, MockHubApi):
        """Explicit local_dir is passed through without modification."""
        from modelscope_hub.compat.file_download import model_file_download

        mock_api = MagicMock()
        mock_api._config.cache_dir = Path("/default/cache")
        mock_api.download_file.return_value = "/my/dir/model.bin"
        MockHubApi.return_value = mock_api

        result = model_file_download(
            "qwen/chat", "model.bin", local_dir="/my/dir",
        )

        mock_api.download_file.assert_called_once()
        call_kwargs = mock_api.download_file.call_args[1]
        assert call_kwargs["local_dir"] == "/my/dir"

    @patch("modelscope_hub.compat.file_download.HubApi")
    def test_dataset_file_download_path_conversion(self, MockHubApi):
        """dataset_file_download also applies path conversion."""
        from modelscope_hub.compat.file_download import dataset_file_download

        mock_api = MagicMock()
        mock_api._config.cache_dir = Path("/default/cache")
        mock_api.download_file.return_value = "/data/hub/org/ds/train.csv"
        MockHubApi.return_value = mock_api

        result = dataset_file_download(
            "org/ds", "train.csv", cache_dir="/data/hub",
        )

        mock_api.download_file.assert_called_once()
        call_kwargs = mock_api.download_file.call_args[1]
        assert call_kwargs["cache_dir"] is None
        assert call_kwargs["local_dir"] == "/data/hub/org/ds"


# ---------------------------------------------------------------------------
# Legacy cache hit simulation (verify paths point to correct location)
# ---------------------------------------------------------------------------
@pytest.mark.mock_only
class TestLegacyCacheHitSimulation:
    """Simulate old cache structure and verify local_dir resolves to it."""

    @patch("modelscope_hub.compat.file_download.HubApi")
    def test_cache_hit_file_exists_at_resolved_path(self, MockHubApi, tmp_path):
        """Pre-existing file at legacy path is reachable via resolved local_dir."""
        from modelscope_hub.compat.file_download import model_file_download

        # Create legacy cache structure
        legacy_dir = tmp_path / "owner" / "name"
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "README.md").write_text("# Hello")

        mock_api = MagicMock()
        mock_api._config.cache_dir = Path("/default/cache")
        mock_api.download_file.return_value = str(legacy_dir / "README.md")
        MockHubApi.return_value = mock_api

        result = model_file_download(
            "owner/name", "README.md", cache_dir=str(tmp_path),
        )

        mock_api.download_file.assert_called_once()
        call_kwargs = mock_api.download_file.call_args[1]
        resolved_local_dir = call_kwargs["local_dir"]

        # Verify local_dir points to the legacy directory where file exists
        assert resolved_local_dir == str(tmp_path / "owner" / "name")
        assert (Path(resolved_local_dir) / "README.md").exists()

    @patch("modelscope_hub.compat.snapshot_download.HubApi")
    def test_snapshot_cache_hit_directory_exists(self, MockHubApi, tmp_path):
        """Pre-existing directory at legacy path is reachable via resolved local_dir."""
        from modelscope_hub.compat.snapshot_download import snapshot_download

        # Create legacy cache structure with multiple files
        legacy_dir = tmp_path / "org" / "model"
        legacy_dir.mkdir(parents=True)
        (legacy_dir / "config.json").write_text("{}")
        (legacy_dir / "model.bin").write_bytes(b"\x00" * 100)

        mock_api = MagicMock()
        mock_api._config.cache_dir = Path("/default/cache")
        mock_api.download_repo.return_value = str(legacy_dir)
        MockHubApi.return_value = mock_api

        result = snapshot_download(
            model_id="org/model", cache_dir=str(tmp_path),
        )

        mock_api.download_repo.assert_called_once()
        call_kwargs = mock_api.download_repo.call_args[1]
        resolved_local_dir = call_kwargs["local_dir"]

        # Verify local_dir points to the legacy directory
        assert resolved_local_dir == str(tmp_path / "org" / "model")
        assert (Path(resolved_local_dir) / "config.json").exists()
        assert (Path(resolved_local_dir) / "model.bin").exists()
