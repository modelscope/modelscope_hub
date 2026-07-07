"""Unit tests for compat layer cache_dir path resolution.

Verifies that ``_resolve_legacy_paths`` passes parameters through so that
the new API uses its standard cache layout
({cache_dir}/{type}s/{owner}--{name}/snapshots/{rev}/...), consistent with
CLI ``ms download`` behavior.
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

    def test_cache_dir_specified_passthrough(self):
        """cache_dir given, local_dir=None -> (cache_dir, None) passed through."""
        from modelscope_hub.compat.file_download import _resolve_legacy_paths

        api = self._make_api()
        result = _resolve_legacy_paths("damo/bert", "/tmp/cache", None, api)
        assert result == ("/tmp/cache", None)

    def test_both_none_passthrough(self):
        """cache_dir=None, local_dir=None -> (None, None) for standard cache layout."""
        from modelscope_hub.compat.file_download import _resolve_legacy_paths

        api = self._make_api("/default/cache")
        result = _resolve_legacy_paths("owner/model", None, None, api)
        assert result == (None, None)

    def test_repo_id_not_used_in_path(self):
        """repo_id is not appended — path construction is left to the new API."""
        from modelscope_hub.compat.file_download import _resolve_legacy_paths

        api = self._make_api()
        result = _resolve_legacy_paths("org/sub/name", "/data", None, api)
        assert result == ("/data", None)

    def test_cache_dir_trailing_slash_preserved(self):
        """cache_dir is passed through as-is (api.py handles Path conversion)."""
        from modelscope_hub.compat.file_download import _resolve_legacy_paths

        api = self._make_api()
        result = _resolve_legacy_paths("owner/repo", "/tmp/cache/", None, api)
        assert result == ("/tmp/cache/", None)


# ---------------------------------------------------------------------------
# snapshot_download integration-level mock tests
# ---------------------------------------------------------------------------
@pytest.mark.mock_only
class TestSnapshotDownloadCacheCompat:
    """Verify path conversion propagates correctly to download_repo."""

    @patch("modelscope_hub.compat.snapshot_download.HubApi")
    def test_cache_dir_passed_through(self, MockHubApi):
        """cache_dir without local_dir -> download_repo gets cache_dir directly."""
        from modelscope_hub.compat.snapshot_download import snapshot_download

        mock_api = MagicMock()
        mock_api._config.cache_dir = Path("/default/cache")
        mock_api.download_repo.return_value = "/tmp/cache/models/damo--bert/snapshots/master"
        MockHubApi.return_value = mock_api

        result = snapshot_download(model_id="damo/bert", cache_dir="/tmp/cache")

        mock_api.download_repo.assert_called_once()
        call_kwargs = mock_api.download_repo.call_args[1]
        assert call_kwargs["cache_dir"] == "/tmp/cache"
        assert call_kwargs["local_dir"] is None

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
    def test_dataset_snapshot_download_cache_dir_passthrough(self, MockHubApi):
        """dataset_snapshot_download passes cache_dir through."""
        from modelscope_hub.compat.snapshot_download import dataset_snapshot_download

        mock_api = MagicMock()
        mock_api._config.cache_dir = Path("/default/cache")
        mock_api.download_repo.return_value = "/data/hub/datasets/my_org--dataset1/snapshots/master"
        MockHubApi.return_value = mock_api

        result = dataset_snapshot_download(
            dataset_id="my_org/dataset1", cache_dir="/data/hub",
        )

        mock_api.download_repo.assert_called_once()
        call_kwargs = mock_api.download_repo.call_args[1]
        assert call_kwargs["cache_dir"] == "/data/hub"
        assert call_kwargs["local_dir"] is None


# ---------------------------------------------------------------------------
# model_file_download / dataset_file_download integration-level mock tests
# ---------------------------------------------------------------------------
@pytest.mark.mock_only
class TestFileDownloadCacheCompat:
    """Verify path conversion propagates correctly to download_file."""

    @patch("modelscope_hub.compat.file_download.HubApi")
    def test_cache_dir_passed_through(self, MockHubApi):
        """cache_dir without local_dir -> download_file gets cache_dir directly."""
        from modelscope_hub.compat.file_download import model_file_download

        mock_api = MagicMock()
        mock_api._config.cache_dir = Path("/default/cache")
        mock_api.download_file.return_value = "/data/hub/models/qwen--chat/snapshots/master/model.bin"
        MockHubApi.return_value = mock_api

        result = model_file_download("qwen/chat", "model.bin", cache_dir="/data/hub")

        mock_api.download_file.assert_called_once()
        call_kwargs = mock_api.download_file.call_args[1]
        assert call_kwargs["cache_dir"] == "/data/hub"
        assert call_kwargs["local_dir"] is None

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
    def test_dataset_file_download_cache_dir_passthrough(self, MockHubApi):
        """dataset_file_download passes cache_dir through."""
        from modelscope_hub.compat.file_download import dataset_file_download

        mock_api = MagicMock()
        mock_api._config.cache_dir = Path("/default/cache")
        mock_api.download_file.return_value = "/data/hub/datasets/org--ds/snapshots/master/train.csv"
        MockHubApi.return_value = mock_api

        result = dataset_file_download(
            "org/ds", "train.csv", cache_dir="/data/hub",
        )

        mock_api.download_file.assert_called_once()
        call_kwargs = mock_api.download_file.call_args[1]
        assert call_kwargs["cache_dir"] == "/data/hub"
        assert call_kwargs["local_dir"] is None


# ---------------------------------------------------------------------------
# Standard cache layout verification (new behavior)
# ---------------------------------------------------------------------------
@pytest.mark.mock_only
class TestStandardCacheLayout:
    """Verify that compat layer now uses the same cache layout as CLI."""

    @patch("modelscope_hub.compat.file_download.HubApi")
    def test_no_args_uses_standard_cache(self, MockHubApi, tmp_path):
        """No cache_dir, no local_dir -> standard cache layout via new API."""
        from modelscope_hub.compat.file_download import model_file_download

        mock_api = MagicMock()
        mock_api._config.cache_dir = Path("/default/cache")
        mock_api.download_file.return_value = "/default/cache/models/owner--name/snapshots/master/README.md"
        MockHubApi.return_value = mock_api

        result = model_file_download("owner/name", "README.md")

        mock_api.download_file.assert_called_once()
        call_kwargs = mock_api.download_file.call_args[1]
        assert call_kwargs["cache_dir"] is None
        assert call_kwargs["local_dir"] is None

    @patch("modelscope_hub.compat.snapshot_download.HubApi")
    def test_snapshot_no_args_uses_standard_cache(self, MockHubApi):
        """snapshot_download with no explicit dirs -> standard cache layout."""
        from modelscope_hub.compat.snapshot_download import snapshot_download

        mock_api = MagicMock()
        mock_api._config.cache_dir = Path("/default/cache")
        mock_api.download_repo.return_value = "/default/cache/models/org--model/snapshots/master"
        MockHubApi.return_value = mock_api

        result = snapshot_download(model_id="org/model")

        mock_api.download_repo.assert_called_once()
        call_kwargs = mock_api.download_repo.call_args[1]
        assert call_kwargs["cache_dir"] is None
        assert call_kwargs["local_dir"] is None
