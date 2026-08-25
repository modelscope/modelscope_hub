"""Unit tests for the ``snapshot_download`` progress_callbacks chain.

These are network-free integration-style tests: only the *bottom* delegate
(``DownloadManager.download_repo``) is mocked, so the real compat wrapper and
the real ``HubApi.download_repo`` facade both execute. This guarantees the
whole ``compat -> HubApi facade -> DownloadManager`` forwarding chain is
exercised (a facade that drops ``progress_callbacks`` would fail here).
"""

from __future__ import annotations

from unittest import mock

from modelscope_hub import ProgressCallback
from modelscope_hub._download import DownloadManager
from modelscope_hub.compat.file_download import model_file_download
from modelscope_hub.compat.snapshot_download import snapshot_download

_snapshot_download_mod = __import__("modelscope_hub.compat.snapshot_download", fromlist=["HubApi"])
_file_download_mod = __import__("modelscope_hub.compat.file_download", fromlist=["HubApi"])


class _DummyCallback(ProgressCallback):
    pass


class TestSnapshotDownloadProgressCallbacks:
    def test_progress_callbacks_forwarded_through_facade(self):
        with mock.patch.object(DownloadManager, "download_repo", return_value="/tmp/snapshot") as m:
            result = snapshot_download(
                "owner/repo",
                progress_callbacks=[_DummyCallback],
                local_files_only=True,
            )

        assert str(result) == "/tmp/snapshot"
        _, kwargs = m.call_args
        assert kwargs["progress_callbacks"] == [_DummyCallback]

    def test_progress_callbacks_default_none(self):
        with mock.patch.object(DownloadManager, "download_repo", return_value="/tmp/snapshot") as m:
            snapshot_download("owner/repo", local_files_only=True)

        _, kwargs = m.call_args
        assert kwargs["progress_callbacks"] is None

    def test_snapshot_local_files_only_does_not_probe_endpoint(self):
        with (
            mock.patch.object(
                DownloadManager,
                "download_repo",
                return_value="/tmp/snapshot",
            ),
            mock.patch.object(
                _snapshot_download_mod.HubApi,
                "resolve_endpoint_for_read",
                side_effect=AssertionError("network should not be used"),
            ),
        ):
            assert snapshot_download("owner/repo", local_files_only=True) == "/tmp/snapshot"

    def test_file_local_files_only_does_not_probe_endpoint(self):
        with (
            mock.patch.object(
                DownloadManager,
                "download_file",
                return_value="/tmp/snapshot/config.json",
            ),
            mock.patch.object(
                _file_download_mod.HubApi,
                "resolve_endpoint_for_read",
                side_effect=AssertionError("network should not be used"),
            ),
        ):
            result = model_file_download("owner/repo", "config.json", local_files_only=True)
            assert result == "/tmp/snapshot/config.json"
