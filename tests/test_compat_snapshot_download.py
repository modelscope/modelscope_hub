"""Test snapshot_download callback forwarding and read-only offline behavior.

Online tests stub only DownloadManager, exercising the real compat -> HubApi -> manager chain.
Offline tests use real temporary caches, forbid HubApi construction and requests, and bypass the online facade.
"""

from __future__ import annotations

from unittest import mock

import pytest
import requests

from modelscope_hub import ProgressCallback
from modelscope_hub._download import DownloadManager
from modelscope_hub.compat.file_download import model_file_download
from modelscope_hub.compat.snapshot_download import snapshot_download

_snapshot_download_mod = __import__("modelscope_hub.compat.snapshot_download", fromlist=["HubApi"])
_file_download_mod = __import__("modelscope_hub.compat.file_download", fromlist=["HubApi"])


class _DummyCallback(ProgressCallback):
    pass


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    from modelscope_hub import config

    monkeypatch.setattr(config, "_default_config", None)
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("MODELSCOPE_HOME", str(tmp_path / "auth"))
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "")


class TestSnapshotDownloadProgressCallbacks:
    def test_progress_callbacks_forwarded_through_facade(self):
        with mock.patch.object(DownloadManager, "download_repo", return_value="/tmp/snapshot") as m:
            result = snapshot_download(
                "owner/repo",
                progress_callbacks=[_DummyCallback],
                endpoint="https://unit.test",
                token="",
            )

        assert str(result) == "/tmp/snapshot"
        _, kwargs = m.call_args
        assert kwargs["progress_callbacks"] == [_DummyCallback]

    def test_progress_callbacks_default_none(self):
        with mock.patch.object(DownloadManager, "download_repo", return_value="/tmp/snapshot") as m:
            snapshot_download("owner/repo", endpoint="https://unit.test", token="")

        _, kwargs = m.call_args
        assert kwargs["progress_callbacks"] is None

    def test_snapshot_local_files_only_does_not_probe_endpoint(self, tmp_path):
        snapshot = tmp_path / "cache/models/owner--repo/snapshots/master"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        before = sorted(tmp_path.rglob("*"))
        with (
            mock.patch.object(_snapshot_download_mod, "HubApi", side_effect=AssertionError("离线不能创建客户端")),
            mock.patch.object(requests.Session, "send", side_effect=AssertionError("离线不能联网")),
        ):
            assert snapshot_download("owner/repo", local_files_only=True) == str(snapshot)
        assert sorted(tmp_path.rglob("*")) == before

    def test_file_local_files_only_does_not_probe_endpoint(self, tmp_path):
        snapshot = tmp_path / "cache/models/owner--repo/snapshots/master"
        snapshot.mkdir(parents=True)
        config_file = snapshot / "config.json"
        config_file.write_text("{}", encoding="utf-8")
        before = sorted(tmp_path.rglob("*"))
        with (
            mock.patch.object(_file_download_mod, "HubApi", side_effect=AssertionError("离线不能创建客户端")),
            mock.patch.object(requests.Session, "send", side_effect=AssertionError("离线不能联网")),
        ):
            assert model_file_download("owner/repo", "config.json", local_files_only=True) == str(config_file)
        assert sorted(tmp_path.rglob("*")) == before
