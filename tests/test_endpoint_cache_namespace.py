"""Cache entries must remain bound to the registry that supplied them."""

from __future__ import annotations

import hashlib
from pathlib import Path

from modelscope_hub._download import DownloadManager
from modelscope_hub.api import HubApi
from modelscope_hub.constants import DEFAULT_ENDPOINT
from modelscope_hub.types import FileInfo


def _replace_download_with_endpoint_marker(monkeypatch) -> None:
    def fake_download(
        self: DownloadManager,
        repo_id: str,
        repo_type: str,
        file_path: str,
        revision: str,
        target: Path,
        **kwargs,
    ) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self._client.endpoint)
        return target

    monkeypatch.setattr(DownloadManager, "_download_with_resume", fake_download)


def test_custom_endpoints_cannot_reuse_each_others_artifacts(tmp_path, monkeypatch):
    _replace_download_with_endpoint_marker(monkeypatch)
    first = HubApi(endpoint="https://registry-a.example")
    second = HubApi(endpoint="https://registry-b.example")

    first_path = first.download_file("owner/repo", "model", "config.json", cache_dir=tmp_path)
    second_path = second.download_file("owner/repo", "model", "config.json", cache_dir=tmp_path)

    assert first_path != second_path
    assert first_path.read_text() == "https://registry-a.example"
    assert second_path.read_text() == "https://registry-b.example"
    assert first_path.relative_to(tmp_path).parts[0] == "endpoints"
    assert second_path.relative_to(tmp_path).parts[0] == "endpoints"


def test_default_endpoint_keeps_existing_cache_layout(tmp_path, monkeypatch):
    _replace_download_with_endpoint_marker(monkeypatch)
    api = HubApi(endpoint=DEFAULT_ENDPOINT)

    path = api.download_file("owner/repo", "model", "config.json", cache_dir=tmp_path)

    assert path == tmp_path / "models" / "owner--repo" / "snapshots" / "master" / "config.json"


def test_custom_endpoint_does_not_reuse_unscoped_legacy_cache(tmp_path, monkeypatch):
    _replace_download_with_endpoint_marker(monkeypatch)
    legacy = tmp_path / "models" / "owner" / "repo"
    legacy.mkdir(parents=True)
    (legacy / "config.json").write_text("default-registry-bytes")
    api = HubApi(endpoint="https://registry.example")

    path = api.download_file("owner/repo", "model", "config.json", cache_dir=tmp_path)

    assert path != legacy / "config.json"
    assert path.read_text() == "https://registry.example"


def test_cache_management_is_scoped_to_current_endpoint(tmp_path, monkeypatch):
    _replace_download_with_endpoint_marker(monkeypatch)
    first = HubApi(endpoint="https://registry-a.example")
    second = HubApi(endpoint="https://registry-b.example")
    first_path = first.download_file("owner/repo", "model", "config.json", cache_dir=tmp_path)
    second_path = second.download_file("owner/repo", "model", "config.json", cache_dir=tmp_path)

    first_info = first.scan_cache(cache_dir=tmp_path)
    second_info = second.scan_cache(cache_dir=tmp_path)
    first.clear_cache(cache_dir=tmp_path, repo_type="model", repo_id="owner/repo")

    assert [repo.repo_id for repo in first_info.repos] == ["owner/repo"]
    assert [repo.repo_id for repo in second_info.repos] == ["owner/repo"]
    assert not first_path.exists()
    assert second_path.exists()


def test_cache_verification_uses_current_endpoint_namespace(tmp_path, monkeypatch):
    _replace_download_with_endpoint_marker(monkeypatch)
    api = HubApi(endpoint="https://registry.example")
    path = api.download_file("owner/repo", "model", "config.json", cache_dir=tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        api,
        "list_repo_files",
        lambda *args, **kwargs: [FileInfo(path="config.json", sha256=digest)],
    )

    result = api.verify_cache("owner/repo", "model", revision="master", cache_dir=tmp_path)

    assert result.verified_path == str(path.parent)
    assert result.checked_count == 1
    assert not result.mismatches
