from __future__ import annotations

from unittest import mock

from modelscope_hub.compat import LegacyHubApi
from modelscope_hub.constants import RepoType


def test_legacy_hub_api_preserves_delete_patterns_contract() -> None:
    api = LegacyHubApi(endpoint="https://modelscope.cn", token="test-token")
    expected = {
        "deleted_files": ["config.json"],
        "failed_files": [],
        "total_files": 1,
    }

    with mock.patch.object(api._api, "delete_files", return_value=expected) as delete_files:
        result = api.delete_files(
            repo_id="owner/repo",
            repo_type=RepoType.MODEL,
            delete_patterns="*.json",
            revision="main",
            commit_message="Delete JSON files",
        )

    delete_files.assert_called_once_with(
        "owner/repo",
        RepoType.MODEL,
        file_paths=None,
        delete_patterns="*.json",
        commit_message="Delete JSON files",
        revision="main",
    )
    assert result == expected


def test_legacy_hub_api_delete_files_supports_explicit_paths() -> None:
    api = LegacyHubApi(token="test-token")

    with mock.patch.object(api._api, "delete_files", return_value={}) as delete_files:
        api.delete_files(
            "owner/repo",
            repo_type="dataset",
            file_paths=["data/old.json"],
        )

    delete_files.assert_called_once_with(
        "owner/repo",
        "dataset",
        file_paths=["data/old.json"],
        delete_patterns=None,
        commit_message=None,
        revision="master",
    )
