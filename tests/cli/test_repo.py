"""Tests for ``ms repo`` group — create / info / delete / list."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from modelscope_hub.errors import NotFoundError
from modelscope_hub.types import PagedResult, RepoInfo


class TestRepoCreate:
    """Verify repo create subcommand."""

    def test_repo_create_model(self, mock_api, run_cli):
        """Create a model repo successfully."""
        exit_code, out, err = run_cli(
            ["repo", "create", "owner/my-model", "--repo-type", "model"]
        )
        assert exit_code == 0
        assert "Created" in out
        mock_api.create_repo.assert_called_once_with(
            "owner/my-model",
            "model",
            visibility=None,
            license=None,
            chinese_name=None,
            description=None,
        )

    def test_repo_create_dataset(self, mock_api, run_cli):
        """Create a dataset repo successfully."""
        exit_code, out, err = run_cli(
            ["repo", "create", "owner/my-data", "--repo-type", "dataset"]
        )
        assert exit_code == 0
        mock_api.create_repo.assert_called_once()
        call_args = mock_api.create_repo.call_args
        assert call_args[0][1] == "dataset"

    def test_repo_create_studio(self, mock_api, run_cli):
        """Create a studio repo successfully."""
        exit_code, out, err = run_cli(
            ["repo", "create", "owner/my-studio", "--repo-type", "studio"]
        )
        assert exit_code == 0
        mock_api.create_repo.assert_called_once()

    def test_repo_create_with_all_options(self, mock_api, run_cli):
        """Create repo with visibility, license, description flags."""
        exit_code, out, err = run_cli([
            "repo", "create", "owner/full-repo",
            "--repo-type", "model",
            "--visibility", "private",
            "--license", "MIT",
            "--chinese-name", "测试模型",
            "--description", "A test model",
        ])
        assert exit_code == 0
        mock_api.create_repo.assert_called_once_with(
            "owner/full-repo",
            "model",
            visibility="private",
            license="MIT",
            chinese_name="测试模型",
            description="A test model",
        )

    def test_repo_create_missing_repo_type(self, run_cli):
        """Missing --repo-type exits with error."""
        exit_code, out, err = run_cli(
            ["repo", "create", "owner/no-type"]
        )
        assert exit_code == 2


class TestRepoInfo:
    """Verify repo info subcommand."""

    def test_repo_info_success(self, mock_api, run_cli):
        """repo info displays metadata."""
        exit_code, out, err = run_cli(
            ["repo", "info", "owner/my-model", "--repo-type", "model"]
        )
        assert exit_code == 0
        assert "test_owner/test_repo" in out
        assert "Apache-2.0" in out

    def test_repo_info_not_found(self, mock_api, run_cli):
        """repo info exits 1 on NotFoundError."""
        mock_api.get_repo.side_effect = NotFoundError("not found", status_code=404)
        exit_code, out, err = run_cli(
            ["repo", "info", "owner/ghost", "--repo-type", "model"]
        )
        assert exit_code == 1
        assert "not found" in err


class TestRepoDelete:
    """Verify repo delete subcommand."""

    def test_repo_delete_confirmed(self, mock_api, run_cli):
        """repo delete with 'y' confirmation succeeds."""
        with patch("builtins.input", return_value="y"):
            exit_code, out, err = run_cli(
                ["repo", "delete", "owner/old-model", "--repo-type", "model"]
            )
        assert exit_code == 0
        assert "Deleted" in out
        mock_api.delete_repo.assert_called_once_with("owner/old-model", "model")

    def test_repo_delete_cancelled(self, mock_api, run_cli):
        """repo delete with 'n' confirmation aborts."""
        with patch("builtins.input", return_value="n"):
            exit_code, out, err = run_cli(
                ["repo", "delete", "owner/keep-it", "--repo-type", "model"]
            )
        assert exit_code == 0
        assert "Aborted" in out
        mock_api.delete_repo.assert_not_called()

    def test_repo_delete_force_yes(self, mock_api, run_cli):
        """repo delete --yes skips confirmation."""
        exit_code, out, err = run_cli(
            ["repo", "delete", "owner/gone", "--repo-type", "model", "--yes"]
        )
        assert exit_code == 0
        assert "Deleted" in out
        mock_api.delete_repo.assert_called_once()


class TestRepoList:
    """Verify repo list subcommand."""

    def test_repo_list_success(self, mock_api, run_cli):
        """repo list displays table with repos."""
        exit_code, out, err = run_cli(
            ["repo", "list", "--repo-type", "model"]
        )
        assert exit_code == 0
        assert "test_owner/repo1" in out
        assert "test_owner/repo2" in out

    def test_repo_list_with_filters(self, mock_api, run_cli):
        """repo list passes owner, search, page, page-size to API."""
        exit_code, out, err = run_cli([
            "repo", "list",
            "--repo-type", "model",
            "--owner", "my_org",
            "--search", "bert",
            "--page", "2",
            "--page-size", "5",
        ])
        assert exit_code == 0
        mock_api.list_repos.assert_called_once_with(
            "model",
            owner="my_org",
            search="bert",
            page_number=2,
            page_size=5,
        )

    def test_repo_list_empty(self, mock_api, run_cli):
        """repo list with no results shows message."""
        mock_api.list_repos.return_value = PagedResult(
            items=[], total_count=0, page_number=1, page_size=10
        )
        exit_code, out, err = run_cli(
            ["repo", "list", "--repo-type", "model"]
        )
        assert exit_code == 0
        assert "no repositories found" in out.lower()
