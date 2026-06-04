"""Tests for repo commands: ms create / info / list / delete.

Includes:
- Parser tests: verify argparse accepts all documented flags and choices
- Execution tests: mock HubApi to verify command logic without network
- Remote tests: real API lifecycle (existing, kept as-is)
"""
from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from modelscope_hub.cli.repo import (
    CreateCommand,
    DeleteCommand,
    InfoCommand,
    ListCommand,
)
from modelscope_hub.types import PagedResult

from .conftest import run_cli


# ===================================================================
# Parser tests — no network, no mocks, pure argparse
# ===================================================================
class TestCreateParser:
    """``ms create`` argument parsing."""

    def test_basic(self, parser):
        args = parser.parse_args(["create", "owner/repo", "--repo-type", "model"])
        assert args.repo_id == "owner/repo"
        assert args.repo_type == "model"

    @pytest.mark.parametrize("repo_type", ["model", "dataset", "studio", "skill"])
    def test_all_repo_types(self, parser, repo_type):
        args = parser.parse_args(["create", "o/r", "--repo-type", repo_type])
        assert args.repo_type == repo_type

    def test_invalid_repo_type_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["create", "o/r", "--repo-type", "mcp"])

    @pytest.mark.parametrize("vis", ["public", "private", "internal"])
    def test_visibility_choices(self, parser, vis):
        args = parser.parse_args(
            ["create", "o/r", "--repo-type", "model", "--visibility", vis]
        )
        assert args.visibility == vis

    def test_invalid_visibility_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["create", "o/r", "--repo-type", "model", "--visibility", "secret"]
            )

    def test_license_flag(self, parser):
        args = parser.parse_args(
            ["create", "o/r", "--repo-type", "model", "--license", "apache-2.0"]
        )
        assert args.license == "apache-2.0"

    def test_chinese_name(self, parser):
        args = parser.parse_args(
            ["create", "o/r", "--repo-type", "model", "--chinese-name", "测试模型"]
        )
        assert args.chinese_name == "测试模型"

    def test_chinese_name_underscore(self, parser):
        args = parser.parse_args(
            ["create", "o/r", "--repo-type", "model", "--chinese_name", "测试"]
        )
        assert args.chinese_name == "测试"

    def test_description(self, parser):
        args = parser.parse_args(
            ["create", "o/r", "--repo-type", "model", "--description", "A test model"]
        )
        assert args.description == "A test model"

    def test_exist_ok_flag(self, parser):
        args = parser.parse_args(
            ["create", "o/r", "--repo-type", "model", "--exist-ok"]
        )
        assert args.exist_ok is True

    def test_exist_ok_underscore(self, parser):
        args = parser.parse_args(
            ["create", "o/r", "--repo-type", "model", "--exist_ok"]
        )
        assert args.exist_ok is True

    def test_exist_ok_default_false(self, parser):
        args = parser.parse_args(["create", "o/r", "--repo-type", "model"])
        assert args.exist_ok is False

    def test_missing_repo_type_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["create", "o/r"])

    @pytest.mark.parametrize("sdk", ["gradio", "streamlit", "docker", "static"])
    def test_studio_sdk_type(self, parser, sdk):
        args = parser.parse_args(
            ["create", "o/r", "--repo-type", "studio", "--sdk-type", sdk]
        )
        assert args.sdk_type == sdk

    def test_invalid_sdk_type_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["create", "o/r", "--repo-type", "studio", "--sdk-type", "flask"]
            )

    def test_studio_sdk_version(self, parser):
        args = parser.parse_args(
            ["create", "o/r", "--repo-type", "studio", "--sdk-version", "4.0"]
        )
        assert args.sdk_version == "4.0"

    def test_studio_base_image(self, parser):
        args = parser.parse_args(
            ["create", "o/r", "--repo-type", "studio", "--base-image", "python:3.11"]
        )
        assert args.base_image == "python:3.11"

    def test_studio_cover_image(self, parser):
        args = parser.parse_args(
            ["create", "o/r", "--repo-type", "studio", "--cover-image", "https://img.png"]
        )
        assert args.cover_image == "https://img.png"

    def test_studio_hardware(self, parser):
        args = parser.parse_args(
            ["create", "o/r", "--repo-type", "studio", "--hardware", "gpu.a10"]
        )
        assert args.hardware == "gpu.a10"

    def test_all_studio_options_combined(self, parser):
        args = parser.parse_args([
            "create", "o/studio1", "--repo-type", "studio",
            "--visibility", "private",
            "--sdk-type", "gradio", "--sdk-version", "4.0",
            "--base-image", "python:3.11",
            "--cover-image", "https://img.png",
            "--hardware", "gpu.a10",
            "--license", "mit",
            "--description", "demo",
            "--chinese-name", "演示",
        ])
        assert args.repo_type == "studio"
        assert args.sdk_type == "gradio"
        assert args.sdk_version == "4.0"
        assert args.hardware == "gpu.a10"


class TestInfoParser:
    """``ms info`` argument parsing."""

    def test_basic(self, parser):
        args = parser.parse_args(["info", "owner/repo", "--repo-type", "model"])
        assert args.repo_id == "owner/repo"
        assert args.repo_type == "model"

    @pytest.mark.parametrize("repo_type", ["model", "dataset", "studio", "skill", "mcp"])
    def test_all_repo_types(self, parser, repo_type):
        args = parser.parse_args(["info", "o/r", "--repo-type", repo_type])
        assert args.repo_type == repo_type

    def test_repo_type_required(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["info", "o/r"])


class TestListParser:
    """``ms list`` argument parsing."""

    @pytest.mark.parametrize("repo_type", ["model", "dataset", "skill", "mcp"])
    def test_all_repo_types(self, parser, repo_type):
        args = parser.parse_args(["list", "--repo-type", repo_type])
        assert args.repo_type == repo_type

    def test_invalid_repo_type_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["list", "--repo-type", "studio"])

    def test_owner_flag(self, parser):
        args = parser.parse_args(["list", "--repo-type", "model", "--owner", "my-org"])
        assert args.owner == "my-org"

    def test_search_flag(self, parser):
        args = parser.parse_args(["list", "--repo-type", "model", "--search", "qwen"])
        assert args.search == "qwen"

    def test_page_flag(self, parser):
        args = parser.parse_args(["list", "--repo-type", "model", "--page", "3"])
        assert args.page_number == 3

    def test_page_size_flag(self, parser):
        args = parser.parse_args(["list", "--repo-type", "model", "--page-size", "20"])
        assert args.page_size == 20

    def test_defaults(self, parser):
        args = parser.parse_args(["list", "--repo-type", "model"])
        assert args.owner is None
        assert args.search is None
        assert args.page_number == 1
        assert args.page_size == 10

    def test_missing_repo_type_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["list"])


class TestDeleteParser:
    """``ms delete`` argument parsing."""

    def test_basic(self, parser):
        args = parser.parse_args(["delete", "owner/repo", "--repo-type", "model"])
        assert args.repo_id == "owner/repo"
        assert args.repo_type == "model"

    @pytest.mark.parametrize("repo_type", ["model", "dataset"])
    def test_valid_repo_types(self, parser, repo_type):
        args = parser.parse_args(["delete", "o/r", "--repo-type", repo_type])
        assert args.repo_type == repo_type

    def test_invalid_repo_type_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["delete", "o/r", "--repo-type", "studio"])

    def test_yes_flag(self, parser):
        args = parser.parse_args(["delete", "o/r", "--repo-type", "model", "--yes"])
        assert args.yes is True

    def test_yes_short_flag(self, parser):
        args = parser.parse_args(["delete", "o/r", "--repo-type", "model", "-y"])
        assert args.yes is True

    def test_yes_default_false(self, parser):
        args = parser.parse_args(["delete", "o/r", "--repo-type", "model"])
        assert args.yes is False


# ===================================================================
# Execution tests — mock HubApi, verify command logic
# ===================================================================
class TestCreateExecute:
    """CreateCommand.execute() logic."""

    def test_create_model(self, parser, mock_api, capsys):
        args = parser.parse_args([
            "create", "owner/my-model", "--repo-type", "model",
            "--visibility", "private", "--license", "apache-2.0",
        ])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            CreateCommand(args).execute()
        mock_api.create_repo.assert_called_once_with(
            "owner/my-model", "model",
            visibility="private", license="apache-2.0",
            chinese_name=None, description=None,
        )
        out = capsys.readouterr().out
        assert "Created" in out

    def test_create_studio_with_extras(self, parser, mock_api, capsys):
        args = parser.parse_args([
            "create", "owner/demo", "--repo-type", "studio",
            "--sdk-type", "gradio", "--sdk-version", "4.0",
            "--hardware", "gpu.a10",
        ])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            CreateCommand(args).execute()
        call_kwargs = mock_api.create_repo.call_args
        assert call_kwargs.kwargs["sdk_type"] == "gradio"
        assert call_kwargs.kwargs["sdk_version"] == "4.0"
        assert call_kwargs.kwargs["hardware"] == "gpu.a10"

    def test_exist_ok_swallows_exist_error(self, parser, mock_api, capsys):
        mock_api.create_repo.side_effect = Exception("Repository already exists")
        args = parser.parse_args([
            "create", "owner/repo", "--repo-type", "model", "--exist-ok",
        ])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            CreateCommand(args).execute()
        out = capsys.readouterr().out
        assert "already exists" in out

    def test_exist_ok_reraises_non_exist_error(self, parser, mock_api):
        mock_api.create_repo.side_effect = Exception("Permission denied")
        args = parser.parse_args([
            "create", "owner/repo", "--repo-type", "model", "--exist-ok",
        ])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            with pytest.raises(Exception, match="Permission denied"):
                CreateCommand(args).execute()

    def test_create_without_exist_ok_raises(self, parser, mock_api):
        mock_api.create_repo.side_effect = Exception("Repository already exists")
        args = parser.parse_args([
            "create", "owner/repo", "--repo-type", "model",
        ])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            with pytest.raises(Exception, match="already exists"):
                CreateCommand(args).execute()

    def test_chinese_name_and_description_forwarded(self, parser, mock_api, capsys):
        args = parser.parse_args([
            "create", "owner/repo", "--repo-type", "model",
            "--chinese-name", "测试模型", "--description", "A test model",
        ])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            CreateCommand(args).execute()
        call_kwargs = mock_api.create_repo.call_args
        assert call_kwargs.kwargs["chinese_name"] == "测试模型"
        assert call_kwargs.kwargs["description"] == "A test model"

    def test_base_image_and_cover_image_forwarded(self, parser, mock_api, capsys):
        args = parser.parse_args([
            "create", "owner/studio1", "--repo-type", "studio",
            "--base-image", "python:3.11", "--cover-image", "https://img.png",
        ])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            CreateCommand(args).execute()
        call_kwargs = mock_api.create_repo.call_args
        assert call_kwargs.kwargs["base_image"] == "python:3.11"
        assert call_kwargs.kwargs["cover_image"] == "https://img.png"


class TestInfoExecute:
    """InfoCommand.execute() logic."""

    def test_info_prints_repo(self, parser, mock_api, capsys):
        args = parser.parse_args(["info", "owner/repo", "--repo-type", "model"])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            InfoCommand(args).execute()
        mock_api.get_repo.assert_called_once_with("owner/repo", "model")
        out = capsys.readouterr().out
        assert "repo_type" in out
        assert "license" in out


class TestListExecute:
    """ListCommand.execute() logic."""

    def test_list_with_results(self, parser, mock_api, capsys):
        args = parser.parse_args([
            "list", "--repo-type", "model", "--owner", "org", "--page-size", "20",
        ])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            ListCommand(args).execute()
        mock_api.list_repos.assert_called_once_with(
            "model", owner="org", search=None, page_number=1, page_size=20,
        )
        out = capsys.readouterr().out
        assert "owner/model1" in out

    def test_list_empty(self, parser, mock_api, capsys):
        mock_api.list_repos.return_value = PagedResult(
            items=[], total_count=0, page_number=1, page_size=10,
        )
        args = parser.parse_args(["list", "--repo-type", "model"])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            ListCommand(args).execute()
        out = capsys.readouterr().out
        assert "no repositories found" in out

    def test_list_with_search(self, parser, mock_api, capsys):
        args = parser.parse_args([
            "list", "--repo-type", "dataset", "--search", "qwen",
        ])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            ListCommand(args).execute()
        mock_api.list_repos.assert_called_once_with(
            "dataset", owner=None, search="qwen", page_number=1, page_size=10,
        )


class TestDeleteExecute:
    """DeleteCommand.execute() logic."""

    def test_delete_with_yes(self, parser, mock_api, capsys):
        args = parser.parse_args([
            "delete", "owner/repo", "--repo-type", "model", "--yes",
        ])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            DeleteCommand(args).execute()
        mock_api.delete_repo.assert_called_once_with("owner/repo", "model")
        out = capsys.readouterr().out
        assert "Deleted" in out

    def test_delete_aborted(self, parser, mock_api, capsys):
        args = parser.parse_args(["delete", "owner/repo", "--repo-type", "model"])
        with (
            patch("modelscope_hub.cli.repo.make_api", return_value=mock_api),
            patch("builtins.input", return_value="n"),
        ):
            DeleteCommand(args).execute()
        mock_api.delete_repo.assert_not_called()
        out = capsys.readouterr().out
        assert "Aborted" in out

    def test_delete_confirmed_interactively(self, parser, mock_api, capsys):
        args = parser.parse_args(["delete", "owner/repo", "--repo-type", "model"])
        with (
            patch("modelscope_hub.cli.repo.make_api", return_value=mock_api),
            patch("builtins.input", return_value="y"),
        ):
            DeleteCommand(args).execute()
        mock_api.delete_repo.assert_called_once()


# ===================================================================
# Backward compat: ``ms repo <action>``
# ===================================================================
class TestRepoCompat:
    """Verify ``ms repo create/info/list/delete`` still works."""

    def test_repo_create(self, parser):
        args = parser.parse_args([
            "repo", "create", "o/r", "--repo-type", "model", "--exist-ok",
        ])
        assert args.repo_id == "o/r"
        assert args.exist_ok is True

    def test_repo_info(self, parser):
        args = parser.parse_args(["repo", "info", "o/r", "--repo-type", "model"])
        assert args.repo_id == "o/r"

    def test_repo_list(self, parser):
        args = parser.parse_args(["repo", "list", "--repo-type", "model"])
        assert args.repo_type == "model"

    def test_repo_delete(self, parser):
        args = parser.parse_args(["repo", "delete", "o/r", "--repo-type", "model", "--yes"])
        assert args.repo_id == "o/r"
        assert args.yes is True


# ===================================================================
# Remote integration tests (existing — require API credentials)
# ===================================================================
@pytest.mark.remote
class TestRepoLifecycle:
    """Test repo CRUD operations with real API."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_repo(self, api, test_owner, repo_name):
        """Store repo metadata for tests; cleanup at end."""
        cls = type(self)
        cls.repo_id = f"{test_owner}/{repo_name}"
        cls.api = api
        yield
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                api.delete_repo(cls.repo_id, "model")
            except Exception:
                pass

    def test_01_create_repo(self, test_token, test_endpoint):
        """Create a private model repo."""
        exit_code, out, err = run_cli(
            ["create", self.repo_id, "--repo-type", "model", "--visibility", "private"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [repo create] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Created" in out

    def test_02_repo_info(self, test_token, test_endpoint):
        """Get repo info shows metadata."""
        exit_code, out, err = run_cli(
            ["info", self.repo_id, "--repo-type", "model"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [repo info] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "repo_id" in out or self.repo_id in out

    def test_03_repo_list(self, test_token, test_endpoint, test_owner):
        """List repos for the test owner."""
        exit_code, out, err = run_cli(
            ["list", "--repo-type", "model", "--owner", test_owner],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [repo list] owner={test_owner}")
        print(f"** exit_code={exit_code}, out={out[:200]!r}, err={err!r}")
        assert exit_code == 0

    def test_04_delete_repo(self, test_token, test_endpoint):
        """Delete the created repo (currently deprecated — expected to fail)."""
        exit_code, out, err = run_cli(
            ["delete", self.repo_id, "--repo-type", "model", "--yes"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [repo delete] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        # delete_repo is deprecated; hub rejects token-based deletion for now
        if exit_code != 0:
            pytest.skip("delete_repo not yet supported via SDK token — clean up via web console")
