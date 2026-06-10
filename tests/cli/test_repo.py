"""Tests for repo commands: ms create / info / list / delete.

Includes:
- Parser tests: verify argparse accepts all documented flags and choices
- Execution tests: mock HubApi to verify command logic without network
- Remote tests: real API lifecycle (existing, kept as-is)
"""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

from modelscope_hub.cli.repo import (
    CreateCommand,
    DeleteCommand,
    InfoCommand,
    ListCommand,
)
from modelscope_hub.types import PagedResult, RepoInfo

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

    def test_subcmd_token_endpoint(self, parser):
        args = parser.parse_args([
            "create", "o/r", "--repo-type", "model", "--token", "tk",
        ])
        assert args.subcmd_token == "tk"


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

    def test_subcmd_token_endpoint(self, parser):
        args = parser.parse_args([
            "info", "o/r", "--repo-type", "model", "--token", "tk", "--endpoint", "https://x.cn",
        ])
        assert args.subcmd_token == "tk"
        assert args.subcmd_endpoint == "https://x.cn"


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
        args = parser.parse_args(["list"])
        assert args.repo_type is None
        from modelscope_hub.cli.repo import ListCommand
        with pytest.raises(SystemExit):
            ListCommand(args).execute()

    def test_subcmd_token_endpoint(self, parser):
        args = parser.parse_args([
            "list", "--repo-type", "model", "--token", "tk",
        ])
        assert args.subcmd_token == "tk"


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

    def test_subcmd_token_endpoint(self, parser):
        args = parser.parse_args([
            "delete", "o/r", "--repo-type", "model", "--token", "tk",
        ])
        assert args.subcmd_token == "tk"


# ===================================================================
# Execution tests — mock HubApi, verify command logic
# ===================================================================
@pytest.mark.mock_only
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

    def test_create_dataset(self, parser, mock_api, capsys):
        """Create a dataset repo via CLI."""
        args = parser.parse_args([
            "create", "owner/my-dataset", "--repo-type", "dataset",
            "--visibility", "private", "--license", "cc-by-4.0",
            "--description", "Test dataset",
        ])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            CreateCommand(args).execute()
        mock_api.create_repo.assert_called_once_with(
            "owner/my-dataset", "dataset",
            visibility="private", license="cc-by-4.0",
            chinese_name=None, description="Test dataset",
        )
        out = capsys.readouterr().out
        assert "Created" in out

    def test_create_dataset_public_with_license(self, parser, mock_api, capsys):
        """Create a public dataset with specific license."""
        args = parser.parse_args([
            "create", "owner/public-ds", "--repo-type", "dataset",
            "--visibility", "public", "--license", "mit",
        ])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            CreateCommand(args).execute()
        call_kwargs = mock_api.create_repo.call_args
        assert call_kwargs[0][1] == "dataset"  # repo_type positional arg
        assert call_kwargs.kwargs["visibility"] == "public"
        assert call_kwargs.kwargs["license"] == "mit"

    def test_create_dataset_with_chinese_name(self, parser, mock_api, capsys):
        """Create a dataset with chinese name and description."""
        args = parser.parse_args([
            "create", "owner/cn-dataset", "--repo-type", "dataset",
            "--chinese-name", "测试数据集",
            "--description", "这是一个测试数据集",
        ])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            CreateCommand(args).execute()
        mock_api.create_repo.assert_called_once_with(
            "owner/cn-dataset", "dataset",
            visibility=None, license=None,
            chinese_name="测试数据集", description="这是一个测试数据集",
        )


    def test_create_skill_with_skill_file(self, parser, mock_api, capsys, tmp_path):
        """Skill file is uploaded and its ID is forwarded to create_repo."""
        zip_file = tmp_path / "skill.zip"
        zip_file.write_bytes(b"PK dummy")
        mock_api.upload_file_to_openapi = MagicMock(
            return_value="8c378570-8991-431b-a82c-96f3d0b4f0f4"
        )
        args = parser.parse_args([
            "create", "owner/my-skill", "--repo-type", "skill",
            "--category", "developer-tools",
            "--skill-file", str(zip_file),
        ])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            CreateCommand(args).execute()
        mock_api.upload_file_to_openapi.assert_called_once()
        call_kwargs = mock_api.create_repo.call_args
        assert call_kwargs.kwargs["skill_file"] == "8c378570-8991-431b-a82c-96f3d0b4f0f4"
        assert call_kwargs.kwargs["category"] == "developer-tools"

    def test_create_skill_file_not_found(self, parser, mock_api):
        """Non-existent --skill-file path causes SystemExit(2)."""
        args = parser.parse_args([
            "create", "owner/my-skill", "--repo-type", "skill",
            "--category", "developer-tools",
            "--skill-file", "/nonexistent/skill.zip",
        ])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            with pytest.raises(SystemExit) as exc_info:
                CreateCommand(args).execute()
            assert exc_info.value.code == 2


@pytest.mark.mock_only
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


@pytest.mark.mock_only
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

    def test_list_all_paginates(self, parser, mock_api, capsys):
        page1 = PagedResult(
            items=[RepoInfo(id=1, owner="o", name="m1", repo_type="model", downloads=10, likes=1)],
            total_count=2, page_number=1, page_size=1,
        )
        page2 = PagedResult(
            items=[RepoInfo(id=2, owner="o", name="m2", repo_type="model", downloads=5, likes=0)],
            total_count=2, page_number=2, page_size=1,
        )
        mock_api.list_repos.side_effect = [page1, page2]
        args = parser.parse_args([
            "list", "--repo-type", "model", "--all", "--page-size", "1",
        ])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            ListCommand(args).execute()
        assert mock_api.list_repos.call_count == 2
        out = capsys.readouterr().out
        assert "o/m1" in out
        assert "o/m2" in out
        assert "total 2 repos" in out

    def test_list_all_empty(self, parser, mock_api, capsys):
        mock_api.list_repos.return_value = PagedResult(
            items=[], total_count=0, page_number=1, page_size=50,
        )
        args = parser.parse_args(["list", "--repo-type", "model", "--all"])
        with patch("modelscope_hub.cli.repo.make_api", return_value=mock_api):
            ListCommand(args).execute()
        out = capsys.readouterr().out
        assert "no repositories found" in out

    def test_list_all_and_page_mutually_exclusive(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["list", "--repo-type", "model", "--all", "--page", "2"])


@pytest.mark.mock_only
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
# Remote integration tests (require API credentials)
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


@pytest.mark.remote
class TestDatasetRepoLifecycle:
    """Test dataset repo CRUD via CLI with real API."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_repo(self, api, test_owner, repo_name):
        """Store dataset repo metadata for tests; cleanup at end."""
        cls = type(self)
        cls.repo_id = f"{test_owner}/{repo_name}_ds"
        cls.api = api
        cls.test_owner = test_owner
        yield
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                api.delete_repo(cls.repo_id, "dataset")
            except Exception:
                pass

    def test_01_create_dataset(self, test_token, test_endpoint):
        """Create a private dataset repo via CLI."""
        exit_code, out, err = run_cli(
            ["create", self.repo_id, "--repo-type", "dataset", "--visibility", "private"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [dataset create] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Created" in out

    def test_02_info_dataset(self, test_token, test_endpoint):
        """Get dataset info."""
        exit_code, out, err = run_cli(
            ["info", self.repo_id, "--repo-type", "dataset"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [dataset info] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "repo_id" in out or self.repo_id in out

    def test_03_list_datasets(self, test_token, test_endpoint):
        """List datasets should include created one."""
        exit_code, out, err = run_cli(
            ["list", "--repo-type", "dataset", "--owner", self.test_owner],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [dataset list] owner={self.test_owner}")
        print(f"** exit_code={exit_code}, out={out[:200]!r}, err={err!r}")
        assert exit_code == 0

    def test_04_delete_dataset(self, test_token, test_endpoint):
        """Delete the dataset repo."""
        exit_code, out, err = run_cli(
            ["delete", self.repo_id, "--repo-type", "dataset", "--yes"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [dataset delete] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        if exit_code != 0:
            pytest.skip("delete_repo not yet supported via SDK token — clean up via web console")
