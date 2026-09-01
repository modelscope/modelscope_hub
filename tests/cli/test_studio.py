"""Tests for the built-in ``studio`` command group."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from modelscope_hub.cli.studio import StudioCommand
from modelscope_hub.constants import RepoType
from modelscope_hub.types import PagedResult, RepoInfo

from .conftest import run_cli


# ===================================================================
# Parser tests
# ===================================================================
class TestStudioParser:
    def test_studio_group_is_registered(self, parser):
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["studio", "--help"])
        assert exc_info.value.code == 0

    def test_deploy(self, parser):
        args = parser.parse_args(["studio", "deploy", "org/demo"])
        assert args._command is StudioCommand
        assert args.studio_action == "deploy"
        assert args.studio_id == "org/demo"

    def test_stop(self, parser):
        args = parser.parse_args(["studio", "stop", "org/demo"])
        assert args.studio_action == "stop"
        assert args.studio_id == "org/demo"

    def test_logs_options(self, parser):
        args = parser.parse_args(
            [
                "studio",
                "logs",
                "org/demo",
                "--type",
                "build",
                "--keyword",
                "ERROR",
                "--page-num",
                "3",
                "--page-size",
                "50",
                "--start-timestamp",
                "10",
                "--end-timestamp",
                "20",
            ]
        )
        assert args.log_type == "build"
        assert args.keyword == "ERROR"
        assert args.page_num == 3
        assert args.page_size == 50
        assert args.start_timestamp == 10
        assert args.end_timestamp == 20

    def test_logs_log_type_alias(self, parser):
        args = parser.parse_args(["studio", "logs", "org/demo", "--log-type", "run"])
        assert args.log_type == "run"

    def test_settings_flags(self, parser):
        args = parser.parse_args(
            [
                "studio",
                "settings",
                "org/demo",
                "--display-name",
                "Demo",
                "--sdk-type",
                "gradio",
                "--private",
            ]
        )
        assert args.display_name == "Demo"
        assert args.sdk_type == "gradio"
        assert args.private is True

    def test_settings_key_value_tokens(self, parser):
        args = parser.parse_args(["studio", "settings", "org/demo", "hardware=cpu", "private=true"])
        assert args.settings == ["hardware=cpu", "private=true"]

    def test_group_level_auth(self, parser):
        args = parser.parse_args(["studio", "--token", "tk", "--endpoint", "https://x.cn", "deploy", "org/demo"])
        assert args.subcmd_token == "tk"
        assert args.subcmd_endpoint == "https://x.cn"

    def test_leaf_level_auth(self, parser):
        args = parser.parse_args(["studio", "deploy", "org/demo", "--token", "tk"])
        assert args.subcmd_token == "tk"

    def test_secret_add(self, parser):
        args = parser.parse_args(["studio", "secret", "add", "org/demo", "API_KEY", "value"])
        assert args.studio_action == "secret"
        assert args.secret_action == "add"
        assert args.studio_id == "org/demo"
        assert args.key == "API_KEY"
        assert args.value == "value"

    def test_list_filters(self, parser):
        args = parser.parse_args(
            [
                "studio",
                "list",
                "--search",
                "chat",
                "--owner",
                "alice",
                "--sort",
                "likes",
                "--status",
                "all",
                "--mcp-support",
                "--hardware-type",
                "xgpu",
                "--page",
                "2",
                "--page-size",
                "20",
            ]
        )
        assert args.studio_action == "list"
        assert args.search == "chat"
        assert args.owner == "alice"
        assert args.sort == "likes"
        assert args.status == "all"
        assert args.mcp_support is True
        assert args.hardware_type == "xgpu"
        assert args.page_number == 2
        assert args.page_size == 20

    def test_list_no_mcp_support_flag(self, parser):
        args = parser.parse_args(["studio", "list", "--no-mcp-support"])
        assert args.mcp_support is False

    def test_list_rejects_unknown_sort(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["studio", "list", "--sort", "downloads"])

    def test_variable_add(self, parser):
        args = parser.parse_args(["studio", "variable", "add", "org/demo", "MODEL", "Qwen"])
        assert args.studio_action == "variable"
        assert args.variable_action == "add"
        assert args.key == "MODEL"
        assert args.value == "Qwen"

    def test_variable_delete_has_confirmation_flag(self, parser):
        args = parser.parse_args(["studio", "variable", "delete", "org/demo", "MODEL", "--yes"])
        assert args.variable_action == "delete"
        assert args.yes is True

    def test_hardware_options(self, parser):
        args = parser.parse_args(["studio", "hardware", "--sdk-type", "gradio", "--studio", "org/demo"])
        assert args.studio_action == "hardware"
        assert args.sdk_type == "gradio"
        assert args.studio_id == "org/demo"

    def test_base_images(self, parser):
        args = parser.parse_args(["studio", "base-images"])
        assert args.studio_action == "base-images"

    def test_sdk_versions_defaults_to_gradio(self, parser):
        args = parser.parse_args(["studio", "sdk-versions"])
        assert args.studio_action == "sdk-versions"
        assert args.sdk_type == "gradio"

    def test_settings_visibility_flag(self, parser):
        args = parser.parse_args(["studio", "settings", "org/demo", "--visibility", "protected"])
        assert args.visibility == "protected"

    def test_settings_visibility_and_private_are_mutually_exclusive(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["studio", "settings", "org/demo", "--visibility", "public", "--private"])


# ===================================================================
# Execution tests
# ===================================================================
@pytest.mark.mock_only
class TestStudioExecute:
    def test_deploy(self, mock_api):
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "deploy", "org/demo"])
        assert code == 0
        assert "Deploy triggered" in out
        mock_api.deploy_repo.assert_called_once_with("org/demo", RepoType.STUDIO)

    def test_stop(self, mock_api):
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "stop", "org/demo"])
        assert code == 0
        assert "Stop triggered" in out
        mock_api.stop_repo.assert_called_once_with("org/demo", RepoType.STUDIO)

    def test_logs(self, mock_api, capsys):
        mock_api.get_repo_logs.return_value = {
            "logs": [
                {"timestamp": "t1", "content": "line1"},
                {"time": "t2", "message": "line2"},
                "line3",
            ],
            "total": 3,
        }
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(
                [
                    "studio",
                    "logs",
                    "org/demo",
                    "--type",
                    "build",
                    "--keyword",
                    "ERR",
                    "--page-num",
                    "2",
                    "--page-size",
                    "5",
                    "--start-timestamp",
                    "10",
                    "--end-timestamp",
                    "20",
                ]
            )
        assert code == 0
        assert "line1" in out
        assert "line2" in out
        assert "line3" in out
        mock_api.get_repo_logs.assert_called_once_with(
            "org/demo",
            RepoType.STUDIO,
            log_type="build",
            page_num=2,
            page_size=5,
            keyword="ERR",
            start_timestamp=10,
            end_timestamp=20,
        )

    def test_settings_flags_and_key_value_tokens(self, mock_api):
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(
                [
                    "studio",
                    "settings",
                    "org/demo",
                    "hardware=cpu",
                    "--display-name",
                    "Demo",
                    "--public",
                ]
            )
        assert code == 0
        assert "Updated settings" in out
        mock_api.update_repo_settings.assert_called_once_with(
            "org/demo",
            RepoType.STUDIO,
            hardware="cpu",
            display_name="Demo",
            private=False,
        )

    def test_settings_requires_at_least_one_setting(self, mock_api):
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "settings", "org/demo"])
        assert code == 2
        assert "No setting specified" in err
        mock_api.update_repo_settings.assert_not_called()

    def test_secret_list(self, mock_api):
        mock_api.list_secrets.return_value = [{"key": "API_KEY"}]
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "secret", "list", "org/demo"])
        assert code == 0
        assert "API_KEY" in out
        mock_api.list_secrets.assert_called_once_with("org/demo", RepoType.STUDIO)

    def test_secret_add(self, mock_api):
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "secret", "add", "org/demo", "API_KEY", "value"])
        assert code == 0
        assert "added" in out.lower()
        mock_api.add_secret.assert_called_once_with("org/demo", "API_KEY", "value", RepoType.STUDIO)

    def test_secret_update(self, mock_api):
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "secret", "update", "org/demo", "API_KEY", "new"])
        assert code == 0
        assert "updated" in out.lower()
        mock_api.update_secret.assert_called_once_with("org/demo", "API_KEY", "new", RepoType.STUDIO)

    def test_secret_delete(self, mock_api):
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "secret", "delete", "org/demo", "API_KEY"])
        assert code == 0
        assert "deleted" in out.lower()
        mock_api.delete_secret.assert_called_once_with("org/demo", "API_KEY", RepoType.STUDIO)

    # -- list ---------------------------------------------------------
    def test_list_renders_studio_columns(self, mock_api):
        mock_api.list_repos.return_value = PagedResult(
            items=[
                RepoInfo(
                    owner="alice",
                    name="demo",
                    sdk_type="gradio",
                    hardware="cpu-basic",
                    likes=7,
                    runtime={"status": "Running"},
                )
            ],
            total_count=1,
            page_number=1,
            page_size=10,
        )
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "list"])
        assert code == 0
        for expected in ("alice/demo", "gradio", "cpu-basic", "Running", "7"):
            assert expected in out

    def test_list_forwards_every_filter(self, mock_api):
        mock_api.list_repos.return_value = PagedResult(items=[], total_count=0)
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(
                ["studio", "list", "--owner", "alice", "--status", "all", "--hardware-type", "xgpu"]
            )
        assert code == 0
        assert "(no studios found)" in out
        kwargs = mock_api.list_repos.call_args.kwargs
        assert kwargs["owner"] == "alice"
        assert kwargs["status"] == "all"
        assert kwargs["hardware_type"] == "xgpu"

    def test_owner_listing_defaults_to_every_status(self, mock_api):
        """The endpoint keeps filtering to running spaces even with an owner set,
        so "list my spaces" would otherwise answer with nothing."""
        mock_api.list_repos.return_value = PagedResult(items=[], total_count=0)
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            run_cli(["studio", "list", "--owner", "alice"])
        assert mock_api.list_repos.call_args.kwargs["status"] == "all"

    def test_explicit_status_wins_over_the_owner_default(self, mock_api):
        mock_api.list_repos.return_value = PagedResult(items=[], total_count=0)
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            run_cli(["studio", "list", "--owner", "alice", "--status", "running"])
        assert mock_api.list_repos.call_args.kwargs["status"] == "running"

    def test_search_without_owner_leaves_the_status_to_the_server(self, mock_api):
        mock_api.list_repos.return_value = PagedResult(items=[], total_count=0)
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            run_cli(["studio", "list", "--search", "chat"])
        assert mock_api.list_repos.call_args.kwargs["status"] is None

    def test_list_shows_protected_visibility_verbatim(self, mock_api):
        """``protected`` has no integer equivalent, so it must survive as a string."""
        mock_api.list_repos.return_value = PagedResult(
            items=[RepoInfo(owner="alice", name="demo", visibility="protected")],
            total_count=1,
        )
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "list"])
        assert code == 0
        assert "protected" in out

    # -- variable -----------------------------------------------------
    def test_variable_list_shows_values(self, mock_api):
        """Unlike secrets, plaintext variable values are public and are printed."""
        mock_api.list_variables.return_value = [{"key": "MODEL", "value": "Qwen2.5-7B"}]
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "variable", "list", "org/demo"])
        assert code == 0
        assert "MODEL" in out
        assert "Qwen2.5-7B" in out
        mock_api.list_variables.assert_called_once_with("org/demo", RepoType.STUDIO)

    def test_variable_list_empty(self, mock_api):
        mock_api.list_variables.return_value = []
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "variable", "list", "org/demo"])
        assert code == 0
        assert "(no variables)" in out

    def test_variable_add(self, mock_api):
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "variable", "add", "org/demo", "MODEL", "Qwen"])
        assert code == 0
        assert "added" in out.lower()
        mock_api.add_variable.assert_called_once_with("org/demo", "MODEL", "Qwen", RepoType.STUDIO)

    def test_variable_update(self, mock_api):
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "variable", "update", "org/demo", "MODEL", "Qwen3"])
        assert code == 0
        assert "updated" in out.lower()
        mock_api.update_variable.assert_called_once_with("org/demo", "MODEL", "Qwen3", RepoType.STUDIO)

    def test_variable_delete_with_yes(self, mock_api):
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "variable", "delete", "org/demo", "MODEL", "--yes"])
        assert code == 0
        assert "deleted" in out.lower()
        mock_api.delete_variable.assert_called_once_with("org/demo", "MODEL", RepoType.STUDIO)

    def test_variable_delete_aborts_without_confirmation(self, mock_api):
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            with patch("builtins.input", return_value="n"):
                code, out, err = run_cli(["studio", "variable", "delete", "org/demo", "MODEL"])
        assert code == 0
        assert "Aborted" in out
        mock_api.delete_variable.assert_not_called()

    # -- resource discovery -------------------------------------------
    def test_hardware_table(self, mock_api):
        mock_api.list_studio_hardware.return_value = [
            {
                "name": "CPU basic",
                "instance_type": "ecs.c6.large",
                "resource_type": "cpu",
                "has_stock": True,
                "stock": 5,
            },
            {
                "name": "GPU A10",
                "instance_type": "ecs.gn7i",
                "resource_type": "gpu",
                "gpu_type": "A10",
                "has_stock": False,
                "cost_after_discount": 8,
                "original_cost": 10,
            },
        ]
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "hardware", "--sdk-type", "gradio"])
        assert code == 0
        assert "CPU basic" in out
        assert "ecs.gn7i" in out
        assert "out of stock" in out
        assert "8 (was 10)" in out
        assert "paid/" in out
        mock_api.list_studio_hardware.assert_called_once_with(sdk_type="gradio", repo_id=None)

    def test_hardware_scoped_to_a_studio(self, mock_api):
        mock_api.list_studio_hardware.return_value = []
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "hardware", "--studio", "org/demo"])
        assert code == 0
        mock_api.list_studio_hardware.assert_called_once_with(sdk_type=None, repo_id="org/demo")

    def test_base_images_table(self, mock_api):
        mock_api.list_studio_base_images.return_value = [{"name": "ubuntu22.04-py311", "tag": "1.0"}]
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "base-images"])
        assert code == 0
        assert "ubuntu22.04-py311" in out

    def test_sdk_versions_table(self, mock_api):
        mock_api.list_studio_sdk_versions.return_value = [{"sdk_type": "gradio", "version": "4.44.1", "tag": "latest"}]
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "sdk-versions"])
        assert code == 0
        assert "4.44.1" in out
        mock_api.list_studio_sdk_versions.assert_called_once_with(sdk_type="gradio")

    def test_sdk_versions_empty_names_the_sdk_type(self, mock_api):
        mock_api.list_studio_sdk_versions.return_value = []
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "sdk-versions", "--sdk-type", "docker"])
        assert code == 0
        assert "docker" in out

    # -- settings -----------------------------------------------------
    def test_settings_forwards_visibility(self, mock_api):
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "settings", "org/demo", "--visibility", "protected"])
        assert code == 0
        mock_api.update_repo_settings.assert_called_once_with("org/demo", RepoType.STUDIO, visibility="protected")

    def test_logs_footer_uses_total_count(self, mock_api):
        """The payload reports ``total_count``; reading ``total`` printed nothing."""
        mock_api.get_repo_logs.return_value = {
            "logs": ["line1"],
            "total_count": 42,
            "total_page_num": 5,
        }
        with patch("modelscope_hub.cli.studio.make_api", return_value=mock_api):
            code, out, err = run_cli(["studio", "logs", "org/demo"])
        assert code == 0
        assert "total 42" in out
        assert "5 page(s)" in out
