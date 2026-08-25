"""Tests for the built-in ``studio`` command group."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from modelscope_hub.cli.studio import StudioCommand
from modelscope_hub.constants import RepoType

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
