"""Tests for ``ms deploy``, ``ms stop``, ``ms logs``, ``ms settings``.

Includes:
- Parser tests: all flags and repo-type choices
- Execution tests: mock HubApi to verify command logic
- Remote tests: real API lifecycle (existing)
"""
from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from modelscope_hub.cli.deploy import (
    DeployCommand,
    LogsCommand,
    SettingsCommand,
    StopCommand,
)

from .conftest import run_cli


# ===================================================================
# Parser tests
# ===================================================================
class TestDeployParser:
    """``ms deploy`` argument parsing."""

    def test_basic(self, parser):
        args = parser.parse_args(["deploy", "owner/studio"])
        assert args.repo_id == "owner/studio"

    def test_repo_type_default_studio(self, parser):
        args = parser.parse_args(["deploy", "o/r"])
        assert args.repo_type == "studio"

    @pytest.mark.parametrize("repo_type", ["studio", "mcp"])
    def test_repo_type_choices(self, parser, repo_type):
        args = parser.parse_args(["deploy", "o/r", "--repo-type", repo_type])
        assert args.repo_type == repo_type

    def test_invalid_repo_type_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["deploy", "o/r", "--repo-type", "model"])


class TestStopParser:
    """``ms stop`` argument parsing."""

    def test_basic(self, parser):
        args = parser.parse_args(["stop", "owner/studio"])
        assert args.repo_id == "owner/studio"

    def test_repo_type_default_studio(self, parser):
        args = parser.parse_args(["stop", "o/r"])
        assert args.repo_type == "studio"

    @pytest.mark.parametrize("repo_type", ["studio", "mcp"])
    def test_repo_type_choices(self, parser, repo_type):
        args = parser.parse_args(["stop", "o/r", "--repo-type", repo_type])
        assert args.repo_type == repo_type

    def test_invalid_repo_type_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["stop", "o/r", "--repo-type", "dataset"])


class TestLogsParser:
    """``ms logs`` argument parsing."""

    def test_basic(self, parser):
        args = parser.parse_args(["logs", "owner/studio"])
        assert args.repo_id == "owner/studio"

    def test_repo_type_default_studio(self, parser):
        args = parser.parse_args(["logs", "o/r"])
        assert args.repo_type == "studio"

    def test_log_type_run(self, parser):
        args = parser.parse_args(["logs", "o/r", "--log-type", "run"])
        assert args.log_type == "run"

    def test_log_type_build(self, parser):
        args = parser.parse_args(["logs", "o/r", "--log-type", "build"])
        assert args.log_type == "build"

    def test_log_type_default_run(self, parser):
        args = parser.parse_args(["logs", "o/r"])
        assert args.log_type == "run"

    def test_invalid_log_type_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["logs", "o/r", "--log-type", "deploy"])

    def test_page_flags(self, parser):
        args = parser.parse_args(["logs", "o/r", "--page", "3", "--page-size", "50"])
        assert args.page_num == 3
        assert args.page_size == 50

    def test_page_defaults(self, parser):
        args = parser.parse_args(["logs", "o/r"])
        assert args.page_num == 1
        assert args.page_size == 100

    def test_keyword(self, parser):
        args = parser.parse_args(["logs", "o/r", "--keyword", "ERROR"])
        assert args.keyword == "ERROR"

    def test_keyword_default_none(self, parser):
        args = parser.parse_args(["logs", "o/r"])
        assert args.keyword is None

    def test_all_options_combined(self, parser):
        args = parser.parse_args([
            "logs", "org/demo",
            "--log-type", "build",
            "--page", "2",
            "--page-size", "50",
            "--keyword", "Exception",
        ])
        assert args.repo_id == "org/demo"
        assert args.log_type == "build"
        assert args.page_num == 2
        assert args.page_size == 50
        assert args.keyword == "Exception"

    def test_invalid_repo_type_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["logs", "o/r", "--repo-type", "model"])

    def test_page_num_alias(self, parser):
        args = parser.parse_args(["logs", "o/r", "--page-num", "5"])
        assert args.page_num == 5


class TestSettingsParser:
    """``ms settings`` argument parsing."""

    def test_basic(self, parser):
        args = parser.parse_args(["settings", "owner/studio", "cpu=4"])
        assert args.repo_id == "owner/studio"
        assert args.settings == ["cpu=4"]

    def test_multiple_kv(self, parser):
        args = parser.parse_args(["settings", "o/r", "cpu=4", "memory=8192"])
        assert args.settings == ["cpu=4", "memory=8192"]

    def test_repo_type_default_studio(self, parser):
        args = parser.parse_args(["settings", "o/r", "x=1"])
        assert args.repo_type == "studio"

    @pytest.mark.parametrize("repo_type", ["studio", "skill"])
    def test_repo_type_choices(self, parser, repo_type):
        args = parser.parse_args(["settings", "o/r", "x=1", "--repo-type", repo_type])
        assert args.repo_type == repo_type

    def test_invalid_repo_type_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["settings", "o/r", "x=1", "--repo-type", "model"])

    def test_missing_settings_args_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["settings", "o/r"])


# ===================================================================
# Execution tests — mock HubApi
# ===================================================================
@pytest.mark.mock_only
class TestDeployExecute:
    """DeployCommand.execute() logic."""

    def test_deploy_studio(self, parser, mock_api, capsys):
        args = parser.parse_args(["deploy", "org/demo", "--repo-type", "studio"])
        with patch("modelscope_hub.cli.deploy.make_api", return_value=mock_api):
            DeployCommand(args).execute()
        mock_api.deploy_repo.assert_called_once_with("org/demo", "studio")
        out = capsys.readouterr().out
        assert "Deploy requested" in out

    def test_deploy_mcp(self, parser, mock_api, capsys):
        args = parser.parse_args(["deploy", "org/mcp-server", "--repo-type", "mcp"])
        with patch("modelscope_hub.cli.deploy.make_api", return_value=mock_api):
            DeployCommand(args).execute()
        mock_api.deploy_repo.assert_called_once_with("org/mcp-server", "mcp")


@pytest.mark.mock_only
class TestStopExecute:
    """StopCommand.execute() logic."""

    def test_stop_studio(self, parser, mock_api, capsys):
        args = parser.parse_args(["stop", "org/demo", "--repo-type", "studio"])
        with patch("modelscope_hub.cli.deploy.make_api", return_value=mock_api):
            StopCommand(args).execute()
        mock_api.stop_repo.assert_called_once_with("org/demo", "studio")
        out = capsys.readouterr().out
        assert "Stop requested" in out

    def test_stop_mcp(self, parser, mock_api, capsys):
        args = parser.parse_args(["stop", "org/mcp-server", "--repo-type", "mcp"])
        with patch("modelscope_hub.cli.deploy.make_api", return_value=mock_api):
            StopCommand(args).execute()
        mock_api.stop_repo.assert_called_once_with("org/mcp-server", "mcp")


@pytest.mark.mock_only
class TestLogsExecute:
    """LogsCommand.execute() logic."""

    def test_logs_with_list_payload(self, parser, mock_api, capsys):
        args = parser.parse_args(["logs", "org/demo"])
        with patch("modelscope_hub.cli.deploy.make_api", return_value=mock_api):
            LogsCommand(args).execute()
        mock_api.get_repo_logs.assert_called_once_with(
            "org/demo", "studio",
            log_type="run", page_num=1, page_size=100, keyword=None,
        )
        out = capsys.readouterr().out
        assert "line1" in out
        assert "line2" in out

    def test_logs_with_keyword(self, parser, mock_api, capsys):
        args = parser.parse_args(["logs", "org/demo", "--keyword", "ERROR"])
        with patch("modelscope_hub.cli.deploy.make_api", return_value=mock_api):
            LogsCommand(args).execute()
        assert mock_api.get_repo_logs.call_args.kwargs["keyword"] == "ERROR"

    def test_logs_build_type(self, parser, mock_api, capsys):
        args = parser.parse_args(["logs", "org/demo", "--log-type", "build"])
        with patch("modelscope_hub.cli.deploy.make_api", return_value=mock_api):
            LogsCommand(args).execute()
        assert mock_api.get_repo_logs.call_args.kwargs["log_type"] == "build"

    def test_logs_page_and_size_forwarded(self, parser, mock_api, capsys):
        args = parser.parse_args(["logs", "org/demo", "--page", "3", "--page-size", "50"])
        with patch("modelscope_hub.cli.deploy.make_api", return_value=mock_api):
            LogsCommand(args).execute()
        kw = mock_api.get_repo_logs.call_args.kwargs
        assert kw["page_num"] == 3
        assert kw["page_size"] == 50

    def test_logs_empty_payload(self, parser, mock_api, capsys):
        mock_api.get_repo_logs.return_value = {"logs": []}
        args = parser.parse_args(["logs", "org/demo"])
        with patch("modelscope_hub.cli.deploy.make_api", return_value=mock_api):
            LogsCommand(args).execute()


@pytest.mark.mock_only
class TestSettingsExecute:
    """SettingsCommand.execute() logic."""

    def test_single_setting(self, parser, mock_api, capsys):
        args = parser.parse_args(["settings", "org/demo", "cpu=4"])
        with patch("modelscope_hub.cli.deploy.make_api", return_value=mock_api):
            SettingsCommand(args).execute()
        mock_api.update_repo_settings.assert_called_once_with(
            "org/demo", "studio", cpu="4",
        )
        out = capsys.readouterr().out
        assert "Updated 1 setting" in out

    def test_multiple_settings(self, parser, mock_api, capsys):
        args = parser.parse_args(["settings", "org/demo", "cpu=4", "memory=8192"])
        with patch("modelscope_hub.cli.deploy.make_api", return_value=mock_api):
            SettingsCommand(args).execute()
        mock_api.update_repo_settings.assert_called_once_with(
            "org/demo", "studio", cpu="4", memory="8192",
        )
        out = capsys.readouterr().out
        assert "Updated 2 setting" in out

    def test_settings_skill_type(self, parser, mock_api, capsys):
        args = parser.parse_args([
            "settings", "org/skill1", "timeout=30", "--repo-type", "skill",
        ])
        with patch("modelscope_hub.cli.deploy.make_api", return_value=mock_api):
            SettingsCommand(args).execute()
        assert mock_api.update_repo_settings.call_args.args[1] == "skill"

    def test_settings_invalid_kv_raises(self, parser, mock_api):
        args = parser.parse_args(["settings", "org/demo", "bad"])
        with patch("modelscope_hub.cli.deploy.make_api", return_value=mock_api):
            with pytest.raises(ValueError, match="key=value"):
                SettingsCommand(args).execute()


# ===================================================================
# Remote integration tests (existing)
# ===================================================================
@pytest.mark.remote
class TestDeployLifecycle:
    """Test deploy/stop/logs/settings with real API on a studio repo."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_studio(self, api, test_owner, repo_name):
        """Create a studio for deploy testing."""
        cls = type(self)
        cls.repo_id = f"{test_owner}/{repo_name}_studio"
        api.create_repo(cls.repo_id, "studio", visibility="private")
        cls.api = api
        yield
        # Ensure stopped then deleted
        try:
            api.stop_repo(cls.repo_id, "studio")
        except Exception:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                api.delete_repo(cls.repo_id, "studio")
            except Exception:
                pass

    def test_01_deploy(self, test_token, test_endpoint):
        """Deploy the studio space."""
        exit_code, out, err = run_cli(
            ["deploy", self.repo_id, "--repo-type", "studio"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [deploy] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Deploy requested" in out

    def test_02_logs(self, test_token, test_endpoint):
        """Fetch run logs (may be empty or 404 while studio is starting)."""
        exit_code, out, err = run_cli(
            ["logs", self.repo_id],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [logs] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out[:300]!r}, err={err!r}")
        if exit_code != 0 and "启动中" in err:
            pytest.skip("Studio still starting — logs not yet available")
        assert exit_code == 0

    def test_03_settings(self, test_token, test_endpoint):
        """Update studio settings."""
        exit_code, out, err = run_cli(
            ["settings", self.repo_id, "cpu=2"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [settings] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Updated" in out

    def test_04_stop(self, test_token, test_endpoint):
        """Stop the studio space."""
        exit_code, out, err = run_cli(
            ["stop", self.repo_id, "--repo-type", "studio"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [stop] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Stop requested" in out
