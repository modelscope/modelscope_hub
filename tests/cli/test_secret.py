"""Tests for ``ms secret`` group — add / list / update / delete.

Includes:
- Parser tests: all subcommands and flags
- Execution tests: mock HubApi for secret CRUD logic
- Remote tests: real API lifecycle (existing)
"""
from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from modelscope_hub.cli.secret import (
    _SecretAdd,
    _SecretDelete,
    _SecretList,
    _SecretUpdate,
)

from .conftest import run_cli


# ===================================================================
# Parser tests
# ===================================================================
class TestSecretAddParser:
    """``ms secret add`` argument parsing."""

    def test_basic(self, parser):
        args = parser.parse_args(["secret", "add", "org/demo", "API_KEY", "sk-xxx"])
        assert args.repo_id == "org/demo"
        assert args.key == "API_KEY"
        assert args.value == "sk-xxx"

    def test_repo_type_default_studio(self, parser):
        args = parser.parse_args(["secret", "add", "o/r", "K", "V"])
        assert args.repo_type == "studio"

    def test_missing_value_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["secret", "add", "o/r", "K"])

    def test_missing_key_and_value_exits(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["secret", "add", "o/r"])

    def test_explicit_repo_type(self, parser):
        args = parser.parse_args([
            "secret", "add", "o/r", "K", "V", "--repo-type", "studio",
        ])
        assert args.repo_type == "studio"


class TestSecretListParser:
    """``ms secret list`` argument parsing."""

    def test_basic(self, parser):
        args = parser.parse_args(["secret", "list", "org/demo"])
        assert args.repo_id == "org/demo"

    def test_repo_type_default_studio(self, parser):
        args = parser.parse_args(["secret", "list", "o/r"])
        assert args.repo_type == "studio"


class TestSecretUpdateParser:
    """``ms secret update`` argument parsing."""

    def test_basic(self, parser):
        args = parser.parse_args(["secret", "update", "org/demo", "KEY", "new_val"])
        assert args.repo_id == "org/demo"
        assert args.key == "KEY"
        assert args.value == "new_val"


class TestSecretDeleteParser:
    """``ms secret delete`` argument parsing."""

    def test_basic(self, parser):
        args = parser.parse_args(["secret", "delete", "org/demo", "KEY"])
        assert args.repo_id == "org/demo"
        assert args.key == "KEY"

    def test_yes_flag(self, parser):
        args = parser.parse_args(["secret", "delete", "o/r", "K", "--yes"])
        assert args.yes is True

    def test_yes_short(self, parser):
        args = parser.parse_args(["secret", "delete", "o/r", "K", "-y"])
        assert args.yes is True

    def test_yes_default_false(self, parser):
        args = parser.parse_args(["secret", "delete", "o/r", "K"])
        assert args.yes is False


# ===================================================================
# Execution tests — mock HubApi
# ===================================================================
class TestSecretAddExecute:
    def test_add_secret(self, parser, mock_api, capsys):
        args = parser.parse_args(["secret", "add", "org/demo", "API_KEY", "sk-xxx"])
        with patch("modelscope_hub.cli.secret.make_api", return_value=mock_api):
            _SecretAdd(args).execute()
        mock_api.add_secret.assert_called_once_with("org/demo", "API_KEY", "sk-xxx", "studio")
        out = capsys.readouterr().out
        assert "Added" in out
        assert "API_KEY" in out


class TestSecretListExecute:
    def test_list_with_secrets(self, parser, mock_api, capsys):
        args = parser.parse_args(["secret", "list", "org/demo"])
        with patch("modelscope_hub.cli.secret.make_api", return_value=mock_api):
            _SecretList(args).execute()
        mock_api.list_secrets.assert_called_once_with("org/demo", "studio")
        out = capsys.readouterr().out
        assert "API_KEY" in out

    def test_list_empty(self, parser, mock_api, capsys):
        mock_api.list_secrets.return_value = []
        args = parser.parse_args(["secret", "list", "org/demo"])
        with patch("modelscope_hub.cli.secret.make_api", return_value=mock_api):
            _SecretList(args).execute()
        out = capsys.readouterr().out
        assert "no secrets" in out


class TestSecretUpdateExecute:
    def test_update_secret(self, parser, mock_api, capsys):
        args = parser.parse_args(["secret", "update", "org/demo", "KEY", "new_val"])
        with patch("modelscope_hub.cli.secret.make_api", return_value=mock_api):
            _SecretUpdate(args).execute()
        mock_api.update_secret.assert_called_once_with("org/demo", "KEY", "new_val", "studio")
        out = capsys.readouterr().out
        assert "Updated" in out


class TestSecretDeleteExecute:
    def test_delete_with_yes(self, parser, mock_api, capsys):
        args = parser.parse_args(["secret", "delete", "org/demo", "KEY", "--yes"])
        with patch("modelscope_hub.cli.secret.make_api", return_value=mock_api):
            _SecretDelete(args).execute()
        mock_api.delete_secret.assert_called_once_with("org/demo", "KEY", "studio")
        out = capsys.readouterr().out
        assert "Deleted" in out

    def test_delete_aborted(self, parser, mock_api, capsys):
        args = parser.parse_args(["secret", "delete", "org/demo", "KEY"])
        with (
            patch("modelscope_hub.cli.secret.make_api", return_value=mock_api),
            patch("builtins.input", return_value="n"),
        ):
            _SecretDelete(args).execute()
        mock_api.delete_secret.assert_not_called()
        out = capsys.readouterr().out
        assert "Aborted" in out

    def test_delete_confirmed_interactively(self, parser, mock_api, capsys):
        args = parser.parse_args(["secret", "delete", "org/demo", "KEY"])
        with (
            patch("modelscope_hub.cli.secret.make_api", return_value=mock_api),
            patch("builtins.input", return_value="y"),
        ):
            _SecretDelete(args).execute()
        mock_api.delete_secret.assert_called_once()


# ===================================================================
# Remote integration tests (existing)
# ===================================================================
@pytest.mark.remote
class TestSecretLifecycle:
    """Test secret CRUD operations with real API on a studio repo."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_repo(self, api, test_owner, repo_name):
        """Create a studio repo for secret testing."""
        cls = type(self)
        cls.repo_id = f"{test_owner}/{repo_name}_secrets"
        api.create_repo(cls.repo_id, "studio", visibility="private")
        cls.api = api
        yield
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                api.delete_repo(cls.repo_id, "studio")
            except Exception:
                pass

    def test_01_add_secret(self, test_token, test_endpoint):
        """Add a new secret."""
        exit_code, out, err = run_cli(
            ["secret", "add", self.repo_id, "TEST_KEY", "test_value"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [secret add] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Added" in out

    def test_02_list_secrets(self, test_token, test_endpoint):
        """List secrets shows the added secret."""
        exit_code, out, err = run_cli(
            ["secret", "list", self.repo_id],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [secret list] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "TEST_KEY" in out

    def test_03_update_secret(self, test_token, test_endpoint):
        """Update an existing secret."""
        exit_code, out, err = run_cli(
            ["secret", "update", self.repo_id, "TEST_KEY", "new_value"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [secret update] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Updated" in out

    def test_04_delete_secret(self, test_token, test_endpoint):
        """Delete the secret with --yes to skip confirmation."""
        exit_code, out, err = run_cli(
            ["secret", "delete", self.repo_id, "TEST_KEY", "--yes"],
            token=test_token,
            endpoint=test_endpoint,
        )
        print(f"\n** [secret delete] repo_id={self.repo_id}")
        print(f"** exit_code={exit_code}, out={out!r}, err={err!r}")
        assert exit_code == 0
        assert "Deleted" in out
