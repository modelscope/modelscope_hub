"""Tests for ``ms secret`` group — list / add / update / delete."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from modelscope_hub.errors import HubError


class TestSecretList:
    """Verify secret list subcommand."""

    def test_secret_list(self, mock_api, run_cli):
        """List secrets displays table."""
        exit_code, out, err = run_cli(["secret", "list", "owner/studio"])
        assert exit_code == 0
        assert "MY_KEY" in out
        mock_api.list_secrets.assert_called_once_with("owner/studio", "studio")

    def test_secret_list_empty(self, mock_api, run_cli):
        """List secrets with no results shows message."""
        mock_api.list_secrets.return_value = []
        exit_code, out, err = run_cli(["secret", "list", "owner/studio"])
        assert exit_code == 0
        assert "no secrets" in out.lower()


class TestSecretAdd:
    """Verify secret add subcommand."""

    def test_secret_add(self, mock_api, run_cli):
        """Add a secret successfully."""
        exit_code, out, err = run_cli(
            ["secret", "add", "owner/studio", "API_KEY", "secret_value"]
        )
        assert exit_code == 0
        assert "Added" in out
        mock_api.add_secret.assert_called_once_with(
            "owner/studio", "API_KEY", "secret_value", "studio"
        )


class TestSecretUpdate:
    """Verify secret update subcommand."""

    def test_secret_update(self, mock_api, run_cli):
        """Update a secret successfully."""
        exit_code, out, err = run_cli(
            ["secret", "update", "owner/studio", "API_KEY", "new_value"]
        )
        assert exit_code == 0
        assert "Updated" in out
        mock_api.update_secret.assert_called_once_with(
            "owner/studio", "API_KEY", "new_value", "studio"
        )


class TestSecretDelete:
    """Verify secret delete subcommand."""

    def test_secret_delete_confirmed(self, mock_api, run_cli):
        """Delete secret with 'y' confirmation."""
        with patch("builtins.input", return_value="y"):
            exit_code, out, err = run_cli(
                ["secret", "delete", "owner/studio", "API_KEY"]
            )
        assert exit_code == 0
        assert "Deleted" in out
        mock_api.delete_secret.assert_called_once_with(
            "owner/studio", "API_KEY", "studio"
        )

    def test_secret_delete_cancelled(self, mock_api, run_cli):
        """Delete secret aborts on 'n'."""
        with patch("builtins.input", return_value="n"):
            exit_code, out, err = run_cli(
                ["secret", "delete", "owner/studio", "API_KEY"]
            )
        assert exit_code == 0
        assert "Aborted" in out
        mock_api.delete_secret.assert_not_called()

    def test_secret_delete_force_yes(self, mock_api, run_cli):
        """Delete secret --yes skips confirmation."""
        exit_code, out, err = run_cli(
            ["secret", "delete", "owner/studio", "API_KEY", "--yes"]
        )
        assert exit_code == 0
        assert "Deleted" in out
        mock_api.delete_secret.assert_called_once()

    def test_secret_delete_api_error(self, mock_api, run_cli):
        """Delete secret exits 1 on API error."""
        mock_api.delete_secret.side_effect = HubError("delete failed")
        exit_code, out, err = run_cli(
            ["secret", "delete", "owner/studio", "KEY", "--yes"]
        )
        assert exit_code == 1
        assert "delete failed" in err
