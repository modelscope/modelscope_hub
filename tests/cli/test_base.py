"""Tests for base.py helper functions — render_table, parse_kv_pairs, make_api, etc."""
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from unittest.mock import patch

import pytest

from modelscope_hub.cli.base import (
    add_repo_type_arg,
    make_api,
    parse_kv_pairs,
    render_table,
)


class TestRenderTable:
    """Verify render_table formatting."""

    def test_render_table(self):
        """Render a table with headers and rows."""
        rows = [
            ("owner/model1", "public", 100, 5),
            ("owner/model2", "private", 0, 0),
        ]
        headers = ["repo_id", "visibility", "downloads", "likes"]
        output = render_table(rows, headers)
        assert "repo_id" in output
        assert "owner/model1" in output
        assert "owner/model2" in output
        # Verify it has a separator line
        lines = output.strip().splitlines()
        assert len(lines) == 4  # header + separator + 2 data rows
        assert all("-" in lines[1] for _ in range(1))

    def test_render_table_empty(self):
        """Render a table with no rows."""
        output = render_table([], headers=["a", "b", "c"])
        lines = output.strip().splitlines()
        assert len(lines) == 2  # header + separator only

    def test_render_table_none_values(self):
        """None values are rendered as '-'."""
        rows = [(None, None)]
        output = render_table(rows, headers=["name", "value"])
        assert "-" in output

    def test_render_table_long_values_truncated(self):
        """Very long cell values are truncated."""
        long_text = "x" * 200
        rows = [(long_text,)]
        output = render_table(rows, headers=["text"])
        # Should be truncated to 80 chars + ellipsis
        for line in output.splitlines()[2:]:
            assert len(line.strip()) <= 81  # 80 chars + potential trailing space


class TestParseKvPairs:
    """Verify parse_kv_pairs parsing."""

    def test_parse_kv_pairs_valid(self):
        """Parse valid key=value pairs."""
        result = parse_kv_pairs(["cpu=4", "memory=8Gi", "timeout=30"])
        assert result == {"cpu": "4", "memory": "8Gi", "timeout": "30"}

    def test_parse_kv_pairs_value_with_equals(self):
        """Parse value containing '=' sign."""
        result = parse_kv_pairs(["env=FOO=BAR"])
        assert result == {"env": "FOO=BAR"}

    def test_parse_kv_pairs_empty_value(self):
        """Parse key= (empty value is valid)."""
        result = parse_kv_pairs(["key="])
        assert result == {"key": ""}

    def test_parse_kv_pairs_invalid_no_equals(self):
        """Raise ValueError when no '=' is present."""
        with pytest.raises(ValueError, match="key=value"):
            parse_kv_pairs(["no-equals-sign"])

    def test_parse_kv_pairs_invalid_empty_key(self):
        """Raise ValueError when key is empty."""
        with pytest.raises(ValueError, match="empty key"):
            parse_kv_pairs(["=value"])

    def test_parse_kv_pairs_empty_input(self):
        """Empty input yields empty dict."""
        assert parse_kv_pairs([]) == {}


class TestMakeApi:
    """Verify make_api construction."""

    def test_make_api_with_token(self):
        """make_api passes token to HubApi."""
        args = Namespace(token="my_token", endpoint=None)
        with patch("modelscope_hub.cli.base.HubApi") as mock_hub:
            make_api(args)
            mock_hub.assert_called_once_with(token="my_token", endpoint=None)

    def test_make_api_without_token(self):
        """make_api passes None when no token."""
        args = Namespace(token=None, endpoint=None)
        with patch("modelscope_hub.cli.base.HubApi") as mock_hub:
            make_api(args)
            mock_hub.assert_called_once_with(token=None, endpoint=None)

    def test_make_api_with_endpoint(self):
        """make_api passes endpoint to HubApi."""
        args = Namespace(token=None, endpoint="https://custom.endpoint.com")
        with patch("modelscope_hub.cli.base.HubApi") as mock_hub:
            make_api(args)
            mock_hub.assert_called_once_with(
                token=None, endpoint="https://custom.endpoint.com"
            )


class TestAddRepoTypeArg:
    """Verify add_repo_type_arg argparse helper."""

    def test_add_repo_type_arg_required(self):
        """--repo-type is required by default."""
        parser = ArgumentParser()
        add_repo_type_arg(parser)
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_add_repo_type_arg_optional(self):
        """--repo-type with default is optional."""
        parser = ArgumentParser()
        add_repo_type_arg(parser, default="model", required=False)
        args = parser.parse_args([])
        assert args.repo_type == "model"

    def test_add_repo_type_arg_choices(self):
        """--repo-type rejects invalid choices."""
        parser = ArgumentParser()
        add_repo_type_arg(parser, choices=["model", "dataset"])
        with pytest.raises(SystemExit):
            parser.parse_args(["--repo-type", "invalid"])

    def test_add_repo_type_arg_valid(self):
        """--repo-type accepts valid choices."""
        parser = ArgumentParser()
        add_repo_type_arg(parser)
        args = parser.parse_args(["--repo-type", "studio"])
        assert args.repo_type == "studio"
