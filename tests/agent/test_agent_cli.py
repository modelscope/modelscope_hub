# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for the slim ``ms agent`` CLI (download/upload/list).

These tests exercise the raw transfer command logic with a stub AgentApi
(no network) and verify the parser only exposes download/upload/list -- the
framework-aware commands (convert/watch/status/backups/restore/stop) now live
in modelscope-agent.
"""
from __future__ import annotations

import argparse
import base64
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modelscope_hub.agent import AgentApi, RemoteFileInfo, is_lfs_file
from modelscope_hub.cli import agent as cli_agent


class _StubClient:
    """In-memory stand-in for AgentApi used by the slim CLI functions."""

    # Class-level storage so instances created inside the CLI share state.
    files: dict[str, bytes] = {}
    agents: list[dict] = []
    exists: bool = True
    commits: list[list[dict]] = []
    lfs_uploads: list[tuple[str, bytes]] = []
    created: list[tuple[str, str]] = []
    created_visibility: list[str] = []

    def __init__(self, *args, **kwargs):
        pass

    # ---- list ----
    def list_agents(self, owner=None, page_number=1, page_size=10):
        return {"items": list(self.agents), "total_count": len(self.agents)}

    # ---- download ----
    def repo_info(self, path, name):
        return {"path": path, "name": name} if self.exists else None

    def list_repo_files(self, path, name, revision="master"):
        return list(self.files.keys())

    def download_repo_file(self, path, name, file_path, revision="master", *, binary=False):
        data = self.files[file_path]
        return data if binary else data.decode("utf-8")

    # ---- upload ----
    def check_repo(self, path, name):
        return self.exists

    def create_repo(self, path, name, framework=None, visibility="public"):
        type(self).created.append((path, name))
        type(self).created_visibility.append(visibility)
        return {"path": path, "name": name}

    def commit_files(self, path, name, actions, revision="master", commit_message="sync"):
        type(self).commits.append(actions)
        return {"ok": True}

    def upload_lfs_file(self, path, name, file_path, content, action="create",
                        revision="master", commit_message="sync"):
        type(self).lfs_uploads.append((file_path, content))
        return {"ok": True}


def _reset_stub():
    _StubClient.files = {}
    _StubClient.agents = []
    _StubClient.exists = True
    _StubClient.commits = []
    _StubClient.lfs_uploads = []
    _StubClient.created = []
    _StubClient.created_visibility = []


class TestSlimParser(unittest.TestCase):
    """The slim CLI must expose only download/upload/list."""

    def _build_parser(self):
        parser = argparse.ArgumentParser(prog="ms")
        parser.add_argument("--token", default=None)
        parser.add_argument("--endpoint", default=None)
        sub = parser.add_subparsers(dest="command")
        cli_agent.AgentCommand.register(sub)
        return parser

    def test_download_upload_list_present(self):
        parser = self._build_parser()
        for action, extra in (
            ("download", ["-r", "u/a"]),
            ("upload", ["-r", "u/a"]),
            ("list", []),
        ):
            args = parser.parse_args(["agent", action, *extra])
            self.assertEqual(args.agent_command, action)

    def test_framework_commands_absent(self):
        parser = self._build_parser()
        for action in ("convert", "watch", "status", "backups", "restore", "stop"):
            with self.assertRaises(SystemExit):
                parser.parse_args(["agent", action])


class TestCmdList(unittest.TestCase):
    def setUp(self):
        _reset_stub()

    @mock.patch.object(cli_agent, "AgentApi", _StubClient)
    def test_list_empty(self):
        rc = cli_agent._cmd_list(None, 1, 10, endpoint="https://x", token="t")
        self.assertEqual(rc, 0)

    @mock.patch.object(cli_agent, "AgentApi", _StubClient)
    def test_list_rows(self):
        _StubClient.agents = [
            {"Path": "user", "Name": "a1", "Framework": "qoder",
             "Visibility": "public", "LastUpdatedDate": "2024-01-02T03:04:05"},
        ]
        rc = cli_agent._cmd_list(None, 1, 10, endpoint="https://x", token="t")
        self.assertEqual(rc, 0)

    def test_list_requires_endpoint(self):
        rc = cli_agent._cmd_list(None, 1, 10, endpoint=None, token="t")
        self.assertEqual(rc, 1)


class TestCmdDownload(unittest.TestCase):
    def setUp(self):
        _reset_stub()

    @mock.patch.object(cli_agent, "AgentApi", _StubClient)
    def test_download_writes_files(self):
        _StubClient.files = {"AGENTS.md": b"hello", "sub/x.txt": b"world"}
        with tempfile.TemporaryDirectory() as d:
            rc = cli_agent._cmd_download(
                repo="user/a", local_dir=d, revision="master",
                endpoint="https://x", token="t", username="user")
            self.assertEqual(rc, 0)
            self.assertEqual((Path(d) / "AGENTS.md").read_bytes(), b"hello")
            self.assertEqual((Path(d) / "sub" / "x.txt").read_bytes(), b"world")

    @mock.patch.object(cli_agent, "AgentApi", _StubClient)
    def test_download_missing_repo(self):
        _StubClient.exists = False
        with tempfile.TemporaryDirectory() as d:
            rc = cli_agent._cmd_download(
                repo="user/a", local_dir=d, revision="master",
                endpoint="https://x", token="t", username="user")
            self.assertEqual(rc, 1)

    def test_download_needs_owner_without_login(self):
        rc = cli_agent._cmd_download(
            repo="a", local_dir=None, revision="master",
            endpoint="https://x", token="", username="")
        self.assertEqual(rc, 1)


class TestCmdUpload(unittest.TestCase):
    def setUp(self):
        _reset_stub()

    @mock.patch.object(cli_agent, "AgentApi", _StubClient)
    def test_upload_normal_files(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "AGENTS.md").write_bytes(b"hello")
            (Path(d) / "sub").mkdir()
            (Path(d) / "sub" / "x.txt").write_bytes(b"world")
            rc = cli_agent._cmd_upload(
                repo="user/a", local_dir=d, revision="master", dry_run=False,
                endpoint="https://x", token="t", username="user")
            self.assertEqual(rc, 0)
            # one commit with two normal-file actions
            self.assertEqual(len(_StubClient.commits), 1)
            paths = {a["path"] for a in _StubClient.commits[0]}
            self.assertEqual(paths, {"AGENTS.md", "sub/x.txt"})
            # content is base64-encoded
            action = _StubClient.commits[0][0]
            self.assertEqual(action["encoding"], "base64")
            base64.b64decode(action["content"])  # must decode without error

    @mock.patch.object(cli_agent, "AgentApi", _StubClient)
    def test_upload_lfs_file(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "model.bin").write_bytes(b"\x00\x01\x02")
            rc = cli_agent._cmd_upload(
                repo="user/a", local_dir=d, revision="master", dry_run=False,
                endpoint="https://x", token="t", username="user")
            self.assertEqual(rc, 0)
            self.assertEqual(len(_StubClient.lfs_uploads), 1)
            self.assertEqual(_StubClient.lfs_uploads[0][0], "model.bin")

    @mock.patch.object(cli_agent, "AgentApi", _StubClient)
    def test_upload_creates_repo_when_absent(self):
        _StubClient.exists = False
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "AGENTS.md").write_bytes(b"hello")
            rc = cli_agent._cmd_upload(
                repo="user/a", local_dir=d, revision="master", dry_run=False,
                endpoint="https://x", token="t", username="user")
            self.assertEqual(rc, 0)
            self.assertEqual(_StubClient.created, [("user", "a")])

    @mock.patch.object(cli_agent, "AgentApi", _StubClient)
    def test_upload_dry_run_no_network(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "AGENTS.md").write_bytes(b"hello")
            rc = cli_agent._cmd_upload(
                repo="user/a", local_dir=d, revision="master", dry_run=True,
                endpoint="https://x", token="t", username="user")
            self.assertEqual(rc, 0)
            self.assertEqual(_StubClient.commits, [])
            self.assertEqual(_StubClient.lfs_uploads, [])

    @mock.patch.object(cli_agent, "AgentApi", _StubClient)
    def test_upload_passes_visibility_to_create_repo(self):
        _StubClient.exists = False
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "AGENTS.md").write_bytes(b"hello")
            rc = cli_agent._cmd_upload(
                repo="user/a", local_dir=d, revision="master", dry_run=False,
                endpoint="https://x", token="t", username="user",
                visibility="private")
            self.assertEqual(rc, 0)
            self.assertEqual(_StubClient.created_visibility, ["private"])


class TestAgentApiHelpers(unittest.TestCase):
    def test_is_lfs_by_extension(self):
        self.assertTrue(is_lfs_file("model.bin", 10))
        self.assertTrue(is_lfs_file("weights.safetensors", 10))
        self.assertFalse(is_lfs_file("AGENTS.md", 10))

    def test_is_lfs_by_size(self):
        self.assertTrue(is_lfs_file("big.txt", 2 * 1024 * 1024))
        self.assertFalse(is_lfs_file("small.txt", 1024))

    def test_remote_file_info(self):
        info = RemoteFileInfo(path="a.md", sha256="abc")
        self.assertEqual(info.path, "a.md")
        self.assertFalse(info.is_lfs)

    def test_agent_api_importable(self):
        self.assertTrue(callable(AgentApi))


if __name__ == "__main__":
    unittest.main()
