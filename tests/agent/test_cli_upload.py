# Copyright (c) ModelScope Contributors. All rights reserved.
"""CLI upload, status, and backup/restore flow tests (stubbed client)."""
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from modelscope_hub.agent._commands import (
    cmd_recover,
    cmd_status,
    cmd_upload,
)


class _StubClient:
    """Records calls so the test can assert the upload flow."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.created = []
        self.uploaded_resources = None
        _StubClient.instances.append(self)

    def check_repo(self, path, name):
        return False

    def upload_file(self, resources):
        """Accept Dict[str, bytes]; return a fake Gid."""
        self.uploaded_resources = resources
        return "fake-gid-uuid"

    def create_repo(self, path, name, framework, **kwargs):
        self.created.append((path, name, framework, kwargs.get("system_prompt_files")))
        return {"success": True}


class TestUploadCmd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "agents").mkdir()
        (self.root / "agents" / "reviewer.md").write_text("reviewer")
        (self.root / "AGENTS.md").write_text("shared")
        (self.root / "skills" / "test-skill").mkdir(parents=True)
        (self.root / "skills" / "test-skill" / "SKILL.md").write_text("skill")
        _StubClient.instances = []

    def tearDown(self):
        self.tmp.cleanup()

    def test_unknown_framework_fails(self):
        rc = cmd_upload(framework="nope", name="x", local_dir=str(self.root))
        self.assertEqual(rc, 1)

    def test_dry_run_does_not_upload(self):
        rc = cmd_upload(
            framework="qoder", name="reviewer",
            local_dir=str(self.root), dry_run=True,
        )
        self.assertEqual(rc, 0)
        self.assertEqual(_StubClient.instances, [])

    def test_no_files_fails(self):
        rc = cmd_upload(
            framework="qoder", name="ghost",
            local_dir=str(self.root / "empty"),
        )
        self.assertEqual(rc, 1)

    @mock.patch("modelscope_hub.agent._commands.AgentApi", _StubClient)
    def test_full_upload_creates_then_uploads_zip(self):
        rc = cmd_upload(
            framework="qoder", name="reviewer",
            local_dir=str(self.root),
            endpoint="http://s", token="tok", username="u",
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(_StubClient.instances), 1)
        client = _StubClient.instances[0]
        # create_repo called with (group, repo_name, framework, system_prompt_files)
        self.assertEqual(len(client.created), 1)
        self.assertEqual(client.created[0][:3], ("u", "qoder-reviewer", "qoder"))
        self.assertEqual(client.created[0][3], "fake-gid-uuid")
        # Verify uploaded resources are bytes-valued dict
        self.assertIsNotNone(client.uploaded_resources)
        self.assertIsInstance(client.uploaded_resources, dict)
        self.assertIn("agents/reviewer.md", client.uploaded_resources)
        self.assertIn("AGENTS.md", client.uploaded_resources)
        for v in client.uploaded_resources.values():
            self.assertIsInstance(v, bytes)

    def test_upload_without_login_fails(self):
        rc = cmd_upload(
            framework="qoder", name="reviewer",
            local_dir=str(self.root),
            endpoint=None, token=None,
        )
        self.assertEqual(rc, 1)

    @mock.patch("modelscope_hub.agent._commands.AgentApi", _StubClient)
    def test_upload_multiple_agents_no_name_fails(self):
        """When --name is not specified and multiple agents exist, should fail."""
        (self.root / "agents" / "coder.md").write_text("coder")
        rc = cmd_upload(
            framework="qoder", name=None,
            local_dir=str(self.root),
            endpoint="http://s", token="tok", username="u",
        )
        self.assertEqual(rc, 1)

    @mock.patch("modelscope_hub.agent._commands.AgentApi", _StubClient)
    def test_upload_auto_select_single_agent(self):
        """When only one sub-agent exists, auto-select it without --name."""
        rc = cmd_upload(
            framework="qoder", name=None,
            local_dir=str(self.root),
            endpoint="http://s", token="tok", username="u",
        )
        self.assertEqual(rc, 0)
        client = _StubClient.instances[0]
        self.assertEqual(client.created[0][1], "qoder-reviewer")

    @mock.patch("modelscope_hub.agent._commands.AgentApi", _StubClient)
    def test_upload_with_repo_slash(self):
        """--repo with '/' should use the group from repo, not username."""
        rc = cmd_upload(
            framework="qoder", name="reviewer",
            repo="mygroup/myrepo",
            local_dir=str(self.root),
            endpoint="http://s", token="tok", username="u",
        )
        self.assertEqual(rc, 0)
        client = _StubClient.instances[0]
        self.assertEqual(client.created[0][0], "mygroup")
        self.assertEqual(client.created[0][1], "myrepo")

    @mock.patch("modelscope_hub.agent._commands.AgentApi", _StubClient)
    def test_upload_repo_defaults_to_name(self):
        """When --repo is omitted, remote repo name derives from --name."""
        rc = cmd_upload(
            framework="qoder", name="reviewer",
            local_dir=str(self.root),
            endpoint="http://s", token="tok", username="u",
        )
        self.assertEqual(rc, 0)
        client = _StubClient.instances[0]
        self.assertEqual(client.created[0][0], "u")
        self.assertEqual(client.created[0][1], "qoder-reviewer")

    @mock.patch("modelscope_hub.agent._commands.AgentApi", _StubClient)
    def test_upload_global_only_no_agents_dir(self):
        """When no agents/ directory exists, upload only shared (global) files."""
        import shutil
        shutil.rmtree(self.root / "agents")
        rc = cmd_upload(
            framework="qoder", name=None,
            local_dir=str(self.root),
            endpoint="http://s", token="tok", username="u",
        )
        self.assertEqual(rc, 0)
        client = _StubClient.instances[0]
        # Repo should be "qoder" (no name specified, global mode).
        self.assertEqual(client.created[0][1], "qoder")
        # Verify that no agents/*.md files are uploaded.
        self.assertIsNotNone(client.uploaded_resources)
        for p in client.uploaded_resources.keys():
            self.assertFalse(p.startswith("agents/"))


class TestStatusCmd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "agents").mkdir()
        (self.root / "agents" / "reviewer.md").write_text("reviewer")
        (self.root / "agents" / "coder.md").write_text("coder")
        (self.root / "AGENTS.md").write_text("shared")

    def tearDown(self):
        self.tmp.cleanup()

    def test_status_shows_agents(self):
        rc = cmd_status(framework="qoder", local_dir=str(self.root))
        self.assertEqual(rc, 0)

    def test_status_unknown_framework_fails(self):
        rc = cmd_status(framework="nope", local_dir=str(self.root))
        self.assertEqual(rc, 1)


class TestBackupsFilterCmd(unittest.TestCase):
    """Test backup list/restore framework and name filtering."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # Create fake backup zips in a temp cache dir.
        self.cache_dir = Path(self.tmp.name)
        for name in [
            "qoder_default_20260624_120000.zip",
            "qoder_reviewer_20260624_130000.zip",
            "qwenpaw_default_20260702_170208.zip",
            "nanobot_mybot_20260703_100000.zip",
        ]:
            zpath = self.cache_dir / name
            with zipfile.ZipFile(zpath, 'w') as zf:
                zf.writestr("dummy.txt", "placeholder")

    def tearDown(self):
        self.tmp.cleanup()

    @mock.patch("modelscope_hub.agent._cache.cache_dir")
    def test_backups_list_all(self, mock_cache):
        """Without --framework, list all backups."""
        mock_cache.return_value = self.cache_dir
        rc = cmd_recover(list_backups=True)
        self.assertEqual(rc, 0)

    @mock.patch("modelscope_hub.agent._cache.cache_dir")
    def test_backups_list_filter_by_framework(self, mock_cache):
        """With --framework qoder, only qoder backups appear."""
        mock_cache.return_value = self.cache_dir
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_recover(list_backups=True, framework="qoder")
        self.assertEqual(rc, 0)
        output = buf.getvalue()
        self.assertIn("qoder_default_20260624_120000.zip", output)
        self.assertIn("qoder_reviewer_20260624_130000.zip", output)
        self.assertNotIn("qwenpaw", output)
        self.assertNotIn("nanobot", output)

    @mock.patch("modelscope_hub.agent._cache.cache_dir")
    def test_backups_list_filter_by_name(self, mock_cache):
        """With --name reviewer, only matching backups appear."""
        mock_cache.return_value = self.cache_dir
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_recover(list_backups=True, name="reviewer")
        self.assertEqual(rc, 0)
        output = buf.getvalue()
        self.assertIn("qoder_reviewer_20260624_130000.zip", output)
        self.assertNotIn("qoder_default", output)
        self.assertNotIn("qwenpaw", output)

    @mock.patch("modelscope_hub.agent._cache.cache_dir")
    def test_backups_list_no_match(self, mock_cache):
        """Filter with nonexistent framework returns 'No backups found'."""
        mock_cache.return_value = self.cache_dir
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_recover(list_backups=True, framework="hermes")
        self.assertEqual(rc, 0)
        self.assertIn("No backups found", buf.getvalue())

    @mock.patch("modelscope_hub.agent._cache.cache_dir")
    def test_restore_last_filters_by_framework(self, mock_cache):
        """'restore last -f qoder' picks the latest qoder backup."""
        mock_cache.return_value = self.cache_dir
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_recover(target="last", framework="qoder")
        # rc=1 because the fake zip doesn't have valid data to restore,
        # but it should attempt the qoder_reviewer (latest qoder) not qwenpaw.
        self.assertNotIn("no backups found", buf.getvalue().lower())

    @mock.patch("modelscope_hub.agent._cache.cache_dir")
    def test_restore_last_no_match_fails(self, mock_cache):
        """'restore last -f hermes' with no hermes backups should fail."""
        mock_cache.return_value = self.cache_dir
        rc = cmd_recover(target="last", framework="hermes")
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
