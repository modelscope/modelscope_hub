# Copyright (c) ModelScope Contributors. All rights reserved.
"""CLI command tests: helper functions, upload/download/convert flows (stubbed client)."""
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from modelscope_hub.agent._commands import (
    available_frameworks,
    build_spec,
    cmd_convert,
    cmd_download,
    cmd_recover,
    cmd_status,
    cmd_upload,
    repo_name,
    resolve_local_name,
    resolve_remote,
)


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


class TestRepoName(unittest.TestCase):
    def test_both_fw_and_name(self):
        self.assertEqual(repo_name("qoder", "reviewer"), "qoder-reviewer")

    def test_name_all(self):
        self.assertEqual(repo_name("qoder", "all"), "qoder")

    def test_fw_only(self):
        self.assertEqual(repo_name("qoder", ""), "qoder")

    def test_name_only(self):
        self.assertEqual(repo_name("", "mybot"), "mybot")

    def test_neither(self):
        self.assertEqual(repo_name("", ""), "default")


class TestResolveRemote(unittest.TestCase):
    def test_repo_with_slash(self):
        group, name = resolve_remote(repo="org/myrepo", username="u")
        self.assertEqual(group, "org")
        self.assertEqual(name, "myrepo")

    def test_repo_without_slash(self):
        group, name = resolve_remote(repo="myrepo", username="u")
        self.assertEqual(group, "u")
        self.assertEqual(name, "myrepo")

    def test_no_repo_derives(self):
        group, name = resolve_remote(name="bot", framework="qoder", username="u")
        self.assertEqual(group, "u")
        self.assertEqual(name, "qoder-bot")


class TestResolveLocalName(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_explicit_name_passes_through(self):
        name, err = resolve_local_name("reviewer", "qoder", self.root)
        self.assertEqual(name, "reviewer")
        self.assertIsNone(err)

    def test_single_agent_auto_select(self):
        (self.root / "agents").mkdir()
        (self.root / "agents" / "mybot.md").write_text("x")
        name, err = resolve_local_name(None, "qoder", self.root)
        self.assertEqual(name, "mybot")
        self.assertIsNone(err)

    def test_multiple_agents_error(self):
        (self.root / "agents").mkdir()
        (self.root / "agents" / "a.md").write_text("x")
        (self.root / "agents" / "b.md").write_text("y")
        name, err = resolve_local_name(None, "qoder", self.root)
        self.assertIsNone(name)
        self.assertIn("multiple", err)

    def test_root_per_agent_omitted_name_is_default(self):
        # qwenpaw is root-per-agent (no {name} placeholder): an omitted --name
        # ALWAYS resolves to 'default', never auto-selecting or erroring on
        # sibling sub-agents (bot-a/bot-b).  Regression for the upload bug.
        name, err = resolve_local_name(None, "qwenpaw", self.root)
        self.assertEqual(name, "default")
        self.assertIsNone(err)

    def test_single_agent_layout_omitted_name_is_default(self):
        # single-agent frameworks (hermes) resolve omitted --name to default too.
        name, err = resolve_local_name(None, "hermes", self.root)
        self.assertEqual(name, "default")
        self.assertIsNone(err)

    def test_all_name_passes_through(self):
        name, err = resolve_local_name("all", "qwenpaw", self.root)
        self.assertEqual(name, "all")
        self.assertIsNone(err)

    def test_explicit_bot_name_passes_through(self):
        name, err = resolve_local_name("bot-a", "qwenpaw", self.root)
        self.assertEqual(name, "bot-a")
        self.assertIsNone(err)


# ---------------------------------------------------------------------------
# Status command tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Upload command tests (stubbed client)
# ---------------------------------------------------------------------------


class _StubClient:
    """Records calls so the test can assert the upload flow."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.created = []
        self.committed_actions = []
        self.uploaded_resources = None
        self.lfs_uploads = []
        _StubClient.instances.append(self)

    def check_repo(self, path, name):
        return False

    def create_repo(self, path, name, framework=None):
        self.created.append((path, name, framework))
        return {"success": True}

    def commit_files(self, path, name, actions, **kwargs):
        self.committed_actions.extend(actions)
        # Track resources from the actions for assertions.
        if self.uploaded_resources is None:
            self.uploaded_resources = {}
        import base64
        for a in actions:
            if a.get("encoding") == "base64" and a.get("content"):
                self.uploaded_resources[a["path"]] = base64.b64decode(a["content"])
        return {"success": True}

    def upload_lfs_file(self, path, name, file_path, content, **kwargs):
        self.lfs_uploads.append((file_path, content))
        if self.uploaded_resources is None:
            self.uploaded_resources = {}
        self.uploaded_resources[file_path] = content
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
        # create_repo called with (group, repo_name)
        self.assertEqual(len(client.created), 1)
        self.assertEqual(client.created[0], ("u", "qoder-reviewer", "qoder"))
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


# ---------------------------------------------------------------------------
# Backup/restore tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Download command tests (stubbed client)
# ---------------------------------------------------------------------------


class _DownloadStub:
    """Serves a fixed nanobot repo so download flows can be exercised offline."""

    instances = []
    STORE = {"SOUL.md": "soul", "USER.md": "user", "memory/MEMORY.md": "mem"}

    def __init__(self, *args, **kwargs):
        _DownloadStub.instances.append(self)

    def repo_info(self, path, name):
        return {"Path": path, "Name": name, "Framework": "nanobot", "Revision": 1}

    def list_repo_files(self, path, name):
        return list(self.STORE)

    def download_repo_file(self, path, name, file_path):
        return self.STORE[file_path]


class _QwenpawAllStub:
    """Serves a qwenpaw all-mode repo (agent-prefixed paths) for convert tests."""

    instances = []
    STORE = {
        ".gitattributes": "x",
        "README.md": "readme",
        "default/AGENTS.md": "# default agents",
        "default/SOUL.md": "# default soul",
        "bot-a/AGENTS.md": "# bot-a agents",
        "bot-a/SOUL.md": "# bot-a soul",
        "bot-a/PROFILE.md": "# bot-a profile",
    }

    def __init__(self, *args, **kwargs):
        _QwenpawAllStub.instances.append(self)

    def repo_info(self, path, name):
        return {"Path": path, "Name": name, "Framework": "qwenpaw", "Revision": 1}

    def list_repo_files(self, path, name):
        return list(self.STORE)

    def download_repo_file(self, path, name, file_path):
        return self.STORE[file_path]


class TestDownload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "ws"
        _DownloadStub.instances = []
        _QwenpawAllStub.instances = []

    def tearDown(self):
        self.tmp.cleanup()

    @mock.patch("modelscope_hub.agent._commands.AgentApi", _DownloadStub)
    def test_download_writes_files(self):
        rc = cmd_download(
            framework="nanobot", repo="nano",
            local_dir=str(self.out),
            endpoint="http://s", token="tok", username="u",
        )
        self.assertEqual(rc, 0)
        self.assertEqual((self.out / "SOUL.md").read_text(), "soul")
        self.assertEqual((self.out / "memory" / "MEMORY.md").read_text(), "mem")

    @mock.patch("modelscope_hub.agent._commands.AgentApi", _DownloadStub)
    def test_download_with_conversion(self):
        # nanobot -> hermes: USER.md must land at hermes' memories/USER.md.
        rc = cmd_download(
            framework="nanobot", repo="nano",
            target="hermes", local_dir=str(self.out),
            endpoint="http://s", token="tok", username="u",
        )
        self.assertEqual(rc, 0)
        self.assertTrue((self.out / "memories" / "USER.md").is_file())
        self.assertFalse((self.out / "USER.md").is_file())

    def test_download_without_login_fails(self):
        rc = cmd_download(
            framework="nanobot", repo="nano",
            local_dir=str(self.out),
            endpoint=None, token=None,
        )
        self.assertEqual(rc, 1)

    def test_download_repo_required(self):
        """Download without --repo should fail."""
        rc = cmd_download(
            framework="nanobot", repo="",
            local_dir=str(self.out),
            endpoint="http://s", token="tok", username="u",
        )
        self.assertEqual(rc, 1)

    @mock.patch("modelscope_hub.agent._commands.AgentApi", _DownloadStub)
    def test_download_with_name_creates_agent(self):
        """Download with --name should write files for that local agent."""
        rc = cmd_download(
            framework="nanobot", repo="nano",
            name="myagent", local_dir=str(self.out),
            endpoint="http://s", token="tok", username="u",
        )
        self.assertEqual(rc, 0)
        self.assertTrue((self.out / "SOUL.md").is_file())

    @mock.patch("modelscope_hub.agent._commands.AgentApi", _DownloadStub)
    def test_download_filters_by_allowlist(self):
        """Files not matching the allowlist patterns should be skipped."""
        orig_store = _DownloadStub.STORE.copy()
        _DownloadStub.STORE = {
            "SOUL.md": "soul",
            "random/junk.txt": "junk",
            "memory/MEMORY.md": "mem",
        }
        try:
            rc = cmd_download(
                framework="nanobot", repo="nano",
                local_dir=str(self.out),
                endpoint="http://s", token="tok", username="u",
            )
            self.assertEqual(rc, 0)
            # random/junk.txt should NOT be written.
            self.assertFalse((self.out / "random" / "junk.txt").exists())
            # Valid files should be written.
            self.assertTrue((self.out / "SOUL.md").is_file())
        finally:
            _DownloadStub.STORE = orig_store

    @mock.patch("modelscope_hub.agent._commands.AgentApi", _DownloadStub)
    def test_download_repo_with_slash(self):
        """--repo with '/' uses the specified group instead of username."""
        rc = cmd_download(
            framework="nanobot", repo="othergroup/nano",
            local_dir=str(self.out),
            endpoint="http://s", token="tok", username="u",
        )
        self.assertEqual(rc, 0)
        self.assertTrue((self.out / "SOUL.md").is_file())

    @mock.patch("modelscope_hub.agent._commands.AgentApi", _QwenpawAllStub)
    def test_download_convert_all_root_to_root(self):
        """qwenpaw -> openclaw with --name all: per-agent convert + re-prefix."""
        rc = cmd_download(
            framework="qwenpaw", repo="qw", name="all", target="openclaw",
            local_dir=str(self.out),
            endpoint="http://s", token="tok", username="u",
        )
        self.assertEqual(rc, 0)
        # default -> workspace/, bot-a -> workspace-bot-a/ (openclaw convention)
        self.assertTrue((self.out / "workspace" / "AGENTS.md").is_file())
        self.assertTrue((self.out / "workspace-bot-a" / "AGENTS.md").is_file())
        self.assertTrue((self.out / "workspace-bot-a" / "SOUL.md").is_file())
        # qwenpaw-only PROFILE.md has no openclaw equivalent: must NOT land as-is.
        self.assertFalse((self.out / "workspace-bot-a" / "PROFILE.md").exists())
        # top-level non-agent files (README) are dropped, never mis-prefixed.
        self.assertFalse((self.out / "README.md").exists())
        self.assertFalse((self.out / "workspace" / "README.md").exists())

    @mock.patch("modelscope_hub.agent._commands.AgentApi", _QwenpawAllStub)
    def test_download_convert_all_cross_layout_rejected(self):
        """qwenpaw -> qoder with --name all is cross-layout: must be rejected."""
        rc = cmd_download(
            framework="qwenpaw", repo="qw", name="all", target="qoder",
            local_dir=str(self.out),
            endpoint="http://s", token="tok", username="u",
        )
        self.assertEqual(rc, 1)

    @mock.patch("modelscope_hub.agent._commands.AgentApi", _QwenpawAllStub)
    def test_download_all_same_framework_keeps_prefixed_paths(self):
        """qwenpaw -> qwenpaw with --name all: no convert, agent prefixes kept."""
        rc = cmd_download(
            framework="qwenpaw", repo="qw", name="all",
            local_dir=str(self.out),
            endpoint="http://s", token="tok", username="u",
        )
        self.assertEqual(rc, 0)
        self.assertTrue((self.out / "default" / "AGENTS.md").is_file())
        self.assertTrue((self.out / "bot-a" / "AGENTS.md").is_file())
        self.assertTrue((self.out / "bot-a" / "PROFILE.md").is_file())
        # non-spec top-level files are skipped.
        self.assertFalse((self.out / "README.md").exists())


# ---------------------------------------------------------------------------
# Convert command tests
# ---------------------------------------------------------------------------


class TestConvert(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name) / "nb"
        self.out = Path(self.tmp.name) / "hm"
        (self.src / "memory").mkdir(parents=True)
        (self.src / "SOUL.md").write_text("nano soul")
        (self.src / "USER.md").write_text("about user")
        (self.src / "memory" / "MEMORY.md").write_text("fact")

    def tearDown(self):
        self.tmp.cleanup()

    def test_convert_local_nanobot_to_hermes(self):
        rc = cmd_convert(
            source_fw="nanobot", target_fw="hermes",
            local_dir=str(self.src), out_dir=str(self.out),
        )
        self.assertEqual(rc, 0)
        self.assertTrue((self.out / "SOUL.md").is_file())
        # nanobot USER.md maps to hermes memories/USER.md
        self.assertTrue((self.out / "memories" / "USER.md").is_file())

    def test_convert_dry_run_writes_nothing(self):
        rc = cmd_convert(
            source_fw="nanobot", target_fw="hermes",
            local_dir=str(self.src), out_dir=str(self.out),
            dry_run=True,
        )
        self.assertEqual(rc, 0)
        self.assertFalse(self.out.exists())

    def test_convert_unknown_framework_fails(self):
        rc = cmd_convert(
            source_fw="nope", target_fw="hermes",
            local_dir=str(self.src),
        )
        self.assertEqual(rc, 1)

    def test_convert_no_source_files_fails(self):
        rc = cmd_convert(
            source_fw="nanobot", target_fw="hermes",
            local_dir=str(self.src / "missing"),
        )
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
