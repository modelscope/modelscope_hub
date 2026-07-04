# Copyright (c) Alibaba, Inc. and its affiliates.
"""Core command logic unit tests (upload / download / convert flows)."""
import tempfile
import unittest
from pathlib import Path

from modelscope_hub.agent._commands import (
    available_frameworks,
    build_spec,
    cmd_convert,
    cmd_download,
    cmd_status,
    cmd_upload,
    repo_name,
    resolve_local_name,
    resolve_remote,
)


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


class TestCmdStatus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "agents").mkdir()
        (self.root / "agents" / "a.md").write_text("a")
        (self.root / "AGENTS.md").write_text("shared")

    def tearDown(self):
        self.tmp.cleanup()

    def test_status_ok(self):
        rc = cmd_status(framework="qoder", local_dir=self.root)
        self.assertEqual(rc, 0)

    def test_unknown_framework(self):
        rc = cmd_status(framework="nope", local_dir=self.root)
        self.assertEqual(rc, 1)


class TestCmdUpload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "agents").mkdir()
        (self.root / "agents" / "reviewer.md").write_text("reviewer")
        (self.root / "AGENTS.md").write_text("shared")

    def tearDown(self):
        self.tmp.cleanup()

    def test_unknown_framework_fails(self):
        rc = cmd_upload(framework="nope", name="x", local_dir=self.root)
        self.assertEqual(rc, 1)

    def test_dry_run_ok(self):
        rc = cmd_upload(
            framework="qoder", name="reviewer",
            local_dir=self.root, dry_run=True,
        )
        self.assertEqual(rc, 0)

    def test_no_files_fails(self):
        rc = cmd_upload(
            framework="qoder", name="ghost",
            local_dir=self.root / "empty",
        )
        self.assertEqual(rc, 1)


class TestCmdConvert(unittest.TestCase):
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

    def test_convert_nanobot_to_hermes(self):
        rc = cmd_convert(
            source_fw="nanobot", target_fw="hermes",
            local_dir=self.src, out_dir=self.out,
        )
        self.assertEqual(rc, 0)
        self.assertTrue((self.out / "SOUL.md").is_file())
        self.assertTrue((self.out / "memories" / "USER.md").is_file())

    def test_convert_dry_run(self):
        rc = cmd_convert(
            source_fw="nanobot", target_fw="hermes",
            local_dir=self.src, out_dir=self.out, dry_run=True,
        )
        self.assertEqual(rc, 0)
        self.assertFalse(self.out.exists())

    def test_convert_unknown_framework(self):
        rc = cmd_convert(source_fw="nope", target_fw="hermes", local_dir=self.src)
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
