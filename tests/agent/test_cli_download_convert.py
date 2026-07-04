# Copyright (c) ModelScope Contributors. All rights reserved.
"""CLI download (stubbed client) and local convert flows."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modelscope_hub.agent._commands import (
    cmd_convert,
    cmd_download,
)


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


class TestDownload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "ws"
        _DownloadStub.instances = []

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
