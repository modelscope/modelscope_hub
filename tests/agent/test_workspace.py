# Copyright (c) Alibaba, Inc. and its affiliates.
"""Sub-agent-aware workspace spec collection tests."""
import tempfile
import unittest
from pathlib import Path

from modelscope_hub.agent import FRAMEWORK_REGISTRY
from modelscope_hub.agent._commands import build_spec
from modelscope_hub.agent.frameworks.nanobot import NanobotWorkspace
from modelscope_hub.agent.frameworks.qoder import QoderWorkspace
from modelscope_hub.agent.frameworks.qwenpaw import QwenpawWorkspace


class TestAgentAwareCollect(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_qoder_collects_named_agent_plus_shared(self):
        (self.root / "agents").mkdir()
        (self.root / "agents" / "reviewer.md").write_text("reviewer agent")
        (self.root / "agents" / "other.md").write_text("other agent")
        (self.root / "AGENTS.md").write_text("shared instructions")
        (self.root / "skills" / "x").mkdir(parents=True)
        (self.root / "skills" / "x" / "SKILL.md").write_text("skill")

        spec = QoderWorkspace(agent_name="reviewer", local_dir=self.root)
        collected = spec.collect()

        self.assertIn("agents/reviewer.md", collected)
        self.assertIn("AGENTS.md", collected)
        self.assertIn("skills/x/SKILL.md", collected)
        self.assertNotIn("agents/other.md", collected)

    def test_name_templating_is_isolated_per_agent(self):
        (self.root / "agents").mkdir()
        (self.root / "agents" / "a.md").write_text("a")
        (self.root / "agents" / "b.md").write_text("b")

        a = QoderWorkspace(agent_name="a", local_dir=self.root).collect()
        b = QoderWorkspace(agent_name="b", local_dir=self.root).collect()
        self.assertEqual(set(a), {"agents/a.md"})
        self.assertEqual(set(b), {"agents/b.md"})

    def test_qoder_list_agents(self):
        (self.root / "agents").mkdir()
        (self.root / "agents" / "a.md").write_text("a")
        (self.root / "agents" / "b.md").write_text("b")
        spec = QoderWorkspace(local_dir=self.root)
        self.assertEqual(spec.list_agents(), ["default", "a", "b"])

    def test_qwenpaw_default_root_uses_agent_name(self):
        spec = QwenpawWorkspace(agent_name="browse-agent")
        self.assertTrue(
            str(spec.default_workspace_root).endswith("workspaces/browse-agent")
        )

    def test_local_dir_override_wins(self):
        (self.root / "SOUL.md").write_text("soul")
        (self.root / "memory").mkdir()
        (self.root / "memory" / "MEMORY.md").write_text("mem")
        spec = NanobotWorkspace(local_dir=self.root)
        self.assertEqual(spec.workspace_root, self.root)
        collected = spec.collect()
        self.assertIn("SOUL.md", collected)
        self.assertIn("memory/MEMORY.md", collected)

    def test_missing_root_returns_empty(self):
        spec = QoderWorkspace(
            agent_name="x", local_dir=self.root / "does-not-exist"
        )
        self.assertEqual(spec.collect(), {})

    def test_registry_includes_all_frameworks(self):
        for fw in ("qoder", "qwenpaw", "openclaw", "hermes", "nanobot", "openhuman"):
            self.assertIn(fw, FRAMEWORK_REGISTRY)


class TestAllPathPrefix(unittest.TestCase):
    """split_all_path / join_all_path for cross-framework all-mode convert."""

    def test_qwenpaw_split(self):
        spec = build_spec("qwenpaw", "all")
        self.assertTrue(spec.is_root_per_agent)
        self.assertEqual(spec.split_all_path("bot-a/AGENTS.md"), ("bot-a", "AGENTS.md"))
        self.assertEqual(spec.split_all_path("default/SOUL.md"), ("default", "SOUL.md"))
        self.assertEqual(
            spec.split_all_path("bot-a/skills/x/SKILL.md"), ("bot-a", "skills/x/SKILL.md"))
        self.assertEqual(spec.split_all_path("README.md"), (None, "README.md"))

    def test_qwenpaw_join(self):
        spec = build_spec("qwenpaw", "all")
        self.assertEqual(spec.join_all_path("bot-a", "AGENTS.md"), "bot-a/AGENTS.md")
        self.assertEqual(spec.join_all_path("default", "SOUL.md"), "default/SOUL.md")

    def test_openclaw_split(self):
        spec = build_spec("openclaw", "all")
        self.assertTrue(spec.is_root_per_agent)
        self.assertEqual(spec.split_all_path("workspace/AGENTS.md"), ("default", "AGENTS.md"))
        self.assertEqual(
            spec.split_all_path("workspace-bot-a/SOUL.md"), ("bot-a", "SOUL.md"))
        self.assertEqual(spec.split_all_path("README.md"), (None, "README.md"))

    def test_openclaw_join(self):
        spec = build_spec("openclaw", "all")
        self.assertEqual(spec.join_all_path("default", "AGENTS.md"), "workspace/AGENTS.md")
        self.assertEqual(spec.join_all_path("bot-a", "SOUL.md"), "workspace-bot-a/SOUL.md")

    def test_roundtrip_qwenpaw_to_openclaw(self):
        src = build_spec("qwenpaw", "all")
        dst = build_spec("openclaw", "all")
        agent, bare = src.split_all_path("bot-a/AGENTS.md")
        self.assertEqual(dst.join_all_path(agent, bare), "workspace-bot-a/AGENTS.md")

    def test_non_root_per_agent_passthrough(self):
        spec = build_spec("qoder", "all")
        self.assertFalse(spec.is_root_per_agent)
        self.assertEqual(spec.split_all_path("agents/x.md"), (None, "agents/x.md"))
        self.assertEqual(spec.join_all_path("x", "agents/x.md"), "agents/x.md")


if __name__ == "__main__":
    unittest.main()
