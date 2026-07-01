"""浏览请求通过 agent 工具处理的冒烟测试。"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_MD = REPO_ROOT / "src" / "prompting" / "system" / "tools.md"
TOOL_CATALOG_PATH = REPO_ROOT / "src" / "agent_runtime" / "tool_catalog.py"

PROFILES_PATH = REPO_ROOT / "src" / "agent_runtime" / "tool_profiles.py"
PROFILES_SPEC = importlib.util.spec_from_file_location("tool_profiles_under_test", PROFILES_PATH)
PROFILES_MODULE = importlib.util.module_from_spec(PROFILES_SPEC)
assert PROFILES_SPEC and PROFILES_SPEC.loader
sys.modules["tool_profiles_under_test"] = PROFILES_MODULE
PROFILES_SPEC.loader.exec_module(PROFILES_MODULE)
should_enable_workspace_read = PROFILES_MODULE.should_enable_workspace_read


class SystemPromptBrowseNudgeTest(unittest.TestCase):
    def test_tools_md_tells_agent_to_call_tools_without_asking(self):
        content = TOOLS_MD.read_text(encoding="utf-8")
        self.assertIn("看文件", content)
        self.assertIn("list_directory", content)
        self.assertIn("read_file", content)
        self.assertIn("search_files", content)
        self.assertIn("不要先反问", content)


class ToolCatalogBrowseToolsRegisteredTest(unittest.TestCase):
    def test_workspace_read_section_declares_browse_tools(self):
        source = TOOL_CATALOG_PATH.read_text(encoding="utf-8")
        self.assertIn('"- list_directory', source)
        self.assertIn('"- read_file', source)
        self.assertIn('"- search_files', source)


class WorkspaceReadProfileEnabledTest(unittest.TestCase):
    def test_workspace_read_and_above_profiles_enable_read_tools(self):
        self.assertTrue(should_enable_workspace_read("workspace_read"))
        self.assertTrue(should_enable_workspace_read("workspace_edit"))
        self.assertTrue(should_enable_workspace_read("workspace_exec"))
        self.assertTrue(should_enable_workspace_read("full"))

    def test_chat_profile_does_not_enable_read_tools(self):
        self.assertFalse(should_enable_workspace_read("chat"))


if __name__ == "__main__":
    unittest.main()
