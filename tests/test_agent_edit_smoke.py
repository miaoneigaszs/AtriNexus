"""Smoke tests for edit requests handled through agent tools."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_MD = REPO_ROOT / "src" / "prompting" / "system" / "tools.md"
TOOL_CATALOG_PATH = REPO_ROOT / "src" / "agent_runtime" / "tool_catalog.py"

PROFILES_PATH = REPO_ROOT / "src" / "agent_runtime" / "tool_profiles.py"
PROFILES_SPEC = importlib.util.spec_from_file_location("tool_profiles_for_edit_smoke", PROFILES_PATH)
PROFILES_MODULE = importlib.util.module_from_spec(PROFILES_SPEC)
assert PROFILES_SPEC and PROFILES_SPEC.loader
sys.modules["tool_profiles_for_edit_smoke"] = PROFILES_MODULE
PROFILES_SPEC.loader.exec_module(PROFILES_MODULE)
should_enable_workspace_edit = PROFILES_MODULE.should_enable_workspace_edit


class SystemPromptEditNudgeTest(unittest.TestCase):
    def test_tools_md_tells_agent_to_call_preview_tools(self):
        content = TOOLS_MD.read_text(encoding="utf-8")
        for keyword in (
            "preview_edit_file",
            "preview_write_file",
            "preview_append_file",
            "rename_path",
        ):
            self.assertIn(keyword, content, f"tools.md 应明确提到 {keyword}")
        self.assertIn("通过", content)
        self.assertIn("不要", content)


class ToolCatalogEditToolsRegisteredTest(unittest.TestCase):
    def test_workspace_edit_section_declares_preview_tools(self):
        source = TOOL_CATALOG_PATH.read_text(encoding="utf-8")
        self.assertIn('"- preview_edit_file', source)
        self.assertIn('"- preview_write_file', source)
        self.assertIn('"- preview_append_file', source)

    def test_workspace_rename_section_declares_rename_tool(self):
        source = TOOL_CATALOG_PATH.read_text(encoding="utf-8")
        self.assertIn('"- rename_path', source)

    def test_tool_specs_defined(self):
        source = TOOL_CATALOG_PATH.read_text(encoding="utf-8")
        for tool in ("preview_edit_file", "preview_write_file", "preview_append_file", "rename_path"):
            self.assertIn(f'name="{tool}"', source, f"工具 {tool} 的 spec 应注册")


class WorkspaceEditProfileEnabledTest(unittest.TestCase):
    def test_edit_and_above_profiles_enable_edit_tools(self):
        self.assertTrue(should_enable_workspace_edit("workspace_edit"))
        self.assertTrue(should_enable_workspace_edit("workspace_exec"))
        self.assertTrue(should_enable_workspace_edit("full"))

    def test_read_and_below_profiles_do_not_enable_edit_tools(self):
        self.assertFalse(should_enable_workspace_edit("workspace_read"))
        self.assertFalse(should_enable_workspace_edit("chat"))


if __name__ == "__main__":
    unittest.main()
