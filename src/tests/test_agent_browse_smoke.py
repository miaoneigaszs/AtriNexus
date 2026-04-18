"""PR14 冒烟：browse 意图从 FastPath 下线后，agent 仍能通过工具处理。

意在证明三个前提到位：
1. 系统 prompt 里有明确引导，agent 遇到"看文件/列目录/搜内容"不会反问；
2. ToolCatalog 的 `workspace-read` 段注册了 `list_directory` / `read_file` / `search_files`；
3. `should_enable_workspace_read` 对 `workspace_read` 及以上档位放行。

不在此处构造完整 provider mock 跑 agent loop —— 那需要 DB 等重依赖。真正
的端到端回归由用户在 `/opt/AtriNexus-v1` 部署后做（"看看 src 目录"/"读 README"/
"搜 fast_path"）。
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
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
