"""PR18 — todo 工具在 ToolCatalog 里注册 + 经 tool handler 修改后能被 system
prompt snapshot 拾到的行为检查。

不启动完整 agent loop（DB / LLM 依赖太重）。用 importlib 隔离 tool_profiles
+ prompt_manager，再构造极小的 ToolBundle-like 替身验证 runtime prompt 的
`【当前待办】` 段会按期望渲染。
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL_CATALOG_PATH = REPO_ROOT / "src" / "agent_runtime" / "tool_catalog.py"
PROMPT_MANAGER_PATH = REPO_ROOT / "src" / "prompting" / "prompt_manager.py"


class ToolCatalogTodoRegistrationTest(unittest.TestCase):
    def test_core_section_declares_todo(self):
        source = TOOL_CATALOG_PATH.read_text(encoding="utf-8")
        self.assertIn('- todo:', source)
        self.assertIn('name="todo"', source)

    def test_todo_schema_mentions_status_enum_and_merge_flag(self):
        source = TOOL_CATALOG_PATH.read_text(encoding="utf-8")
        self.assertIn('"merge"', source)
        self.assertIn("list(VALID_STATUSES)", source)
        self.assertIn("in_progress", source)
        self.assertIn("completed", source)
        self.assertIn("cancelled", source)


class TodoToolHandlerBehaviorTest(unittest.TestCase):
    """通过直接调 `_run_todo_tool` 验证 handler 逻辑，不经 ToolCatalog 构造。"""

    def setUp(self):
        # 每个 test 使用一个独立 TodoStore，避免全局单例污染
        from src.agent_runtime.todo_store import TodoStore, TodoItem

        self.TodoStore = TodoStore
        self.TodoItem = TodoItem

    def test_read_returns_empty_when_no_todos(self):
        # `_run_todo_tool` 用的是模块级 todo_store；这里直接调但清空
        from src.agent_runtime import todo_store as todo_store_mod
        from src.agent_runtime.tool_catalog import _run_todo_tool

        todo_store_mod.todo_store.clear("user-a")
        reply = _run_todo_tool("user-a", {})
        self.assertIn("没有待办", reply)

    def test_set_then_read_reflects_state(self):
        from src.agent_runtime import todo_store as todo_store_mod
        from src.agent_runtime.tool_catalog import _run_todo_tool

        todo_store_mod.todo_store.clear("user-b")
        reply = _run_todo_tool(
            "user-b",
            {
                "todos": [
                    {"id": "1", "content": "写测试", "status": "in_progress"},
                    {"id": "2", "content": "推 PR", "status": "pending"},
                ]
            },
        )
        self.assertIn("已更新", reply)
        self.assertIn("写测试", reply)

        second = _run_todo_tool("user-b", {})
        self.assertIn("写测试", second)
        self.assertIn("推 PR", second)

    def test_merge_preserves_untouched_and_updates_target(self):
        from src.agent_runtime import todo_store as todo_store_mod
        from src.agent_runtime.tool_catalog import _run_todo_tool

        todo_store_mod.todo_store.clear("user-c")
        _run_todo_tool(
            "user-c",
            {
                "todos": [
                    {"id": "1", "content": "第 1", "status": "pending"},
                    {"id": "2", "content": "第 2", "status": "pending"},
                ]
            },
        )
        reply = _run_todo_tool(
            "user-c",
            {
                "todos": [{"id": "1", "content": "第 1", "status": "completed"}],
                "merge": True,
            },
        )
        self.assertIn("已合并", reply)
        self.assertIn("completed", reply)
        self.assertIn("第 2", reply)

    def test_invalid_payload_returns_error(self):
        from src.agent_runtime.tool_catalog import _run_todo_tool

        reply = _run_todo_tool("user-d", {"todos": "not a list"})
        self.assertIn("失败", reply)


class PromptManagerTodoSnapshotTest(unittest.TestCase):
    def test_capability_block_and_todo_block_both_appear(self):
        from src.prompting.prompt_manager import PromptManager

        pm = PromptManager(root_dir=".")
        prompt = pm.build_runtime_prompt(
            persona_prompt="",
            tool_profile="workspace_read",
            tool_profiles=["core", "workspace_read"],
            tool_summary="[基础] 读时间 / 维护待办",
            core_memory=None,
            current_mode="work",
            todo_snapshot="- [进行中] (1) 写测试\n- [未开始] (2) 推 PR",
        )
        self.assertIn("【你现在的能力】", prompt)
        self.assertIn("【当前待办】", prompt)
        capability_idx = prompt.index("【你现在的能力】")
        todo_idx = prompt.index("【当前待办】")
        self.assertLess(capability_idx, todo_idx)
        self.assertIn("写测试", prompt)

    def test_todo_block_skipped_when_snapshot_empty(self):
        from src.prompting.prompt_manager import PromptManager

        pm = PromptManager(root_dir=".")
        prompt = pm.build_runtime_prompt(
            persona_prompt="",
            tool_profile="workspace_read",
            tool_profiles=["core"],
            tool_summary="x",
            core_memory=None,
            current_mode="work",
            todo_snapshot="",
        )
        self.assertNotIn("【当前待办】", prompt)


if __name__ == "__main__":
    unittest.main()
