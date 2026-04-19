"""PR21 — per-tool loop-guard 放宽。

只读浏览类（search_files / list_directory / read_file / glob）重复上限放宽到 4，
判重签名只看 tool_name + primary_path；run_command / preview_* / rename_path
保持严格（全参数 + 上限 2）。
"""

from __future__ import annotations

import unittest

from src.agent_runtime.agent_tool_guard import (
    AgentToolGuard,
    MAX_TOOL_REPEAT_COUNT,
    RELAXED_LOOP_TOOL_NAMES,
    RELAXED_TOOL_REPEAT_COUNT,
    TOOL_LOOP_STATE,
)


class ToolSignatureTest(unittest.TestCase):
    def setUp(self):
        self.guard = AgentToolGuard(tool_catalog=None)

    def test_read_file_ignores_offset_and_limit(self):
        sig_a = self.guard.build_tool_signature(
            "read_file", {"path": "README.md", "offset": 1, "limit": 100}
        )
        sig_b = self.guard.build_tool_signature(
            "read_file", {"path": "README.md", "offset": 200, "limit": 100}
        )
        self.assertEqual(sig_a, sig_b)
        self.assertIn("path=README.md", sig_a)

    def test_search_files_ignores_query(self):
        sig_a = self.guard.build_tool_signature(
            "search_files", {"path": "src", "query": "foo"}
        )
        sig_b = self.guard.build_tool_signature(
            "search_files", {"path": "src", "query": "bar"}
        )
        self.assertEqual(sig_a, sig_b)

    def test_list_directory_signature_is_path_only(self):
        sig_a = self.guard.build_tool_signature("list_directory", {"path": "src"})
        sig_b = self.guard.build_tool_signature("list_directory", {"path": "docs"})
        self.assertNotEqual(sig_a, sig_b)

    def test_glob_signature_is_path_only_ignoring_pattern(self):
        sig_a = self.guard.build_tool_signature(
            "glob", {"path": ".", "pattern": "*.py"}
        )
        sig_b = self.guard.build_tool_signature(
            "glob", {"path": ".", "pattern": "**/*.md"}
        )
        self.assertEqual(sig_a, sig_b)

    def test_run_command_keeps_full_args_in_signature(self):
        sig_a = self.guard.build_tool_signature(
            "run_command", {"command": "find src | wc -l"}
        )
        sig_b = self.guard.build_tool_signature(
            "run_command", {"command": "find docs | wc -l"}
        )
        self.assertNotEqual(sig_a, sig_b)

    def test_preview_edit_file_keeps_full_args_in_signature(self):
        sig_a = self.guard.build_tool_signature(
            "preview_edit_file",
            {"path": "README.md", "find_text": "a", "replace_text": "b"},
        )
        sig_b = self.guard.build_tool_signature(
            "preview_edit_file",
            {"path": "README.md", "find_text": "a", "replace_text": "c"},
        )
        self.assertNotEqual(sig_a, sig_b)


class RepeatLimitTest(unittest.TestCase):
    def test_relaxed_tools_use_4(self):
        for name in ("search_files", "list_directory", "read_file", "glob"):
            self.assertEqual(AgentToolGuard._repeat_limit_for(name), RELAXED_TOOL_REPEAT_COUNT)
            self.assertEqual(RELAXED_TOOL_REPEAT_COUNT, 4)

    def test_strict_tools_use_default(self):
        for name in ("run_command", "preview_edit_file", "preview_write_file", "rename_path"):
            self.assertEqual(AgentToolGuard._repeat_limit_for(name), MAX_TOOL_REPEAT_COUNT)
            self.assertEqual(MAX_TOOL_REPEAT_COUNT, 2)


class LoopGuardIntegrationTest(unittest.TestCase):
    """把 check_tool_loop 跟真实 context var 串起来，端到端验证边界行为。"""

    def setUp(self):
        self.guard = AgentToolGuard(tool_catalog=None)
        self.token = TOOL_LOOP_STATE.set({"counts": {}, "recent": []})
        self.addCleanup(lambda: TOOL_LOOP_STATE.reset(self.token))

    def test_read_file_with_shifting_offset_allowed_four_times(self):
        for offset in (1, 100, 200, 300):
            result = self.guard.check_tool_loop(
                "read_file", {"path": "long.txt", "offset": offset, "limit": 50}
            )
            self.assertIsNone(result, f"offset={offset} 应被允许")

    def test_read_file_same_path_blocked_on_fifth_call(self):
        for _ in range(4):
            self.guard.check_tool_loop("read_file", {"path": "same.txt"})
        result = self.guard.check_tool_loop("read_file", {"path": "same.txt"})
        self.assertIsNotNone(result)
        self.assertIn("重复", result)

    def test_run_command_blocked_after_two_repeats(self):
        cmd = {"command": "find src"}
        self.guard.check_tool_loop("run_command", cmd)
        self.guard.check_tool_loop("run_command", cmd)
        result = self.guard.check_tool_loop("run_command", cmd)
        self.assertIsNotNone(result)
        self.assertIn("重复", result)

    def test_preview_edit_file_blocked_after_two_repeats(self):
        args = {"path": "README.md", "find_text": "a", "replace_text": "b"}
        self.guard.check_tool_loop("preview_edit_file", args)
        self.guard.check_tool_loop("preview_edit_file", args)
        result = self.guard.check_tool_loop("preview_edit_file", args)
        self.assertIsNotNone(result)

    def test_search_files_different_query_same_path_still_one_signature(self):
        for query in ("foo", "bar", "baz", "qux"):
            result = self.guard.check_tool_loop(
                "search_files", {"path": "src", "query": query}
            )
            self.assertIsNone(result, f"query={query} 应被允许")
        # 第 5 次同 path 即便换 query 也应触发上限
        fifth = self.guard.check_tool_loop(
            "search_files", {"path": "src", "query": "new"}
        )
        self.assertIsNotNone(fifth)

    def test_different_paths_keep_independent_signatures(self):
        # read_file 不同 path 互不干扰，各自独立计数。
        for path in ("a.md", "b.md", "c.md", "d.md", "e.md"):
            result = self.guard.check_tool_loop("read_file", {"path": path})
            self.assertIsNone(result)


class RelaxedSetContentTest(unittest.TestCase):
    def test_relaxed_set_matches_roadmap(self):
        self.assertEqual(
            RELAXED_LOOP_TOOL_NAMES,
            {"search_files", "list_directory", "read_file", "glob"},
        )


if __name__ == "__main__":
    unittest.main()
