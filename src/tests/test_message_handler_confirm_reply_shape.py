"""PR31 — 长 confirm_reply 对用户友好展示。

message_handler 顶层依赖 emoji / config 等在 CI 中可能不可用，
直接 ast 抽取 _compact_confirm_reply_for_user 函数源码 + 常量，
不 import 真实模块，保证测试无副作用。
"""

from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


def _load_compact_fn():
    module_path = Path(__file__).resolve().parents[1] / "conversation" / "message_handler.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    fn_src = None
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("_COMPACT_CONFIRM_REPLY"):
                    constants[target.id] = ast.literal_eval(node.value)
        elif isinstance(node, ast.FunctionDef) and node.name == "_compact_confirm_reply_for_user":
            fn_src = ast.get_source_segment(source, node)

    assert fn_src, "_compact_confirm_reply_for_user 函数不在 message_handler 模块顶层"

    namespace = {"List": list}
    namespace.update(constants)
    exec(fn_src, namespace)
    return namespace["_compact_confirm_reply_for_user"]


class CompactConfirmReplyShortOutputTest(unittest.TestCase):
    def setUp(self):
        self.compact = _load_compact_fn()

    def test_short_output_is_returned_as_is(self):
        text = "命令: uptime\n退出码: 0\n执行模式: shell-readonly\n\n标准输出:\n12:00 up 1 day"
        self.assertEqual(self.compact(text), text)

    def test_empty_string_returns_empty(self):
        self.assertEqual(self.compact(""), "")

    def test_output_just_below_threshold_untouched(self):
        body = "\n".join(f"line {i}" for i in range(18))
        text = f"命令: cat x\n退出码: 0\n执行模式: shell-readonly\n\n标准输出:\n{body}"
        # 总行数 = 5 (meta + blank + marker) + 18 = 23 < 25, chars < 1500
        self.assertEqual(self.compact(text), text)


class CompactConfirmReplyLongOutputTest(unittest.TestCase):
    def setUp(self):
        self.compact = _load_compact_fn()

    def test_long_output_is_compacted_with_preview(self):
        body = "\n".join(f"line {i}" for i in range(100))
        text = f"命令: find . -type f\n退出码: 0\n执行模式: shell-readonly\n\n标准输出:\n{body}"
        result = self.compact(text)

        self.assertIn("✓ 命令已执行", result)
        self.assertIn("命令: find . -type f", result)
        self.assertIn("退出码: 0", result)
        self.assertIn("输出预览", result)
        self.assertIn("line 0", result)
        self.assertIn("line 7", result)
        self.assertNotIn("line 99", result)
        self.assertIn("已存入上下文", result)

    def test_very_long_single_line_triggers_char_threshold(self):
        text = (
            "命令: echo big\n退出码: 0\n执行模式: shell-readonly\n\n"
            "标准输出:\n" + ("x" * 2000)
        )
        result = self.compact(text)
        self.assertIn("✓ 命令已执行", result)
        self.assertIn("命令: echo big", result)
        self.assertIn("已存入上下文", result)

    def test_long_output_without_standard_shape_falls_back_to_head_preview(self):
        text = "\n".join(f"random-line-{i}" for i in range(50))
        result = self.compact(text)
        self.assertIn("✓ 命令已执行", result)
        self.assertIn("random-line-0", result)
        self.assertIn("已存入上下文", result)


if __name__ == "__main__":
    unittest.main()
