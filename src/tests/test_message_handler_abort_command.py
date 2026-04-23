"""message_handler 顶层 abort 命令匹配测试。"""

from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


def _load_abort_helpers():
    module_path = Path(__file__).resolve().parents[1] / "conversation" / "message_handler.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    constants = {}
    function_sources = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "_ABORT_COMMANDS",
                    "_ABORT_TRAILING_PUNCTUATION",
                }:
                    constants[target.id] = ast.literal_eval(node.value)
        elif isinstance(node, ast.FunctionDef) and node.name in {
            "_normalize_abort_command",
            "_is_abort_command",
        }:
            function_sources[node.name] = ast.get_source_segment(source, node)

    assert "_normalize_abort_command" in function_sources
    assert "_is_abort_command" in function_sources

    namespace = {}
    namespace.update(constants)
    exec(function_sources["_normalize_abort_command"], namespace)
    exec(function_sources["_is_abort_command"], namespace)
    return namespace["_normalize_abort_command"], namespace["_is_abort_command"]


class AbortCommandMatchTest(unittest.TestCase):
    def setUp(self):
        self.normalize, self.is_abort = _load_abort_helpers()

    def test_explicit_abort_commands_match_after_normalization(self):
        cases = [
            "取消",
            " 取消 ",
            "取消！！！",
            "停止。",
            "别弄了…",
            "不做了",
            "算了",
            "stop",
            " STOP ",
            "cancel!!!",
            "abort?",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertTrue(self.is_abort(text))

    def test_negated_or_extended_phrases_do_not_match(self):
        cases = [
            "不取消",
            "不要停止",
            "先别 cancel",
            "不是让你 stop",
            "不用 abort",
            "取消一下这个文件名",
            "停止输出 markdown",
            "算了我再想想",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertFalse(self.is_abort(text))

    def test_normalize_only_strips_layout_and_trailing_punctuation(self):
        self.assertEqual(self.normalize("  STOP!!!  "), "stop")
        self.assertEqual(self.normalize("取消。。"), "取消")
        self.assertEqual(self.normalize("先别 cancel"), "先别 cancel")


if __name__ == "__main__":
    unittest.main()
