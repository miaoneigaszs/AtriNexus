"""PR19 — read_file offset/limit + glob 行为测试。

直接在 WorkspaceRuntime 上跑，绕开 DB 链——runtime 构造只要一个 workspace
路径，不依赖 SessionService / LLM。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.agent_runtime.runtime import WorkspaceRuntime


def _write_file(root: Path, rel: str, content: str) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


class ReadFileOffsetLimitTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.runtime = WorkspaceRuntime(str(self.root))

    def _ten_line_file(self, name: str = "ten.txt") -> str:
        content = "\n".join(f"line {i}" for i in range(1, 11))
        _write_file(self.root, name, content)
        return name

    def test_default_reads_whole_file_with_line_numbers(self):
        path = self._ten_line_file()
        reply = self.runtime.read_file(path)
        self.assertIn("1-10 行，共 10 行", reply)
        self.assertIn("     1\tline 1", reply)
        self.assertIn("    10\tline 10", reply)

    def test_offset_skips_leading_lines(self):
        path = self._ten_line_file()
        reply = self.runtime.read_file(path, offset=5)
        self.assertIn("5-10 行，共 10 行", reply)
        self.assertNotIn("line 4\n", reply)
        self.assertIn("     5\tline 5", reply)
        self.assertIn("    10\tline 10", reply)

    def test_limit_caps_returned_lines(self):
        path = self._ten_line_file()
        reply = self.runtime.read_file(path, offset=1, limit=3)
        self.assertIn("1-3 行", reply)
        self.assertIn("仍剩 7 行未读", reply)
        self.assertIn("offset=4", reply)

    def test_offset_plus_limit_windows_correctly(self):
        path = self._ten_line_file()
        reply = self.runtime.read_file(path, offset=4, limit=3)
        self.assertIn("4-6 行", reply)
        self.assertIn("仍剩 4 行未读", reply)
        self.assertIn("offset=7", reply)
        self.assertIn("     4\tline 4", reply)
        self.assertIn("     6\tline 6", reply)
        self.assertNotIn("line 3\n", reply)
        self.assertNotIn("line 7\n", reply)

    def test_offset_past_end_reports_range(self):
        path = self._ten_line_file()
        reply = self.runtime.read_file(path, offset=999)
        self.assertIn("offset=999 超过文件末尾", reply)
        self.assertIn("共 10 行", reply)

    def test_empty_file_is_handled(self):
        _write_file(self.root, "empty.txt", "")
        reply = self.runtime.read_file("empty.txt")
        self.assertIn("[文件为空]", reply)

    def test_missing_file_gives_user_message(self):
        reply = self.runtime.read_file("nope.txt")
        self.assertIn("文件不存在", reply)


class GlobPathsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.runtime = WorkspaceRuntime(str(self.root))
        _write_file(self.root, "a.py", "x")
        _write_file(self.root, "b.py", "y")
        _write_file(self.root, "src/inner.py", "z")
        _write_file(self.root, "README.md", "md")
        _write_file(self.root, ".git/ignored.py", "hidden")

    def test_top_level_py_pattern(self):
        reply = self.runtime.glob_paths("*.py")
        self.assertIn("a.py", reply)
        self.assertIn("b.py", reply)
        self.assertNotIn("src/inner.py", reply)

    def test_recursive_glob_covers_nested(self):
        reply = self.runtime.glob_paths("**/*.py")
        self.assertIn("a.py", reply)
        self.assertIn("src/inner.py", reply)

    def test_skip_dirs_are_excluded(self):
        reply = self.runtime.glob_paths("**/*.py")
        self.assertNotIn(".git/ignored.py", reply)

    def test_empty_pattern_rejected(self):
        self.assertIn("缺少 glob 模式", self.runtime.glob_paths(""))

    def test_no_match_reports_cleanly(self):
        reply = self.runtime.glob_paths("*.nonexistent")
        self.assertIn("未匹配任何路径", reply)

    def test_md_pattern_is_scoped(self):
        reply = self.runtime.glob_paths("*.md")
        self.assertIn("README.md", reply)
        self.assertNotIn("a.py", reply)


class ReadFileToolSpecTest(unittest.TestCase):
    def test_spec_lists_offset_and_limit_params(self):
        from pathlib import Path

        tool_catalog_path = Path(__file__).resolve().parents[1] / "agent_runtime" / "tool_catalog.py"
        source = tool_catalog_path.read_text(encoding="utf-8")
        self.assertIn('"offset"', source)
        self.assertIn('"limit"', source)
        self.assertIn("1-indexed starting line", source)


class GlobToolSpecTest(unittest.TestCase):
    def test_glob_tool_registered_in_catalog(self):
        from pathlib import Path

        tool_catalog_path = Path(__file__).resolve().parents[1] / "agent_runtime" / "tool_catalog.py"
        source = tool_catalog_path.read_text(encoding="utf-8")
        self.assertIn('name="glob"', source)
        self.assertIn("Find paths by glob pattern", source)

    def test_workspace_read_detailed_lines_mention_new_tools(self):
        from pathlib import Path

        tool_catalog_path = Path(__file__).resolve().parents[1] / "agent_runtime" / "tool_catalog.py"
        source = tool_catalog_path.read_text(encoding="utf-8")
        self.assertIn("- glob:", source)
        self.assertIn("- read_file: 读文件，带行号", source)


if __name__ == "__main__":
    unittest.main()
