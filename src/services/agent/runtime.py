from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Dict, List
import uuid


class WorkspaceRuntime:
    """只读 workspace runtime。"""

    MAX_FILE_CHARS = 12000
    MAX_MATCHES = 50
    SKIP_DIRS = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
    }

    def __init__(self, workspace_root: str) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self._pending_changes: Dict[str, Dict[str, str]] = {}

    def list_directory(self, path: str = ".") -> str:
        target = self._resolve_path(path)
        if not target.exists():
            return f"路径不存在: {path}"
        if not target.is_dir():
            return f"目标不是目录: {path}"

        entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        lines: List[str] = [f"目录: {self._to_relative(target)}"]
        for entry in entries[:200]:
            kind = "dir" if entry.is_dir() else "file"
            lines.append(f"- [{kind}] {entry.name}")
        if len(entries) > 200:
            lines.append(f"... 其余 {len(entries) - 200} 项已省略")
        return "\n".join(lines)

    def read_file(self, path: str) -> str:
        target = self._resolve_path(path)
        if not target.exists():
            return f"文件不存在: {path}"
        if not target.is_file():
            return f"目标不是文件: {path}"

        text = target.read_text(encoding="utf-8", errors="ignore")
        if len(text) > self.MAX_FILE_CHARS:
            text = text[: self.MAX_FILE_CHARS]
            suffix = "\n\n[内容过长，已截断]"
        else:
            suffix = ""
        return f"文件: {self._to_relative(target)}\n\n{text}{suffix}"

    def search_files(self, query: str, path: str = ".") -> str:
        query = query.strip()
        if not query:
            return "缺少搜索关键词"

        root = self._resolve_path(path)
        if not root.exists():
            return f"路径不存在: {path}"

        matches: List[str] = []
        for file_path in self._iter_files(root):
            try:
                with file_path.open("r", encoding="utf-8", errors="ignore") as file_obj:
                    for lineno, line in enumerate(file_obj, start=1):
                        if query.lower() not in line.lower():
                            continue
                        snippet = line.strip()
                        matches.append(f"{self._to_relative(file_path)}:{lineno}: {snippet}")
                        if len(matches) >= self.MAX_MATCHES:
                            return "\n".join(matches + ["[结果过多，已截断]"])
            except OSError:
                continue

        if not matches:
            return f"未找到包含「{query}」的内容"
        return "\n".join(matches)

    def preview_write_file(self, path: str, content: str) -> str:
        target = self._resolve_path(path)
        old_text = ""
        if target.exists():
            if not target.is_file():
                return f"目标不是文件: {path}"
            old_text = target.read_text(encoding="utf-8", errors="ignore")

        change_id = self._store_pending_change(target, old_text, content)
        diff = self._build_diff(target, old_text, content)
        return self._format_preview(change_id, target, diff)

    def preview_edit_file(self, path: str, find_text: str, replace_text: str) -> str:
        if not find_text:
            return "缺少待替换文本"

        target = self._resolve_path(path)
        if not target.exists():
            return f"文件不存在: {path}"
        if not target.is_file():
            return f"目标不是文件: {path}"

        old_text = target.read_text(encoding="utf-8", errors="ignore")
        count = old_text.count(find_text)
        if count == 0:
            return "未找到待替换文本"
        if count > 1:
            return "待替换文本出现多次，请提供更精确的片段"

        new_text = old_text.replace(find_text, replace_text, 1)
        change_id = self._store_pending_change(target, old_text, new_text)
        diff = self._build_diff(target, old_text, new_text)
        return self._format_preview(change_id, target, diff)

    def apply_pending_change(self, change_id: str) -> str:
        pending = self._pending_changes.get(change_id)
        if not pending:
            return f"未找到待应用变更: {change_id}"

        target = self._resolve_path(pending["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(pending["new_text"], encoding="utf-8")
        del self._pending_changes[change_id]
        return f"已应用变更: {change_id} -> {self._to_relative(target)}"

    def discard_pending_change(self, change_id: str) -> str:
        if change_id not in self._pending_changes:
            return f"未找到待丢弃变更: {change_id}"
        del self._pending_changes[change_id]
        return f"已丢弃变更: {change_id}"

    def list_pending_changes(self) -> str:
        if not self._pending_changes:
            return "当前没有待审批变更"
        lines = ["待审批变更："]
        for change_id, pending in self._pending_changes.items():
            lines.append(f"- {change_id}: {pending['path']}")
        return "\n".join(lines)

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = (self.workspace_root / raw_path).resolve()
        if os.path.commonpath([str(self.workspace_root), str(candidate)]) != str(self.workspace_root):
            raise ValueError("路径超出 workspace 范围")
        return candidate

    def _to_relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.workspace_root))
        except ValueError:
            return str(path)

    def _iter_files(self, root: Path):
        if root.is_file():
            yield root
            return

        for current_root, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in self.SKIP_DIRS]
            for filename in filenames:
                yield Path(current_root) / filename

    def _store_pending_change(self, target: Path, old_text: str, new_text: str) -> str:
        change_id = str(uuid.uuid4())[:8]
        self._pending_changes[change_id] = {
            "path": str(self._to_relative(target)),
            "old_text": old_text,
            "new_text": new_text,
        }
        return change_id

    def _build_diff(self, target: Path, old_text: str, new_text: str) -> str:
        diff_lines = difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile=f"a/{self._to_relative(target)}",
            tofile=f"b/{self._to_relative(target)}",
            lineterm="",
        )
        diff = "\n".join(diff_lines)
        return diff or "[无差异]"

    def _format_preview(self, change_id: str, target: Path, diff: str) -> str:
        return (
            f"待审批变更 ID: {change_id}\n"
            f"目标文件: {self._to_relative(target)}\n\n"
            f"{diff}\n\n"
            f"如需真正落盘，后续必须显式审批并应用该变更。"
        )
