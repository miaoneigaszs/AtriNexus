from __future__ import annotations

import os
from pathlib import Path
from typing import List


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
