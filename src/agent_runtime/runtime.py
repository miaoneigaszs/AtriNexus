from __future__ import annotations

import difflib
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import uuid


@dataclass(frozen=True)
class CommandExecutionPolicy:
    """轻量命令执行策略。"""

    shell_operator_tokens: Tuple[str, ...] = ("|", "||", "&", "&&", ";", ">", ">>", "<", "2>", "$(", "`")
    confirm_patterns: Tuple[str, ...] = (
        "rm -rf",
        "rm -fr",
        "rm -f",
        "rm -r",
        "del /s",
        "del /q",
        "rd /s",
        "rmdir /s",
        "move ",
        "mv ",
        "ren ",
        "rename ",
        "git clean",
        "git reset --hard",
        "git checkout --",
    )
    safe_bin_readonly: Tuple[str, ...] = (
        "find",
        "du",
        "wc",
        "stat",
        "file",
        "which",
        "env",
        "tree",
        "basename",
        "dirname",
        "realpath",
    )


# 命令分隔符：用于切割 "cmd1 | cmd2; cmd3 && cmd4" 之类的 pipeline，
# 每段都要独立判定是否 readonly。`&&` / `||` 必须放在单字符 `&` / `|` 之前，
# 否则 alternation 会先匹配单字符。
_PIPELINE_SEPARATORS = re.compile(r"\|\||&&|\||;|&")
# 重定向操作符：右侧是文件名，不是命令——只保留左侧做 readonly 判定。
_REDIRECT_OPERATORS = re.compile(r"2>|>>|>|<")


@dataclass(frozen=True)
class CommandExecutionPlan:
    mode: str
    reason: str = ""
    argv: Tuple[str, ...] = ()


class WorkspaceRuntime:
    """面向企微助手的轻量 workspace runtime。"""

    MAX_FILE_CHARS = 12000
    MAX_MATCHES = 50
    MAX_OUTPUT_CHARS = 12000
    DEFAULT_COMMAND_TIMEOUT = 20
    SKIP_DIRS = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "node_modules",
    }

    def __init__(
        self,
        workspace_root: str,
        command_policy: Optional[CommandExecutionPolicy] = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.command_policy = command_policy or CommandExecutionPolicy()
        self._pending_changes: Dict[str, Dict[str, str]] = {}
        self._pending_commands: Dict[str, Dict[str, str]] = {}

    def run_command(
        self,
        command: str,
        timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT,
        owner_user_id: Optional[str] = None,
    ) -> str:
        command = command.strip()
        if not command:
            return "缺少要执行的命令"

        plan = self._build_command_plan(command)

        if plan.mode == "deny":
            return "命令被拒绝：包含高风险或破坏性操作"

        if plan.mode == "confirm":
            confirm_id = self._store_pending_command(command, timeout_seconds, owner_user_id)
            return (
                f"这是高风险命令，暂不直接执行。\n"
                f"确认 ID: {confirm_id}\n"
                f"命令: {command}\n\n"
                "请选择：\n"
                f"1. 确认执行 {confirm_id}\n"
                f"2. 取消执行 {confirm_id}\n\n"
                "也可以直接回复：1 / 2 / 确定 / 取消"
            )

        return self._execute_command(command, timeout_seconds, plan)

    def confirm_pending_command(self, confirm_id: str, owner_user_id: Optional[str] = None) -> str:
        pending = self._pending_commands.get(confirm_id)
        if not pending:
            return f"未找到待确认命令: {confirm_id}"
        owner_error = self._check_pending_owner(pending, owner_user_id, "命令")
        if owner_error:
            return owner_error

        command = pending["command"]
        timeout_seconds = int(pending["timeout_seconds"])
        del self._pending_commands[confirm_id]
        return self._execute_command(command, timeout_seconds, self._build_command_plan(command))

    def discard_pending_command(self, confirm_id: str, owner_user_id: Optional[str] = None) -> str:
        if confirm_id not in self._pending_commands:
            return f"未找到待取消命令: {confirm_id}"
        owner_error = self._check_pending_owner(self._pending_commands[confirm_id], owner_user_id, "命令")
        if owner_error:
            return owner_error
        del self._pending_commands[confirm_id]
        return f"已取消待执行命令: {confirm_id}"

    def _execute_command(
        self,
        command: str,
        timeout_seconds: int,
        plan: Optional[CommandExecutionPlan] = None,
    ) -> str:
        timeout = max(1, min(int(timeout_seconds), 120))
        plan = plan or self._build_command_plan(command)
        try:
            if plan.mode == "direct":
                completed = subprocess.run(
                    list(plan.argv),
                    shell=False,
                    cwd=str(self.workspace_root),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    timeout=timeout,
                )
            else:
                completed = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(self.workspace_root),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                    timeout=timeout,
                )
        except subprocess.TimeoutExpired:
            return f"命令执行超时（>{timeout}s）: {command}"
        except Exception as exc:
            return f"命令执行失败: {exc}"

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        mode_label = {
            "direct": "direct",
            "shell-safe": "shell-readonly",
        }.get(plan.mode, "confirmed-shell")
        sections = [
            f"命令: {command}",
            f"退出码: {completed.returncode}",
            f"执行模式: {mode_label}",
        ]
        if stdout:
            sections.append(f"标准输出:\n{self._truncate_text(stdout, self.MAX_OUTPUT_CHARS)}")
        if stderr:
            sections.append(f"标准错误:\n{self._truncate_text(stderr, self.MAX_OUTPUT_CHARS)}")
        if not stdout and not stderr:
            sections.append("命令执行完成，无输出")
        return "\n\n".join(sections)

    def list_directory(self, path: str = ".") -> str:
        target, error = self.resolve_path_or_error(path)
        if error:
            return error
        if not target.exists():
            return f"路径不存在: {path}"
        if not target.is_dir():
            return f"目标不是目录: {path}"

        entries = sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        lines: List[str] = [f"目录: {self.to_relative(target)}"]
        for entry in entries[:200]:
            kind = "dir" if entry.is_dir() else "file"
            lines.append(f"- [{kind}] {entry.name}")
        if len(entries) > 200:
            lines.append(f"... 其余 {len(entries) - 200} 项已省略")
        return "\n".join(lines)

    def read_file(self, path: str, offset: Optional[int] = None, limit: Optional[int] = None) -> str:
        """读取文件，行号从 1 起按 offset/limit 分页；输出带行号（对标 Claude Code Read）。"""
        target, error = self.resolve_path_or_error(path)
        if error:
            return error
        if not target.exists():
            return f"文件不存在: {path}"
        if not target.is_file():
            return f"目标不是文件: {path}"

        text = target.read_text(encoding="utf-8", errors="ignore")
        all_lines = text.splitlines()
        total_lines = len(all_lines)

        if total_lines == 0:
            return f"文件: {self.to_relative(target)}\n\n[文件为空]"

        start_idx = max(0, (offset or 1) - 1)
        if start_idx >= total_lines:
            return (
                f"文件: {self.to_relative(target)} (共 {total_lines} 行)\n\n"
                f"[offset={offset} 超过文件末尾]"
            )

        end_idx = total_lines if not limit or limit <= 0 else min(total_lines, start_idx + limit)
        sliced = all_lines[start_idx:end_idx]

        numbered_body = "\n".join(
            f"{start_idx + i + 1:>6}\t{line}" for i, line in enumerate(sliced)
        )

        total_chars_used = sum(len(line) for line in sliced) + len(sliced)
        truncated_by_chars = False
        if total_chars_used > self.MAX_FILE_CHARS:
            kept_lines: List[str] = []
            running = 0
            for i, line in enumerate(sliced):
                addition = len(line) + 1
                if running + addition > self.MAX_FILE_CHARS and kept_lines:
                    break
                kept_lines.append(f"{start_idx + i + 1:>6}\t{line}")
                running += addition
            numbered_body = "\n".join(kept_lines)
            end_idx = start_idx + len(kept_lines)
            truncated_by_chars = True

        last_line_no = start_idx + (end_idx - start_idx)
        header = (
            f"文件: {self.to_relative(target)} "
            f"(第 {start_idx + 1}-{last_line_no} 行，共 {total_lines} 行)"
        )
        suffix_parts: List[str] = []
        if last_line_no < total_lines:
            suffix_parts.append(
                f"[仍剩 {total_lines - last_line_no} 行未读，"
                f"用 offset={last_line_no + 1} 继续读取]"
            )
        if truncated_by_chars:
            suffix_parts.append("[本段按字符数截断]")
        suffix = ("\n\n" + "\n".join(suffix_parts)) if suffix_parts else ""
        return f"{header}\n\n{numbered_body}{suffix}"

    def read_file_line(self, path: str, position: str = "last") -> str:
        """读取文件首行或末行，适合简单、确定性的文件提问。"""
        target, error = self.resolve_path_or_error(path)
        if error:
            return error
        if not target.exists():
            return f"文件不存在: {path}"
        if not target.is_file():
            return f"目标不是文件: {path}"

        lines = target.read_text(encoding="utf-8", errors="ignore").splitlines()
        if not lines:
            return f"文件: {self.to_relative(target)}\n\n[文件为空]"

        normalized = "first" if position == "first" else "last"
        line = lines[0] if normalized == "first" else lines[-1]
        label = "首行" if normalized == "first" else "末行"
        return f"文件: {self.to_relative(target)}\n{label}: {line}"

    def search_files(self, query: str, path: str = ".") -> str:
        query = query.strip()
        if not query:
            return "缺少搜索关键词"

        root, error = self.resolve_path_or_error(path)
        if error:
            return error
        if not root.exists():
            return f"路径不存在: {path}"

        matches: List[str] = []
        for file_path in self.iter_files(root):
            try:
                with file_path.open("r", encoding="utf-8", errors="ignore") as file_obj:
                    for lineno, line in enumerate(file_obj, start=1):
                        if query.lower() not in line.lower():
                            continue
                        snippet = line.strip()
                        matches.append(f"{self.to_relative(file_path)}:{lineno}: {snippet}")
                        if len(matches) >= self.MAX_MATCHES:
                            return "\n".join(matches + ["[结果过多，已截断]"])
            except OSError:
                continue

        if not matches:
            return f"未找到包含「{query}」的内容"
        return "\n".join(matches)

    def preview_write_file(self, path: str, content: str, owner_user_id: Optional[str] = None) -> str:
        target, error = self.resolve_path_or_error(path)
        if error:
            return error
        old_text = ""
        if target.exists():
            if not target.is_file():
                return f"目标不是文件: {path}"
            old_text = target.read_text(encoding="utf-8", errors="ignore")

        change_id = self._store_pending_change(target, old_text, content, owner_user_id)
        diff = self._build_diff(target, old_text, content)
        return self._format_preview(change_id, target, diff)

    def preview_append_file(
        self,
        path: str,
        content: str,
        position: str = "end",
        owner_user_id: Optional[str] = None,
    ) -> str:
        """预览在文件头部或尾部追加内容。"""
        target, error = self.resolve_path_or_error(path)
        if error:
            return error
        if not target.exists():
            return f"文件不存在: {path}"
        if not target.is_file():
            return f"目标不是文件: {path}"

        old_text = target.read_text(encoding="utf-8", errors="ignore")
        normalized_position = position.strip().lower()
        if normalized_position not in {"start", "end"}:
            return "追加位置无效，只支持 start 或 end"

        if normalized_position == "start":
            new_text = content + old_text
        else:
            new_text = old_text + content

        change_id = self._store_pending_change(target, old_text, new_text, owner_user_id)
        diff = self._build_diff(target, old_text, new_text)
        return self._format_preview(change_id, target, diff)

    def preview_replace_span(
        self,
        path: str,
        start_index: int,
        end_index: int,
        replacement_text: str,
        owner_user_id: Optional[str] = None,
    ) -> str:
        """预览按字符范围替换文件中的一段内容。"""
        target, error = self.resolve_path_or_error(path)
        if error:
            return error
        if not target.exists():
            return f"文件不存在: {path}"
        if not target.is_file():
            return f"目标不是文件: {path}"

        old_text = target.read_text(encoding="utf-8", errors="ignore")
        if start_index < 0 or end_index < start_index or end_index > len(old_text):
            return "替换范围无效"

        new_text = old_text[:start_index] + replacement_text + old_text[end_index:]
        change_id = self._store_pending_change(target, old_text, new_text, owner_user_id)
        diff = self._build_diff(target, old_text, new_text)
        return self._format_preview(change_id, target, diff)

    def preview_edit_file(
        self,
        path: str,
        find_text: str,
        replace_text: str,
        owner_user_id: Optional[str] = None,
    ) -> str:
        if not find_text:
            return "缺少待替换文本"

        target, error = self.resolve_path_or_error(path)
        if error:
            return error
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
        change_id = self._store_pending_change(target, old_text, new_text, owner_user_id)
        diff = self._build_diff(target, old_text, new_text)
        return self._format_preview(change_id, target, diff)

    def rename_path(self, source_path: str, target_path: str) -> str:
        source, error = self.resolve_path_or_error(source_path)
        if error:
            return error
        target, error = self.resolve_path_or_error(target_path)
        if error:
            return error
        if not source.exists():
            return f"源路径不存在: {source_path}"
        if target.exists():
            return f"目标路径已存在: {target_path}"

        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        return f"已重命名: {self.to_relative(source)} -> {self.to_relative(target)}"

    def apply_pending_change(self, change_id: str, owner_user_id: Optional[str] = None) -> str:
        pending = self._pending_changes.get(change_id)
        if not pending:
            return f"未找到待应用变更: {change_id}"
        owner_error = self._check_pending_owner(pending, owner_user_id, "修改")
        if owner_error:
            return owner_error

        target, error = self.resolve_path_or_error(pending["path"])
        if error:
            return error
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(pending["new_text"], encoding="utf-8")
        del self._pending_changes[change_id]
        return f"已应用变更: {change_id} -> {self.to_relative(target)}"

    def discard_pending_change(self, change_id: str, owner_user_id: Optional[str] = None) -> str:
        if change_id not in self._pending_changes:
            return f"未找到待丢弃变更: {change_id}"
        owner_error = self._check_pending_owner(self._pending_changes[change_id], owner_user_id, "修改")
        if owner_error:
            return owner_error
        del self._pending_changes[change_id]
        return f"已丢弃变更: {change_id}"

    def get_latest_pending_change_id(self, owner_user_id: Optional[str] = None) -> Optional[str]:
        """返回当前用户最近一次待确认修改的 ID。"""
        return self._get_latest_pending_id(self._pending_changes, owner_user_id)

    def get_latest_pending_command_id(self, owner_user_id: Optional[str] = None) -> Optional[str]:
        """返回当前用户最近一次待确认命令的 ID。"""
        return self._get_latest_pending_id(self._pending_commands, owner_user_id)

    def _resolve_path(self, raw_path: str) -> Path:
        candidate = (self.workspace_root / raw_path).resolve()
        if os.path.commonpath([str(self.workspace_root), str(candidate)]) != str(self.workspace_root):
            raise ValueError("路径超出 workspace 范围")
        return candidate

    def resolve_path_or_error(self, raw_path: str) -> Tuple[Optional[Path], Optional[str]]:
        try:
            return self._resolve_path(raw_path), None
        except ValueError:
            return None, "路径超出 workspace 范围"

    def to_relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.workspace_root))
        except ValueError:
            return str(path)

    def iter_files(self, root: Path):
        if root.is_file():
            yield root
            return

        for current_root, dirnames, filenames in os.walk(root):
            dirnames[:] = [name for name in dirnames if name not in self.SKIP_DIRS]
            for filename in filenames:
                yield Path(current_root) / filename

    MAX_GLOB_RESULTS = 200

    def glob_paths(self, pattern: str, path: str = ".") -> str:
        """按 glob 模式搜文件/目录路径，只返路径清单（不读内容）。

        pattern 支持标准 glob（`*.py` / `src/**/*.md`）；path 指定 glob 根，
        默认 workspace 根。SKIP_DIRS 里的目录（.git / node_modules 等）自动跳过。
        """
        cleaned_pattern = (pattern or "").strip()
        if not cleaned_pattern:
            return "缺少 glob 模式"

        root, error = self.resolve_path_or_error(path)
        if error:
            return error
        if not root.exists():
            return f"路径不存在: {path}"

        try:
            raw_matches = sorted(root.glob(cleaned_pattern))
        except (ValueError, OSError) as exc:
            return f"glob 失败: {exc}"

        filtered: List[str] = []
        for candidate in raw_matches:
            try:
                relative = candidate.relative_to(self.workspace_root)
            except ValueError:
                continue
            parts = relative.parts
            if any(part in self.SKIP_DIRS for part in parts):
                continue
            filtered.append(str(relative).replace("\\", "/"))
            if len(filtered) >= self.MAX_GLOB_RESULTS:
                break

        if not filtered:
            return f"未匹配任何路径: pattern={cleaned_pattern!r}, path={path!r}"

        lines = [f"匹配 {len(filtered)} 项（pattern={cleaned_pattern!r}, path={path!r}）:"]
        lines.extend(filtered)
        if len(raw_matches) > self.MAX_GLOB_RESULTS:
            lines.append(
                f"[仅保留前 {self.MAX_GLOB_RESULTS} 项，"
                f"原始匹配共 {len(raw_matches)} 项，请缩小 pattern 或 path]"
            )
        return "\n".join(lines)

    def _truncate_text(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n\n[输出过长，已截断]"

    def _build_command_plan(self, command: str) -> CommandExecutionPlan:
        normalized = " ".join(command.lower().split())
        if any(pattern in normalized for pattern in self.command_policy.confirm_patterns):
            return CommandExecutionPlan(mode="confirm", reason="confirm-pattern")
        if any(token in command for token in self.command_policy.shell_operator_tokens):
            if self._is_readonly_pipeline(command):
                return CommandExecutionPlan(mode="shell-safe", reason="safe-readonly")
            return CommandExecutionPlan(mode="confirm", reason="shell-operator")

        try:
            argv = tuple(shlex.split(command, posix=os.name != "nt"))
        except ValueError:
            return CommandExecutionPlan(mode="confirm", reason="parse-failed")

        if not argv:
            return CommandExecutionPlan(mode="deny", reason="empty")

        return CommandExecutionPlan(mode="direct", reason="direct", argv=argv)

    def _is_readonly_pipeline(self, command: str) -> bool:
        """Return True 当整条 shell 命令里每个管道段都只调 SAFE_BIN_READONLY。

        接受：`|`、`;`、`&&`、`||`、`&`，以及 `>`、`>>`、`<`、`2>` 这类把文件名作为
        RHS 的重定向（LHS 的命令会被单独校验）。拒绝：`$(...)` 或 backtick 命令替换
        ——里面可以藏任何东西，一律走确认路径。
        """
        if "$(" in command or "`" in command:
            return False

        segments = _PIPELINE_SEPARATORS.split(command)
        saw_any = False
        for segment in segments:
            command_part = _REDIRECT_OPERATORS.split(segment, maxsplit=1)[0]
            tokens = command_part.strip().split()
            if not tokens:
                continue
            if not self._is_readonly_command_tokens(tokens):
                return False
            saw_any = True
        return saw_any

    def _is_readonly_command_tokens(self, tokens: List[str]) -> bool:
        """校验一段 argv 是否落在 SAFE_BIN_READONLY 里。

        特殊处理 `env` 前缀：`env FOO=bar realcmd` —— 真正跑的是 realcmd，必须
        realcmd 也在白名单里。`env` 单独不带参数只是打印环境变量，保留为 readonly。
        """
        if not tokens:
            return False
        readonly = set(self.command_policy.safe_bin_readonly)
        name = tokens[0].rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
        if name != "env":
            return name in readonly

        if len(tokens) == 1:
            return True
        for tok in tokens[1:]:
            if "=" in tok and not tok.startswith("-") and not tok.startswith("="):
                continue
            real = tok.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
            return real in readonly
        return True

    def _store_pending_command(self, command: str, timeout_seconds: int, owner_user_id: Optional[str]) -> str:
        confirm_id = uuid.uuid4().hex[:12]
        self._pending_commands[confirm_id] = {
            "command": command,
            "timeout_seconds": str(timeout_seconds),
            "owner_user_id": owner_user_id or "",
        }
        return confirm_id

    def _store_pending_change(
        self,
        target: Path,
        old_text: str,
        new_text: str,
        owner_user_id: Optional[str],
    ) -> str:
        change_id = uuid.uuid4().hex[:12]
        self._pending_changes[change_id] = {
            "path": str(self.to_relative(target)),
            "old_text": old_text,
            "new_text": new_text,
            "owner_user_id": owner_user_id or "",
        }
        return change_id

    def _check_pending_owner(
        self,
        pending: Dict[str, str],
        owner_user_id: Optional[str],
        action_name: str,
    ) -> Optional[str]:
        pending_owner = pending.get("owner_user_id", "")
        if pending_owner and owner_user_id and pending_owner != owner_user_id:
            return f"该待确认{action_name}不属于当前用户"
        return None

    def _get_latest_pending_id(
        self,
        store: Dict[str, Dict[str, str]],
        owner_user_id: Optional[str],
    ) -> Optional[str]:
        for pending_id, payload in reversed(list(store.items())):
            pending_owner = payload.get("owner_user_id", "")
            if owner_user_id and pending_owner and pending_owner != owner_user_id:
                continue
            if owner_user_id and not pending_owner:
                continue
            return pending_id
        return None

    def _build_diff(self, target: Path, old_text: str, new_text: str) -> str:
        diff_lines = difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile=f"a/{self.to_relative(target)}",
            tofile=f"b/{self.to_relative(target)}",
            lineterm="",
        )
        diff = "\n".join(diff_lines)
        return diff or "[无差异]"

    def _format_preview(self, change_id: str, target: Path, diff: str) -> str:
        return (
            f"待审批变更 ID: {change_id}\n"
            f"目标文件: {self.to_relative(target)}\n\n"
            f"{diff}\n\n"
            "请选择：\n"
            f"1. 确认修改 {change_id}\n"
            f"2. 取消修改 {change_id}\n\n"
            "也可以直接回复：1 / 2 / 确定 / 取消。"
        )
