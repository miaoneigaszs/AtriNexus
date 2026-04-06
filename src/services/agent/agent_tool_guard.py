from __future__ import annotations

import contextvars
import difflib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage


logger = logging.getLogger("wecom")

TOOL_OVERVIEW_HINTS = (
    "有哪些工具",
    "有什么工具",
    "能用什么工具",
    "可以用什么工具",
    "能做什么",
    "会什么",
    "能力有哪些",
)
MAX_TOOL_RESULT_CHARS = 4000
MAX_TOOL_ARG_CHARS = 500
MAX_TOOL_REPEAT_COUNT = 2
MAX_TOOL_HISTORY = 30
WORKSPACE_PATH_TOOL_KEYS = {"path", "source_path", "target_path"}
RUN_COMMAND_TOOL_NAME = "run_command"
READ_FILE_TOOL_NAME = "read_file"
LIST_DIRECTORY_TOOL_NAME = "list_directory"
SEARCH_FILES_TOOL_NAME = "search_files"
EXPLORATION_TOOL_NAMES = {LIST_DIRECTORY_TOOL_NAME, SEARCH_FILES_TOOL_NAME}
MAX_LIST_DIRECTORY_LINES = 40
MAX_SEARCH_RESULT_LINES = 20
MAX_READ_FILE_CHARS = 2500

TOOL_LOOP_STATE: contextvars.ContextVar[Dict[str, Any] | None] = contextvars.ContextVar(
    "tool_loop_state",
    default=None,
)


class AgentToolGuard:
    """集中处理工具调用前后的修正、护栏和结果整形。"""

    def __init__(self, tool_catalog: Any) -> None:
        self.tool_catalog = tool_catalog

    def create_loop_state(self) -> Dict[str, Any]:
        return {"counts": {}, "recent": []}

    def set_loop_state(self, value: Dict[str, Any]):
        return TOOL_LOOP_STATE.set(value)

    def reset_loop_state(self, token) -> None:
        TOOL_LOOP_STATE.reset(token)

    def build_tool_middleware(self):
        @wrap_tool_call
        async def managed_tool_call(request, handler):
            tool_name = request.tool_call.get("name", "<unknown>")
            tool_args = request.tool_call.get("args", {})
            repaired_args, repair_message = self.repair_tool_args(tool_name, tool_args)
            if repaired_args != tool_args:
                request.tool_call["args"] = repaired_args
            tool_args = request.tool_call.get("args", {})

            validation_error = self.validate_tool_args(tool_name, tool_args)
            if validation_error:
                logger.warning(
                    "Tool call blocked by validation: name=%s args=%s error=%s",
                    tool_name,
                    self.summarize_tool_args(tool_args),
                    validation_error,
                )
                return ToolMessage(
                    content=validation_error,
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )

            loop_guard_message = self.check_tool_loop(tool_name, tool_args)
            if loop_guard_message:
                logger.warning(
                    "Tool call blocked by loop guard: name=%s args=%s",
                    tool_name,
                    self.summarize_tool_args(tool_args),
                )
                return ToolMessage(
                    content=loop_guard_message,
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )

            logger.info(
                "Tool call start: name=%s args=%s%s",
                tool_name,
                self.summarize_tool_args(tool_args),
                f" repair={repair_message}" if repair_message else "",
            )
            try:
                response = await handler(request)
            except Exception as exc:
                logger.warning("Tool call failed: name=%s error=%s", tool_name, exc)
                return ToolMessage(
                    content=f"工具 {tool_name} 执行失败：{exc}",
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )

            if not isinstance(response, ToolMessage):
                logger.info("Tool call end: name=%s result_type=%s", tool_name, type(response).__name__)
                return response

            content = self.extract_tool_message_text(response)
            content = self.shape_tool_result(tool_name, tool_args, content)
            response = self.replace_tool_message_content(response, content)
            self.record_tool_outcome(tool_name, tool_args, response.status, content)
            logger.info(
                "Tool call end: name=%s status=%s content_chars=%s",
                tool_name,
                response.status,
                len(content),
            )
            if len(content) <= MAX_TOOL_RESULT_CHARS:
                return response

            return ToolMessage(
                content=self.truncate_tool_text(content),
                tool_call_id=response.tool_call_id,
                status=response.status,
                artifact=response.artifact,
                name=response.name,
                id=response.id,
            )

        return managed_tool_call

    def looks_like_tool_overview(self, message: str) -> bool:
        normalized = (message or "").strip()
        return any(hint in normalized for hint in TOOL_OVERVIEW_HINTS)

    def format_tool_overview(self, tool_bundle: Any) -> str:
        tool_names = [tool.name for tool in tool_bundle.tools]
        profile_text = "、".join(tool_bundle.profiles) if tool_bundle.profiles else "无"
        lines = [
            "我刚检查了当前这条消息下启用的工具。",
            "",
            "当前工具组：",
            profile_text,
            "",
            "当前可用工具：",
        ]
        for name in tool_names:
            lines.append(f"- {name}")
        if tool_bundle.detailed_summary_lines:
            lines.append("")
            lines.append("这些工具当前分别能做：")
            lines.extend(tool_bundle.detailed_summary_lines)
        return "\n".join(lines)

    def summarize_tool_args(self, tool_args: Any) -> str:
        raw = str(tool_args)
        if len(raw) <= MAX_TOOL_ARG_CHARS:
            return raw
        return raw[:MAX_TOOL_ARG_CHARS] + "...[truncated]"

    def repair_tool_args(self, tool_name: str, tool_args: Any) -> Tuple[Any, str]:
        if not isinstance(tool_args, dict):
            return tool_args, ""

        repaired = dict(tool_args)
        notes: List[str] = []

        if tool_name == RUN_COMMAND_TOOL_NAME:
            command = repaired.get("command")
            if isinstance(command, str):
                stripped = command.strip()
                if stripped != command:
                    repaired["command"] = stripped
                    notes.append("trim-command")
            return repaired, ", ".join(notes)

        if tool_name not in {
            LIST_DIRECTORY_TOOL_NAME,
            READ_FILE_TOOL_NAME,
            "preview_write_file",
            "preview_edit_file",
            "preview_append_file",
            "rename_path",
        }:
            return repaired, ""

        for key in WORKSPACE_PATH_TOOL_KEYS:
            value = repaired.get(key)
            if not isinstance(value, str):
                continue
            candidate, note = self.repair_workspace_path_argument(
                value,
                expect_dir=(tool_name == LIST_DIRECTORY_TOOL_NAME and key == "path"),
                allow_dir=(tool_name == "rename_path"),
            )
            if note:
                notes.append(f"{key}:{note}")
            if candidate != value:
                repaired[key] = candidate

        return repaired, ", ".join(notes)

    def repair_workspace_path_argument(
        self,
        value: str,
        *,
        expect_dir: bool,
        allow_dir: bool,
    ) -> Tuple[str, str]:
        normalized = str(value).strip().strip("`'\"“”‘’")
        if not normalized:
            return value, ""
        if normalized.lower() == "readme":
            return "README.md", "readme"

        runtime = self.tool_catalog.runtime
        candidate, error = runtime._resolve_path_or_error(normalized)
        if not error and candidate and candidate.exists():
            if expect_dir and candidate.is_dir():
                return normalized, ""
            if not expect_dir and (candidate.is_file() or (allow_dir and candidate.is_dir())):
                return normalized, ""

        file_candidate = self.find_workspace_candidate(normalized, expect_dir=expect_dir, allow_dir=allow_dir)
        if file_candidate:
            return file_candidate, "path-repaired"
        return normalized, ""

    def find_workspace_candidate(
        self,
        value: str,
        *,
        expect_dir: bool,
        allow_dir: bool,
    ) -> Optional[str]:
        runtime = self.tool_catalog.runtime
        query = value.replace("\\", "/").strip().strip("/")
        if not query:
            return None
        query_name = Path(query).name
        query_key = self.normalize_lookup_key(query_name)
        if not query_key:
            return None

        candidates: List[Tuple[float, str]] = []
        if not expect_dir:
            for file_path in runtime._iter_files(runtime.workspace_root):
                relative_path = runtime._to_relative(file_path)
                score = self.score_workspace_candidate(query_key, file_path.name, relative_path)
                if score >= 0.86:
                    candidates.append((score, relative_path))

        if expect_dir or allow_dir:
            for current_root, dirnames, _ in os.walk(runtime.workspace_root):
                dirnames[:] = [name for name in dirnames if name not in runtime.SKIP_DIRS]
                for dirname in dirnames:
                    relative_path = str(
                        Path(
                            os.path.relpath(
                                os.path.join(current_root, dirname),
                                runtime.workspace_root,
                            )
                        )
                    )
                    score = self.score_workspace_candidate(query_key, dirname, relative_path)
                    if score >= 0.86:
                        candidates.append((score, relative_path))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (-item[0], len(item[1])))
        best_score, best_path = candidates[0]
        if len(candidates) > 1 and abs(best_score - candidates[1][0]) < 0.03:
            return None
        return best_path

    def score_workspace_candidate(self, query_key: str, name: str, relative_path: str) -> float:
        name_key = self.normalize_lookup_key(name)
        path_key = self.normalize_lookup_key(relative_path)
        score = max(
            difflib.SequenceMatcher(None, query_key, name_key).ratio(),
            difflib.SequenceMatcher(None, query_key, path_key).ratio(),
        )
        if query_key == name_key:
            return 1.0
        if name_key.startswith(query_key):
            return max(score, 0.93)
        if query_key in name_key:
            return max(score, 0.88)
        return score

    def normalize_lookup_key(self, value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")

    def check_tool_loop(self, tool_name: str, tool_args: Any) -> Optional[str]:
        loop_state = TOOL_LOOP_STATE.get()
        if loop_state is None:
            return None

        signature = self.build_tool_signature(tool_name, tool_args)
        counts = loop_state.setdefault("counts", {})
        recent = loop_state.setdefault("recent", [])
        counts[signature] = int(counts.get(signature, 0)) + 1
        recent_signatures = loop_state.setdefault("recent_signatures", [])
        recent_signatures.append(signature)
        if len(recent_signatures) > MAX_TOOL_HISTORY:
            del recent_signatures[0]

        if counts[signature] > MAX_TOOL_REPEAT_COUNT:
            return (
                f"工具 {tool_name} 已重复尝试同样的参数多次，请停止绕圈，"
                "改用别的工具、直接给出结论，或先向用户确认目标。"
            )

        if len(recent_signatures) >= 4 and recent_signatures[-1] == recent_signatures[-3] and recent_signatures[-2] == recent_signatures[-4]:
            return (
                f"工具链出现来回循环：{tool_name}。请不要继续重复试探，"
                "直接总结当前已知信息，或先向用户确认下一步。"
            )

        if self.is_no_progress_tool_call(tool_name, tool_args, recent):
            return (
                f"工具 {tool_name} 继续试探也不会有新进展。"
                "请直接根据现有结果回答，或向用户确认具体目标。"
            )

        return None

    def record_tool_outcome(self, tool_name: str, tool_args: Any, status: str, content: str) -> None:
        loop_state = TOOL_LOOP_STATE.get()
        if loop_state is None:
            return

        recent = loop_state.setdefault("recent", [])
        recent.append(
            {
                "tool_name": tool_name,
                "signature": self.build_tool_signature(tool_name, tool_args),
                "status": status,
                "path": self.extract_primary_path(tool_args),
                "content_hash": hash(content),
            }
        )
        if len(recent) > MAX_TOOL_HISTORY:
            del recent[0]

    def is_no_progress_tool_call(
        self,
        tool_name: str,
        tool_args: Any,
        recent: List[Dict[str, Any]],
    ) -> bool:
        if tool_name not in EXPLORATION_TOOL_NAMES or not recent:
            return False

        recent_successes = [item for item in reversed(recent) if item.get("status") != "error"]
        if not recent_successes:
            return False

        if any(item.get("tool_name") == READ_FILE_TOOL_NAME for item in recent_successes[:3]):
            return True

        recent_explorations = [
            item for item in recent_successes[:4] if item.get("tool_name") in EXPLORATION_TOOL_NAMES
        ]
        if len(recent_explorations) < 3:
            return False

        current_path = self.extract_primary_path(tool_args)
        previous_paths = {
            str(item.get("path", "")).strip()
            for item in recent_explorations
            if str(item.get("path", "")).strip()
        }
        if current_path and current_path in previous_paths:
            return True
        return False

    def validate_tool_args(self, tool_name: str, tool_args: Any) -> Optional[str]:
        if not isinstance(tool_args, dict):
            return None

        if tool_name == SEARCH_FILES_TOOL_NAME:
            query = str(tool_args.get("query", "")).strip()
            normalized = "".join(ch for ch in query.lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
            if not query:
                return "search_files 缺少搜索关键词。"
            if len(normalized) < 2 and "." not in query and "/" not in query and "\\" not in query:
                return "search_files 的关键词太短，请给更具体一点的内容。"

        if tool_name in {
            LIST_DIRECTORY_TOOL_NAME,
            READ_FILE_TOOL_NAME,
            "preview_write_file",
            "preview_edit_file",
            "preview_append_file",
        }:
            path = str(tool_args.get("path", "")).strip()
            if not path:
                return f"{tool_name} 缺少路径参数。"

        if tool_name == "rename_path":
            source_path = str(tool_args.get("source_path", "")).strip()
            target_path = str(tool_args.get("target_path", "")).strip()
            if not source_path or not target_path:
                return "rename_path 需要同时提供 source_path 和 target_path。"
            if source_path == target_path:
                return "rename_path 的源路径和目标路径相同，不需要重复执行。"

        if tool_name == "preview_edit_file":
            find_text = str(tool_args.get("find_text", ""))
            replace_text = str(tool_args.get("replace_text", ""))
            if not find_text.strip():
                return "preview_edit_file 缺少待替换文本。"
            if find_text == replace_text:
                return "preview_edit_file 的替换前后文本相同，不需要执行。"

        return None

    def extract_primary_path(self, tool_args: Any) -> str:
        if not isinstance(tool_args, dict):
            return ""
        for key in ("path", "source_path", "target_path"):
            value = str(tool_args.get(key, "")).strip()
            if value:
                return value
        return ""

    def shape_tool_result(self, tool_name: str, tool_args: Any, content: str) -> str:
        if tool_name == LIST_DIRECTORY_TOOL_NAME:
            return self.shape_directory_result(content)
        if tool_name == SEARCH_FILES_TOOL_NAME:
            return self.shape_search_result(content)
        if tool_name == READ_FILE_TOOL_NAME:
            return self.shape_read_file_result(content, self.extract_primary_path(tool_args))
        return content

    def shape_directory_result(self, content: str) -> str:
        lines = content.splitlines()
        if len(lines) <= MAX_LIST_DIRECTORY_LINES:
            return content
        kept = lines[:MAX_LIST_DIRECTORY_LINES]
        omitted = max(0, len(lines) - len(kept))
        kept.append(f"... 其余 {omitted} 项已省略")
        return "\n".join(kept)

    def shape_search_result(self, content: str) -> str:
        lines = content.splitlines()
        if len(lines) <= MAX_SEARCH_RESULT_LINES:
            return content
        kept = lines[:MAX_SEARCH_RESULT_LINES]
        kept.append(f"[匹配结果较多，仅保留前 {MAX_SEARCH_RESULT_LINES} 条；如需继续，请缩小路径或关键词]")
        return "\n".join(kept)

    def shape_read_file_result(self, content: str, path: str) -> str:
        if len(content) <= MAX_READ_FILE_CHARS:
            return content

        header, separator, body = content.partition("\n\n")
        if not separator:
            return self.truncate_tool_text(content)

        trimmed_body = body[: MAX_READ_FILE_CHARS - len(header) - 120].rstrip()
        hint_path = path or "文件"
        return (
            f"{header}\n\n{trimmed_body}\n\n"
            f"[文件内容较长，已只保留前半段。若要继续，请指定更小范围，例如“{hint_path}最后一行”或“{hint_path} 某段内容”。]"
        )

    def replace_tool_message_content(self, message: ToolMessage, content: str) -> ToolMessage:
        if message.content == content:
            return message
        return ToolMessage(
            content=content,
            tool_call_id=message.tool_call_id,
            status=message.status,
            artifact=message.artifact,
            name=message.name,
            id=message.id,
        )

    def build_tool_signature(self, tool_name: str, tool_args: Any) -> str:
        try:
            serialized = json.dumps(tool_args, sort_keys=True, ensure_ascii=False, default=str)
        except TypeError:
            serialized = str(tool_args)
        return f"{tool_name}:{serialized}"

    def extract_tool_message_text(self, message: ToolMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
            return "\n".join(parts)
        return str(content)

    def truncate_tool_text(self, text: str) -> str:
        if len(text) <= MAX_TOOL_RESULT_CHARS:
            return text
        return text[:MAX_TOOL_RESULT_CHARS] + "\n\n[工具输出过长，已截断]"
