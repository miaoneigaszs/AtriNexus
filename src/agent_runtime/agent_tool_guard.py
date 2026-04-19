"""工具调用前后的修正、护栏、结果整形。

本模块**框架中立**——不再依赖 LangChain。逻辑通过 `AgentHooks` 协议的
`before_tool_call` / `after_tool_call` 方法暴露；LangChain 集成由
`agent_runtime/middleware.py` 的翻译层负责，Phase 4 自建 agent loop 后翻译层
消失，hook 直接被调用。

Loop 检测仍靠 contextvar 做逐 run 隔离：agent 运行前 set_loop_state，结束后
reset_loop_state，中间 record_tool_outcome 写入当前 run 的状态。
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from src.conversation.fast_path_resolution import WorkspacePathResolver

from src.agent_runtime.hooks import (
    AfterToolCallContext,
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
)
from src.agent_runtime.user_runtime import abort_requested


logger = logging.getLogger("wecom")

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


class _NoopSessionService:
    def get_last_workspace_target(self, _user_id: str) -> Dict[str, str]:
        return {}

    def set_pending_workspace_resolution(self, *_args, **_kwargs) -> None:
        return None


class AgentToolGuard:
    """集中处理工具调用前后的修正、护栏和结果整形；实现 AgentHooks 的两个 tool 钩子。"""

    def __init__(self, tool_catalog: Any) -> None:
        self.tool_catalog = tool_catalog

    # ── Loop 状态生命周期 ───────────────────────────────────────────────

    def create_loop_state(self) -> Dict[str, Any]:
        return {"counts": {}, "recent": []}

    def set_loop_state(self, value: Dict[str, Any]):
        return TOOL_LOOP_STATE.set(value)

    def reset_loop_state(self, token) -> None:
        TOOL_LOOP_STATE.reset(token)

    # ── AgentHooks 接口 ─────────────────────────────────────────────────

    async def before_tool_call(
        self, ctx: BeforeToolCallContext
    ) -> Optional[BeforeToolCallResult]:
        # 在任何工具调用前先检查用户是否要求取消；是则让 agent loop 捕获 CancelledError。
        if abort_requested():
            logger.info("Tool call aborted: name=%s reason=user-abort", ctx.tool_name)
            raise asyncio.CancelledError("user requested abort")

        repaired_args, repair_note = self.repair_tool_args(ctx.tool_name, ctx.args)
        effective_args = repaired_args if repaired_args != ctx.args else ctx.args

        validation_error = self.validate_tool_args(ctx.tool_name, effective_args)
        if validation_error:
            logger.warning(
                "Tool call blocked by validation: name=%s args=%s error=%s",
                ctx.tool_name,
                self.summarize_tool_args(effective_args),
                validation_error,
            )
            return BeforeToolCallResult(block=True, reason=validation_error)

        loop_guard_message = self.check_tool_loop(ctx.tool_name, effective_args)
        if loop_guard_message:
            logger.warning(
                "Tool call blocked by loop guard: name=%s args=%s",
                ctx.tool_name,
                self.summarize_tool_args(effective_args),
            )
            return BeforeToolCallResult(block=True, reason=loop_guard_message)

        logger.info(
            "Tool call start: name=%s args=%s%s",
            ctx.tool_name,
            self.summarize_tool_args(effective_args),
            f" repair={repair_note}" if repair_note else "",
        )
        if repaired_args != ctx.args:
            return BeforeToolCallResult(repaired_args=repaired_args)
        return None

    async def after_tool_call(
        self, ctx: AfterToolCallContext
    ) -> Optional[AfterToolCallResult]:
        shaped = self.shape_tool_result(ctx.tool_name, ctx.args, ctx.result_content)
        shaped = self._maybe_truncate(shaped)
        self.record_tool_outcome(
            ctx.tool_name,
            ctx.args,
            "error" if ctx.is_error else "ok",
            shaped,
        )
        logger.info(
            "Tool call end: name=%s status=%s content_chars=%s",
            ctx.tool_name,
            "error" if ctx.is_error else "ok",
            len(shaped),
        )
        if shaped != ctx.result_content:
            return AfterToolCallResult(content=shaped)
        return None

    # ── Arg repair ──────────────────────────────────────────────────────

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

        resolver = WorkspacePathResolver(self.tool_catalog.runtime, _NoopSessionService())
        repaired = resolver.repair_path_if_confident(
            normalized,
            expect_dir=expect_dir,
            allow_dir=allow_dir,
        )
        if repaired != normalized:
            return repaired, "path-repaired"
        return normalized, ""

    # ── Loop guard ──────────────────────────────────────────────────────

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

        if (
            len(recent_signatures) >= 4
            and recent_signatures[-1] == recent_signatures[-3]
            and recent_signatures[-2] == recent_signatures[-4]
        ):
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

    # ── Arg validation ──────────────────────────────────────────────────

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

    # ── Result shaping ──────────────────────────────────────────────────

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
            return self._maybe_truncate(content)

        trimmed_body = body[: MAX_READ_FILE_CHARS - len(header) - 120].rstrip()
        hint_path = path or "文件"
        return (
            f"{header}\n\n{trimmed_body}\n\n"
            f"[文件内容较长，已只保留前半段。若要读后续内容，再次调用 read_file "
            f"并指定 offset=N（下一段的起始行号）；或传 limit 限制每段行数。"
            f"路径：{hint_path}。]"
        )

    def _maybe_truncate(self, text: str) -> str:
        if len(text) <= MAX_TOOL_RESULT_CHARS:
            return text
        return text[:MAX_TOOL_RESULT_CHARS] + "\n\n[工具输出过长，已截断]"

    # ── Shared utilities ────────────────────────────────────────────────

    def build_tool_signature(self, tool_name: str, tool_args: Any) -> str:
        try:
            serialized = json.dumps(tool_args, sort_keys=True, ensure_ascii=False, default=str)
        except TypeError:
            serialized = str(tool_args)
        return f"{tool_name}:{serialized}"
