from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from src.agent_runtime.tool_catalog import ToolCatalog
from src.agent_runtime.tool_profiles import merge_tool_profile, normalize_tool_profile
from src.prompting.prompt_manager import PromptManager
from src.platform_core.session_service import SessionService
from src.conversation.fast_path_config import (
    FAST_PATH_ENV_VAR,
    FAST_PATH_MODE_DISABLED,
    FAST_PATH_MODE_FULL,
    FastPathOutcome,
    INTENT_DISABLED,
    INTENT_NONE,
    INTENT_PENDING_RESOLUTION,
    read_fast_path_mode,
)
from src.conversation.fast_path_rewrite import FastPathRewriteHelper
from src.conversation.fast_path_resolution import WorkspacePathResolver

if TYPE_CHECKING:
    from src.ai.llm_service import LLMService


class FastPathRouter:
    """处理不需要完整 agent loop 的确定性请求。"""

    def __init__(
        self,
        tool_catalog: ToolCatalog,
        session_service: SessionService,
        llm_service: Optional["LLMService"] = None,
    ) -> None:
        self.tool_catalog = tool_catalog
        self.session_service = session_service
        self.llm_service = llm_service
        self.prompt_manager = PromptManager(str(self.tool_catalog.runtime.workspace_root))
        self.path_resolver = WorkspacePathResolver(self.tool_catalog.runtime, self.session_service)
        self.rewrite_helper = FastPathRewriteHelper(
            self.tool_catalog.runtime,
            self.llm_service,
            self.prompt_manager,
        )

    def try_handle(self, user_id: str, message: str) -> FastPathOutcome:
        message = (message or "").strip()
        if not message:
            return FastPathOutcome.miss()

        if read_fast_path_mode() == FAST_PATH_MODE_DISABLED:
            return FastPathOutcome.miss(INTENT_DISABLED)

        # Phase A 完成后，FastPath 不再做消息归一化或意图识别——所有非空消息直接落到
        # agent loop；FastPathRouter 只剩候选解析状态机入口（try_handle_pending_resolution）。
        return FastPathOutcome.miss()

    def try_handle_pending_resolution(self, user_id: str, message: str) -> Optional[str]:
        pending = self.session_service.get_workspace_browser_pending(user_id)
        if not pending:
            pending = self.session_service.get_pending_workspace_resolution(user_id)
        if not pending:
            return None

        choice = self.path_resolver.parse_resolution_choice(message)
        if choice is None:
            return None

        if choice == 0:
            self.session_service.clear_pending_workspace_resolution(user_id)
            return "已取消这次文件定位。你可以直接给我更准确的文件名或路径。"

        candidates = pending.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            self.session_service.clear_pending_workspace_resolution(user_id)
            return "这次文件定位候选已经失效，请重新描述一次你的请求。"

        index = choice - 1
        if index < 0 or index >= len(candidates):
            return f"候选只有 {len(candidates)} 个，请回复 1 到 {len(candidates)}，或回复“不是”。"

        candidate = candidates[index]
        path = str(candidate.get("path", "")).strip()
        target_type = str(candidate.get("type", "")).strip() or "file"
        self.session_service.clear_pending_workspace_resolution(user_id)
        return self._execute_resolved_action(
            user_id=user_id,
            action=str(pending.get("action", "")).strip(),
            path=path,
            payload=pending.get("payload", {}),
            target_type=target_type,
        )

    def _promote_tool_profile(self, user_id: str, inferred: str) -> None:
        current = self.session_service.get_tool_profile(user_id)
        merged = merge_tool_profile(current, inferred)
        if merged != normalize_tool_profile(current):
            self.session_service.set_tool_profile(user_id, merged)

    def _remember_browse_result(
        self,
        user_id: str,
        *,
        intent: str,
        path: str,
        target_type: str,
        reply: str,
        query: str = "",
        line_position: str = "",
    ) -> str:
        if not reply.startswith(
            (
                "文件不存在",
                "路径不存在",
                "目标不是",
                "源路径不存在",
                "路径不允许访问",
                "未找到待替换文本",
                "替换范围无效",
            )
        ):
            self.session_service.set_workspace_browser_focus(
                user_id,
                path=path,
                target_type=target_type,
                intent=intent,
                query=query,
                line_position=line_position,
            )
        return reply

    def _execute_resolved_action(
        self,
        *,
        user_id: str,
        action: str,
        path: str,
        payload: Any,
        target_type: str,
    ) -> str:
        payload = payload if isinstance(payload, dict) else {}

        if action == "read_file":
            self._promote_tool_profile(user_id, "workspace_read")
            reply = self.tool_catalog.runtime.read_file(path)
        elif action == "read_file_line":
            self._promote_tool_profile(user_id, "workspace_read")
            reply = self.tool_catalog.runtime.read_file_line(path, str(payload.get("position", "last")))
        elif action == "list_directory":
            self._promote_tool_profile(user_id, "workspace_read")
            reply = self.tool_catalog.runtime.list_directory(path)
        elif action == "search_files":
            self._promote_tool_profile(user_id, "workspace_read")
            query = str(payload.get("query", "")).strip()
            reply = self.tool_catalog.runtime.search_files(query, path)
            return self._remember_browse_result(
                user_id,
                intent="browse_search",
                path=path,
                target_type="dir",
                reply=reply,
                query=query,
            )
        elif action == "preview_edit_file":
            self._promote_tool_profile(user_id, "workspace_edit")
            reply = self.tool_catalog.runtime.preview_edit_file(
                path,
                str(payload.get("find_text", "")),
                str(payload.get("replace_text", "")),
                owner_user_id=user_id,
            )
        elif action == "preview_write_file":
            self._promote_tool_profile(user_id, "workspace_edit")
            reply = self.tool_catalog.runtime.preview_write_file(
                path,
                str(payload.get("content", "")),
                owner_user_id=user_id,
            )
        elif action == "preview_append_file":
            self._promote_tool_profile(user_id, "workspace_edit")
            reply = self.tool_catalog.runtime.preview_append_file(
                path,
                str(payload.get("content", "")),
                position=str(payload.get("position", "end")),
                owner_user_id=user_id,
            )
        elif action == "rewrite_block":
            self._promote_tool_profile(user_id, "workspace_edit")
            reply = self.rewrite_helper.handle_block_rewrite(
                user_id,
                path,
                str(payload.get("target", "第一段")),
                str(payload.get("instruction", "改短一点")),
            )
        elif action == "rename_path":
            self._promote_tool_profile(user_id, "workspace_edit")
            reply = self.tool_catalog.runtime.rename_path(path, str(payload.get("target_path", "")))
        else:
            return "这次文件定位候选已经失效，请重新描述一次你的请求。"

        intent = {
            "read_file": "browse_read",
            "read_file_line": "browse_read_line",
            "list_directory": "browse_list",
        }.get(action, "")
        return self._remember_browse_result(
            user_id,
            intent=intent,
            path=path,
            target_type="dir" if target_type == "dir" else "file",
            reply=reply,
            line_position=str(payload.get("position", "")),
        )
