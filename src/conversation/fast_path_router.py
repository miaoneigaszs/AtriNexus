from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from src.agent_runtime.tool_catalog import ToolCatalog
from src.agent_runtime.tool_profiles import merge_tool_profile, normalize_tool_profile
from src.prompting.prompt_manager import PromptManager
from src.platform_core.session_service import SessionService
from src.conversation.fast_path_intents import (
    WorkspaceBrowseRequest,
    extract_append_request,
    extract_block_rewrite_request,
    extract_followup_rename_target,
    extract_rename_paths,
    extract_replace_request,
    extract_rewrite_request,
    extract_workspace_browse_request,
    is_profile_overview,
    is_tool_overview,
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

    def try_handle(self, user_id: str, message: str) -> Optional[str]:
        message = (message or "").strip()
        if not message:
            return None

        self.path_resolver.begin(user_id)
        normalized_message = self.path_resolver.normalize_request_text(message)

        def dispatch_before_remembered_action(
            *,
            inferred_profile: str,
            request,
            target_type: str,
            remembered_path,
            action,
        ) -> Optional[str]:
            pending_reply = self.path_resolver.take_pending_reply()
            if pending_reply:
                return pending_reply
            if not request:
                return None
            self._promote_tool_profile(user_id, inferred_profile)
            self.session_service.set_last_workspace_target(user_id, remembered_path(request), target_type)
            return action(request)

        def dispatch_after_remembered_action(
            *,
            inferred_profile: str,
            request,
            target_type: str,
            remembered_path,
            blocked_prefixes: Tuple[str, ...],
            action,
        ) -> Optional[str]:
            pending_reply = self.path_resolver.take_pending_reply()
            if pending_reply:
                return pending_reply
            if not request:
                return None
            self._promote_tool_profile(user_id, inferred_profile)
            reply = action(request)
            if not reply.startswith(blocked_prefixes):
                self.session_service.set_last_workspace_target(user_id, remembered_path(request), target_type)
            return reply

        if is_tool_overview(normalized_message):
            return self._handle_tool_overview(user_id, normalized_message)

        if is_profile_overview(normalized_message):
            return self._handle_profile_overview(user_id, normalized_message)

        block_rewrite_request = self._extract_block_rewrite_request(normalized_message)
        dispatch_reply = dispatch_before_remembered_action(
            inferred_profile="workspace_edit",
            request=block_rewrite_request,
            target_type="file",
            remembered_path=lambda request: request[0],
            action=lambda request: self.rewrite_helper.handle_block_rewrite(
                user_id,
                request[0],
                request[1],
                request[2],
            ),
        )
        if dispatch_reply is not None:
            return dispatch_reply

        replace_request = self._extract_replace_request(normalized_message)
        dispatch_reply = dispatch_before_remembered_action(
            inferred_profile="workspace_edit",
            request=replace_request,
            target_type="file",
            remembered_path=lambda request: request[0],
            action=lambda request: self.tool_catalog.runtime.preview_edit_file(
                request[0],
                request[1],
                request[2],
                owner_user_id=user_id,
            ),
        )
        if dispatch_reply is not None:
            return dispatch_reply

        rewrite_request = self._extract_rewrite_request(normalized_message)
        dispatch_reply = dispatch_before_remembered_action(
            inferred_profile="workspace_edit",
            request=rewrite_request,
            target_type="file",
            remembered_path=lambda request: request[0],
            action=lambda request: self.tool_catalog.runtime.preview_write_file(
                request[0],
                request[1],
                owner_user_id=user_id,
            ),
        )
        if dispatch_reply is not None:
            return dispatch_reply

        append_request = self._extract_append_request(normalized_message)
        dispatch_reply = dispatch_before_remembered_action(
            inferred_profile="workspace_edit",
            request=append_request,
            target_type="file",
            remembered_path=lambda request: request[0],
            action=lambda request: self.tool_catalog.runtime.preview_append_file(
                request[0],
                request[1],
                position=request[2],
                owner_user_id=user_id,
            ),
        )
        if dispatch_reply is not None:
            return dispatch_reply

        rename_paths = self._extract_rename_paths(normalized_message)
        dispatch_reply = dispatch_after_remembered_action(
            inferred_profile="workspace_edit",
            request=rename_paths,
            target_type="file",
            remembered_path=lambda request: request[1],
            blocked_prefixes=("未找到源路径", "路径不允许访问", "目标路径无效"),
            action=lambda request: self.tool_catalog.runtime.rename_path(request[0], request[1]),
        )
        if dispatch_reply is not None:
            return dispatch_reply

        browse_request = self._extract_workspace_browse_request(user_id, normalized_message)
        if browse_request:
            browse_reply = self._handle_workspace_browse_request(user_id, browse_request)
            pending_reply = self.path_resolver.take_pending_reply()
            if pending_reply:
                return pending_reply
            if browse_reply is not None:
                return browse_reply

        return None

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

    def _handle_tool_overview(self, user_id: str, message: str) -> str:
        tool_profile = self.session_service.get_tool_profile(user_id)
        tool_bundle = self.tool_catalog.build_tool_bundle(
            user_id=user_id,
            message=message,
            allow_tools=True,
            tool_profile=tool_profile,
        )
        lines = [
            "我刚检查了当前会话下启用的能力。",
            "",
            f"当前能力档位：{tool_profile}",
            "当前工具组：",
            "、".join(tool_bundle.profiles) if tool_bundle.profiles else "无",
            "",
            "当前可用工具：",
        ]
        for tool in tool_bundle.tools:
            lines.append(f"- {tool.name}")
        if tool_bundle.detailed_summary_lines:
            lines.append("")
            lines.append("这些工具当前分别能做：")
            lines.extend(tool_bundle.detailed_summary_lines)
        return "\n".join(lines)

    def _handle_profile_overview(self, user_id: str, message: str) -> str:
        tool_profile = normalize_tool_profile(self.session_service.get_tool_profile(user_id))
        current_mode = self.session_service.get_current_mode(user_id)
        tool_bundle = self.tool_catalog.build_tool_bundle(
            user_id=user_id,
            message=message,
            allow_tools=True,
            tool_profile=tool_profile,
        )
        lines = [
            "我刚检查了当前会话状态。",
            "",
            f"当前模式：{current_mode}",
            f"当前能力档位：{tool_profile}",
            "当前工具组：",
            "、".join(tool_bundle.profiles) if tool_bundle.profiles else "无",
        ]
        if tool_bundle.compact_summary_lines:
            lines.append("")
            lines.append("当前能力边界：")
            lines.extend(tool_bundle.compact_summary_lines)
        return "\n".join(lines)

    def _promote_tool_profile(self, user_id: str, inferred: str) -> None:
        current = self.session_service.get_tool_profile(user_id)
        merged = merge_tool_profile(current, inferred)
        if merged != normalize_tool_profile(current):
            self.session_service.set_tool_profile(user_id, merged)

    def _extract_workspace_browse_request(
        self,
        user_id: str,
        message: str,
    ) -> Optional[WorkspaceBrowseRequest]:
        browser_state = self.session_service.get_workspace_browser_state(user_id)
        return extract_workspace_browse_request(message, browser_state)

    def _handle_workspace_browse_request(
        self,
        user_id: str,
        request: WorkspaceBrowseRequest,
    ) -> Optional[str]:
        browser_state = self.session_service.get_workspace_browser_state(user_id)
        focus = browser_state.get("focus", {}) if isinstance(browser_state, dict) else {}
        focus_path = str(focus.get("path", "")).strip()
        focus_type = str(focus.get("type", "")).strip()

        if request.intent == "browse_list":
            target_path = request.path_hint or (focus_path if focus_type == "dir" else "")
            if not target_path:
                return None
            resolved = self.path_resolver.resolve_path_hint(
                target_path,
                expect_dir=True,
                action="list_directory",
            )
            if resolved.status != "resolved":
                return None
            self._promote_tool_profile(user_id, "workspace_read")
            reply = self.tool_catalog.runtime.list_directory(resolved.path)
            return self._remember_browse_result(
                user_id,
                intent="browse_list",
                path=resolved.path,
                target_type="dir",
                reply=reply,
            )

        if request.intent == "browse_read":
            if request.reference_mode == "focus_file":
                target_path = focus_path if focus_type == "file" else ""
            else:
                target_path = request.path_hint or focus_path
            if not target_path:
                return None
            resolved = self.path_resolver.resolve_path_hint(
                target_path,
                expect_file=True,
                action="read_file",
            )
            if resolved.status != "resolved":
                return None
            self._promote_tool_profile(user_id, "workspace_read")
            reply = self.tool_catalog.runtime.read_file(resolved.path)
            return self._remember_browse_result(
                user_id,
                intent="browse_read",
                path=resolved.path,
                target_type="file",
                reply=reply,
            )

        if request.intent == "browse_read_line":
            line_position = request.line_position or "last"
            if request.reference_mode == "focus_file":
                target_path = focus_path if focus_type == "file" else ""
            else:
                target_path = request.path_hint or focus_path
            if not target_path:
                return None
            resolved = self.path_resolver.resolve_path_hint(
                target_path,
                expect_file=True,
                action="read_file_line",
                payload={"position": line_position},
            )
            if resolved.status != "resolved":
                return None
            self._promote_tool_profile(user_id, "workspace_read")
            reply = self.tool_catalog.runtime.read_file_line(resolved.path, line_position)
            return self._remember_browse_result(
                user_id,
                intent="browse_read_line",
                path=resolved.path,
                target_type="file",
                reply=reply,
                line_position=line_position,
            )

        if request.intent == "browse_search":
            if request.reference_mode == "focus_dir":
                if focus_type == "dir":
                    target_path = focus_path
                elif focus_type == "file":
                    parent = PurePosixPath(focus_path).parent
                    target_path = str(parent) if str(parent) != "." else "."
                else:
                    target_path = "."
            else:
                target_path = request.path_hint or "."
            query = str(request.query or "").strip()
            if not query:
                return None
            normalized_path = self.path_resolver.normalize_path_fragment(target_path) or "."
            if normalized_path != ".":
                resolved = self.path_resolver.resolve_path_hint(
                    normalized_path,
                    expect_dir=True,
                    action="search_files",
                    payload={"query": query},
                )
                if resolved.status != "resolved":
                    return None
                normalized_path = resolved.path
            self._promote_tool_profile(user_id, "workspace_read")
            reply = self.tool_catalog.runtime.search_files(query, normalized_path)
            return self._remember_browse_result(
                user_id,
                intent="browse_search",
                path=normalized_path,
                target_type="dir",
                reply=reply,
                query=query,
            )

        return None

    def _extract_replace_request(self, message: str) -> Optional[Tuple[str, str, str]]:
        extracted = extract_replace_request(message)
        if not extracted:
            return None
        raw_path, find_text, replace_text = extracted
        path = self.path_resolver.normalize_path_fragment(raw_path)
        path = self.path_resolver.resolve_existing_path_hint(
            path,
            expect_file=True,
            action="preview_edit_file",
            payload={
                "find_text": find_text,
                "replace_text": replace_text,
            },
        )
        if path and find_text:
            return path, find_text, replace_text
        return None

    def _extract_rewrite_request(self, message: str) -> Optional[Tuple[str, str]]:
        extracted = extract_rewrite_request(message)
        if not extracted:
            return None
        raw_path, content = extracted
        path = self.path_resolver.normalize_path_fragment(raw_path)
        path = self.path_resolver.resolve_existing_path_hint(
            path,
            expect_file=True,
            action="preview_write_file",
            payload={"content": content},
        )
        if path:
            return path, content
        return None

    def _extract_block_rewrite_request(self, message: str) -> Optional[Tuple[str, str, str]]:
        extracted = extract_block_rewrite_request(message)
        if not extracted:
            return None
        raw_path, target, instruction = extracted
        path = self.path_resolver.normalize_path_fragment(raw_path)
        path = self.path_resolver.resolve_existing_path_hint(
            path,
            expect_file=True,
            action="rewrite_block",
            payload={
                "target": target,
                "instruction": instruction,
            },
        )
        if path:
            return path, target, instruction
        return None

    def _extract_append_request(self, message: str) -> Optional[Tuple[str, str, str]]:
        extracted = extract_append_request(message)
        if not extracted:
            return None
        raw_path, content, position = extracted
        path = self.path_resolver.normalize_path_fragment(raw_path)
        path = self.path_resolver.resolve_existing_path_hint(
            path,
            expect_file=True,
            action="preview_append_file",
            payload={
                "content": content,
                "position": position,
            },
        )
        if path:
            return path, content, position
        return None

    def _extract_rename_paths(self, message: str) -> Optional[Tuple[str, str]]:
        extracted = extract_rename_paths(message)
        if extracted:
            raw_source, raw_target = extracted
            source = self.path_resolver.normalize_path_fragment(raw_source)
            target = self.path_resolver.normalize_path_fragment(raw_target)
            source = self.path_resolver.resolve_existing_path_hint(
                source,
                action="rename_path",
                payload={"target_path": target},
            )
            if not source or not target:
                return None

            target_path = PurePosixPath(target)
            if len(target_path.parts) == 1 and source not in {".", ""}:
                source_parent = PurePosixPath(source).parent
                if str(source_parent) != ".":
                    target = str(source_parent / target_path.name)
            return source, target
        return self._extract_followup_rename_target(message)

    def _extract_followup_rename_target(self, message: str) -> Optional[Tuple[str, str]]:
        target = extract_followup_rename_target(message)
        if not target:
            return None

        last_target = self.session_service.get_last_workspace_target(self.path_resolver.current_user_id)
        if str(last_target.get("type", "")).strip() != "file":
            return None

        source = str(last_target.get("path", "")).strip()
        if not source:
            return None

        target = self.path_resolver.normalize_path_fragment(target)
        if not target:
            return None

        target_path = PurePosixPath(target)
        if len(target_path.parts) == 1 and source not in {".", ""}:
            source_parent = PurePosixPath(source).parent
            if str(source_parent) != ".":
                target = str(source_parent / target_path.name)
        return source, target

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
