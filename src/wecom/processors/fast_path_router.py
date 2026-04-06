from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from src.services.agent.tool_catalog import ToolCatalog
from src.services.agent.tool_profiles import merge_tool_profile, normalize_tool_profile
from src.services.prompt_manager import PromptManager
from src.services.session_service import SessionService
from src.wecom.processors.fast_path_intents import (
    extract_append_request,
    extract_block_rewrite_request,
    extract_directory_path,
    extract_followup_rename_target,
    extract_read_file_line_request,
    extract_read_file_path,
    extract_rename_paths,
    extract_replace_request,
    extract_rewrite_request,
    extract_search_request,
    is_profile_overview,
    is_tool_overview,
)
from src.wecom.processors.fast_path_rewrite import FastPathRewriteHelper
from src.wecom.processors.fast_path_resolution import WorkspacePathResolver

if TYPE_CHECKING:
    from src.services.ai.llm_service import LLMService


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

        if is_tool_overview(normalized_message):
            return self._handle_tool_overview(user_id, normalized_message)

        if is_profile_overview(normalized_message):
            return self._handle_profile_overview(user_id, normalized_message)

        block_rewrite_request = self._extract_block_rewrite_request(normalized_message)
        pending_reply = self.path_resolver.take_pending_reply()
        if pending_reply:
            return pending_reply
        if block_rewrite_request:
            path, target, instruction = block_rewrite_request
            self._promote_tool_profile(user_id, "workspace_edit")
            self.session_service.set_last_workspace_target(user_id, path, "file")
            return self.rewrite_helper.handle_block_rewrite(user_id, path, target, instruction)

        replace_request = self._extract_replace_request(normalized_message)
        pending_reply = self.path_resolver.take_pending_reply()
        if pending_reply:
            return pending_reply
        if replace_request:
            path, find_text, replace_text = replace_request
            self._promote_tool_profile(user_id, "workspace_edit")
            self.session_service.set_last_workspace_target(user_id, path, "file")
            return self.tool_catalog.runtime.preview_edit_file(
                path,
                find_text,
                replace_text,
                owner_user_id=user_id,
            )

        rewrite_request = self._extract_rewrite_request(normalized_message)
        pending_reply = self.path_resolver.take_pending_reply()
        if pending_reply:
            return pending_reply
        if rewrite_request:
            path, content = rewrite_request
            self._promote_tool_profile(user_id, "workspace_edit")
            self.session_service.set_last_workspace_target(user_id, path, "file")
            return self.tool_catalog.runtime.preview_write_file(
                path,
                content,
                owner_user_id=user_id,
            )

        append_request = self._extract_append_request(normalized_message)
        pending_reply = self.path_resolver.take_pending_reply()
        if pending_reply:
            return pending_reply
        if append_request:
            path, content, position = append_request
            self._promote_tool_profile(user_id, "workspace_edit")
            self.session_service.set_last_workspace_target(user_id, path, "file")
            return self.tool_catalog.runtime.preview_append_file(
                path,
                content,
                position=position,
                owner_user_id=user_id,
            )

        rename_paths = self._extract_rename_paths(normalized_message)
        pending_reply = self.path_resolver.take_pending_reply()
        if pending_reply:
            return pending_reply
        if rename_paths:
            source_path, target_path = rename_paths
            self._promote_tool_profile(user_id, "workspace_edit")
            reply = self.tool_catalog.runtime.rename_path(source_path, target_path)
            if not reply.startswith(("未找到源路径", "路径不允许访问", "目标路径无效")):
                self.session_service.set_last_workspace_target(user_id, target_path, "file")
            return reply

        read_line_request = self._extract_read_file_line_request(normalized_message)
        pending_reply = self.path_resolver.take_pending_reply()
        if pending_reply:
            return pending_reply
        if read_line_request:
            path, position = read_line_request
            self._promote_tool_profile(user_id, "workspace_read")
            reply = self.tool_catalog.runtime.read_file_line(path, position)
            if not reply.startswith(("文件不存在", "目标不是文件", "路径不允许访问")):
                self.session_service.set_last_workspace_target(user_id, path, "file")
            return reply

        search_request = self._extract_search_request(normalized_message)
        if search_request:
            query, path = search_request
            self._promote_tool_profile(user_id, "workspace_read")
            return self.tool_catalog.runtime.search_files(query, path)

        file_path = self._extract_read_file_path(normalized_message)
        pending_reply = self.path_resolver.take_pending_reply()
        if pending_reply:
            return pending_reply
        if file_path:
            self._promote_tool_profile(user_id, "workspace_read")
            reply = self.tool_catalog.runtime.read_file(file_path)
            if not reply.startswith(("文件不存在", "目标不是文件", "路径不允许访问")):
                self.session_service.set_last_workspace_target(user_id, file_path, "file")
            return reply

        dir_path = self._extract_directory_path(normalized_message)
        pending_reply = self.path_resolver.take_pending_reply()
        if pending_reply:
            return pending_reply
        if dir_path:
            self._promote_tool_profile(user_id, "workspace_read")
            reply = self.tool_catalog.runtime.list_directory(dir_path)
            if not reply.startswith(("路径不存在", "目标不是目录", "路径不允许访问")):
                self.session_service.set_last_workspace_target(user_id, dir_path, "dir")
            return reply

        followup_reply = self._handle_followup_reference(user_id, normalized_message)
        if followup_reply:
            return followup_reply

        return None

    def try_handle_pending_resolution(self, user_id: str, message: str) -> Optional[str]:
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

    def _extract_read_file_path(self, message: str) -> Optional[str]:
        raw = extract_read_file_path(message)
        if not raw:
            return None
        normalized = self.path_resolver.normalize_path_fragment(raw)
        normalized = self.path_resolver.resolve_existing_path_hint(
            normalized,
            expect_file=True,
            action="read_file",
        )
        if normalized and self.path_resolver.looks_like_existing_file(normalized):
            return normalized
        return None

    def _extract_directory_path(self, message: str) -> Optional[str]:
        raw = extract_directory_path(message)
        if not raw:
            return None
        normalized = self.path_resolver.normalize_path_fragment(raw)
        normalized = self.path_resolver.resolve_existing_path_hint(
            normalized,
            expect_dir=True,
            action="list_directory",
        )
        if normalized and self.path_resolver.looks_like_existing_dir(normalized):
            return normalized
        return None

    def _extract_search_request(self, message: str) -> Optional[Tuple[str, str]]:
        extracted = extract_search_request(message)
        if not extracted:
            return None
        query, path = extracted
        normalized_path = self.path_resolver.normalize_path_fragment(path or ".")
        return query, normalized_path or "."

    def _extract_read_file_line_request(self, message: str) -> Optional[Tuple[str, str]]:
        extracted = extract_read_file_line_request(message)
        if not extracted:
            return None
        raw_path, position = extracted
        path = self.path_resolver.normalize_path_fragment(raw_path)
        path = self.path_resolver.resolve_existing_path_hint(
            path,
            expect_file=True,
            action="read_file_line",
            payload={"position": position},
        )
        if path:
            return path, position
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

    def _handle_followup_reference(self, user_id: str, message: str) -> Optional[str]:
        last_target = self.session_service.get_last_workspace_target(user_id)
        path = str(last_target.get("path", "")).strip()
        target_type = str(last_target.get("type", "")).strip()
        if not path:
            return None

        if target_type == "file" and any(token in message for token in ("它", "这个文件")):
            if any(keyword in message for keyword in ("内容", "写的什么", "写了什么", "有什么", "有哪些")):
                self._promote_tool_profile(user_id, "workspace_read")
                return self.tool_catalog.runtime.read_file(path)
            if any(keyword in message for keyword in ("最后一行", "末行", "第一行", "首行")):
                position = "first" if any(keyword in message for keyword in ("第一行", "首行")) else "last"
                self._promote_tool_profile(user_id, "workspace_read")
                return self.tool_catalog.runtime.read_file_line(path, position)

        return None

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
            remembered_type = "dir" if target_type == "dir" else "file"
            self.session_service.set_last_workspace_target(user_id, path, remembered_type)
        return reply
