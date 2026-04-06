from __future__ import annotations

import difflib
import os
import re
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from src.services.agent.tool_catalog import ToolCatalog
from src.services.agent.tool_profiles import merge_tool_profile, normalize_tool_profile
from src.services.prompt_manager import PromptManager
from src.services.session_service import SessionService

if TYPE_CHECKING:
    from src.services.ai.llm_service import LLMService


class FastPathRouter:
    """处理不需要完整 agent loop 的确定性请求。"""

    TOOL_OVERVIEW_PATTERN = re.compile(
        r"(有哪些工具|有什么工具|能用什么工具|可以用什么工具|能做什么|会什么|能力有哪些)"
    )
    PROFILE_OVERVIEW_PATTERN = re.compile(
        r"(当前能力档位|当前工具档位|当前模式|当前会话模式|我现在是什么模式|我现在是什么档位|现在是什么模式|现在是什么档位)"
    )
    READ_FILE_PATTERNS = (
        re.compile(
            r"(?:帮我|麻烦|请)?(?:看看|看下|查看|读一下|读取|打开)\s*(?P<path>[A-Za-z0-9_./\\-]+)\s*(?:写的什么|写了什么|内容是什么|里有什么|里有哪些)?",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:读一下|读取|查看|看看|打开)\s*(?P<path>[^\s，。！？]+(?:\.[A-Za-z0-9_-]+|README(?:\.md)?))",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?P<path>[^\s，。！？]+(?:\.[A-Za-z0-9_-]+|README(?:\.md)?))\s*(?:里写的什么|写了什么|内容是什么)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?P<path>[^\s，。！？]+(?:\.[A-Za-z0-9_-]+|README(?:\.md)?))\s*里(?:有什么|有哪些)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?P<path>[^\s，。！？]+)\s*(?:写的什么|写了什么|内容是什么|里有什么|里有哪些)",
            re.IGNORECASE,
        ),
    )
    READ_FILE_LINE_PATTERNS = (
        re.compile(
            r"(?P<path>[^\s，。！？]+(?:\.[A-Za-z0-9_-]+|README(?:\.md)?))\s*(?P<position>最后一行|末行|最后1行|第一行|首行|第1行)",
            re.IGNORECASE,
        ),
    )
    LIST_DIR_PATTERNS = (
        re.compile(r"(?:看看|查看|列出)\s*(?P<path>[^\s，。！？]+)\s*目录"),
        re.compile(r"(?P<path>[^\s，。！？]+)\s*目录(?:里|下)?(?:有什么|有哪些)?"),
        re.compile(r"(?P<path>[^\s，。！？]+)\s*里(?:写的什么|有什么|有哪些)"),
    )
    SEARCH_FILE_PATTERNS = (
        re.compile(
            r"在\s*(?P<path>[^\s，。！？]+)\s*(?:里|中|目录里|目录下|下面)?(?:搜索|查找)\s*(?P<query>[^\n，。！？]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:搜索|查找)\s*(?P<query>[^\s，。！？]+)\s*(?:内容|文本|关键词)?(?:在\s*(?P<path>[^\s，。！？]+))?",
            re.IGNORECASE,
        ),
    )
    REPLACE_PATTERNS = (
        re.compile(
            r"把\s*(?P<path>.+?)\s*(?:里的|里面的|中的|内容里的)\s*[\"“'‘](?P<find>[\s\S]+?)[\"”'’]\s*(?:改成|替换成|替换为)\s*[\"“'‘](?P<replace>[\s\S]+?)[\"”'’]",
            re.IGNORECASE,
        ),
    )
    REWRITE_PATTERNS = (
        re.compile(
            r"把\s*(?P<path>.+?)\s*(?:内容)?(?:改成|改为|写成|重写成|覆盖成)\s*[\"“'‘](?P<content>[\s\S]+?)[\"”'’]",
            re.IGNORECASE,
        ),
    )
    APPEND_PATTERNS = (
        re.compile(
            r"(?:在|给)\s*(?P<path>.+?)\s*(?P<position>末尾|结尾|开头|前面|后面)\s*(?:追加|加上|补上|添加)\s*[\"“'‘](?P<content>[\s\S]+?)[\"”'’]",
            re.IGNORECASE,
        ),
    )
    REWRITE_BLOCK_PATTERNS = (
        re.compile(
            r"把\s*(?P<path>.+?)\s*(?P<target>第一段|开头|首段|标题)\s*(?:改得|改成|改为)?\s*(?P<instruction>更清楚|更清晰|更简洁|更简短|更短|更正式|更专业|更口语化|更自然|更有条理|改短一点|改正式一点|改清楚一点)",
            re.IGNORECASE,
        ),
    )
    RENAME_PATTERNS = (
        re.compile(
            r"把\s*(?P<source>.+?)\s*(?:重命名|改名|改文件名)\s*(?:为|成)\s*(?P<target>[^\s，。！？]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"把\s*(?P<source>.+?)\s*(?:移动到|挪到)\s*(?P<target>[^\s，。！？]+)",
            re.IGNORECASE,
        ),
    )
    FOLLOWUP_RENAME_PATTERN = re.compile(
        r"^(?:改为|改成|重命名为|命名为)\s*(?P<target>[^\s，。！？]+)$",
        re.IGNORECASE,
    )

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
        self._current_user_id = ""
        self._pending_resolution_reply: Optional[str] = None

    def try_handle(self, user_id: str, message: str) -> Optional[str]:
        message = (message or "").strip()
        if not message:
            return None

        self._current_user_id = user_id
        self._pending_resolution_reply = None
        self._upgrade_tool_profile(user_id, message)

        if self.TOOL_OVERVIEW_PATTERN.search(message):
            return self._handle_tool_overview(user_id, message)

        if self.PROFILE_OVERVIEW_PATTERN.search(message):
            return self._handle_profile_overview(user_id, message)

        block_rewrite_request = self._extract_block_rewrite_request(message)
        if self._pending_resolution_reply:
            return self._pending_resolution_reply
        if block_rewrite_request:
            path, target, instruction = block_rewrite_request
            self.session_service.set_last_workspace_target(user_id, path, "file")
            return self._handle_block_rewrite(user_id, path, target, instruction)

        replace_request = self._extract_replace_request(message)
        if self._pending_resolution_reply:
            return self._pending_resolution_reply
        if replace_request:
            path, find_text, replace_text = replace_request
            self.session_service.set_last_workspace_target(user_id, path, "file")
            return self.tool_catalog.runtime.preview_edit_file(
                path,
                find_text,
                replace_text,
                owner_user_id=user_id,
            )

        rewrite_request = self._extract_rewrite_request(message)
        if self._pending_resolution_reply:
            return self._pending_resolution_reply
        if rewrite_request:
            path, content = rewrite_request
            self.session_service.set_last_workspace_target(user_id, path, "file")
            return self.tool_catalog.runtime.preview_write_file(
                path,
                content,
                owner_user_id=user_id,
            )

        append_request = self._extract_append_request(message)
        if self._pending_resolution_reply:
            return self._pending_resolution_reply
        if append_request:
            path, content, position = append_request
            self.session_service.set_last_workspace_target(user_id, path, "file")
            return self.tool_catalog.runtime.preview_append_file(
                path,
                content,
                position=position,
                owner_user_id=user_id,
            )

        rename_paths = self._extract_rename_paths(message)
        if self._pending_resolution_reply:
            return self._pending_resolution_reply
        if rename_paths:
            source_path, target_path = rename_paths
            reply = self.tool_catalog.runtime.rename_path(source_path, target_path)
            if not reply.startswith(("未找到源路径", "路径不允许访问", "目标路径无效")):
                self.session_service.set_last_workspace_target(user_id, target_path, "file")
            return reply

        read_line_request = self._extract_read_file_line_request(message)
        if self._pending_resolution_reply:
            return self._pending_resolution_reply
        if read_line_request:
            path, position = read_line_request
            reply = self.tool_catalog.runtime.read_file_line(path, position)
            if not reply.startswith(("文件不存在", "目标不是文件", "路径不允许访问")):
                self.session_service.set_last_workspace_target(user_id, path, "file")
            return reply

        search_request = self._extract_search_request(message)
        if search_request:
            query, path = search_request
            return self.tool_catalog.runtime.search_files(query, path)

        file_path = self._extract_read_file_path(message)
        if self._pending_resolution_reply:
            return self._pending_resolution_reply
        if file_path:
            reply = self.tool_catalog.runtime.read_file(file_path)
            if not reply.startswith(("文件不存在", "目标不是文件", "路径不允许访问")):
                self.session_service.set_last_workspace_target(user_id, file_path, "file")
            return reply

        dir_path = self._extract_directory_path(message)
        if self._pending_resolution_reply:
            return self._pending_resolution_reply
        if dir_path:
            reply = self.tool_catalog.runtime.list_directory(dir_path)
            if not reply.startswith(("路径不存在", "目标不是目录", "路径不允许访问")):
                self.session_service.set_last_workspace_target(user_id, dir_path, "dir")
            return reply

        followup_reply = self._handle_followup_reference(user_id, message)
        if followup_reply:
            return followup_reply

        return None

    def try_handle_pending_resolution(self, user_id: str, message: str) -> Optional[str]:
        pending = self.session_service.get_pending_workspace_resolution(user_id)
        if not pending:
            return None

        choice = self._parse_resolution_choice(message)
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
        if tool_bundle.summary_lines:
            lines.append("")
            lines.append("这些工具当前分别能做：")
            lines.extend(tool_bundle.summary_lines)
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
        if tool_bundle.summary_lines:
            lines.append("")
            lines.append("当前能力边界：")
            lines.extend(tool_bundle.summary_lines)
        return "\n".join(lines)

    def _upgrade_tool_profile(self, user_id: str, message: str) -> None:
        inferred = self.tool_catalog.infer_tool_profile(message)
        current = self.session_service.get_tool_profile(user_id)
        merged = merge_tool_profile(current, inferred)
        if merged != normalize_tool_profile(current):
            self.session_service.set_tool_profile(user_id, merged)

    def _extract_read_file_path(self, message: str) -> Optional[str]:
        for pattern in self.READ_FILE_PATTERNS:
            match = pattern.search(message)
            if not match:
                continue
            normalized = self._normalize_path_fragment(match.group("path"))
            normalized = self._resolve_existing_path_hint(
                normalized,
                expect_file=True,
                action="read_file",
            )
            if normalized and self._looks_like_existing_file(normalized):
                return normalized
        return None

    def _extract_directory_path(self, message: str) -> Optional[str]:
        for pattern in self.LIST_DIR_PATTERNS:
            match = pattern.search(message)
            if not match:
                continue
            normalized = self._normalize_path_fragment(match.group("path"))
            normalized = self._resolve_existing_path_hint(
                normalized,
                expect_dir=True,
                action="list_directory",
            )
            if normalized and self._looks_like_existing_dir(normalized):
                return normalized
        return None

    def _extract_search_request(self, message: str) -> Optional[Tuple[str, str]]:
        for pattern in self.SEARCH_FILE_PATTERNS:
            match = pattern.search(message)
            if not match:
                continue
            query = (match.group("query") or "").strip()
            path = self._normalize_path_fragment(match.groupdict().get("path", "") or ".")
            query = query.strip("`'\"“”‘’")
            if not query:
                continue
            return query, path or "."
        return None

    def _extract_read_file_line_request(self, message: str) -> Optional[Tuple[str, str]]:
        for pattern in self.READ_FILE_LINE_PATTERNS:
            match = pattern.search(message)
            if not match:
                continue
            raw_position = match.group("position")
            position = "first" if raw_position in {"第一行", "首行", "第1行"} else "last"
            path = self._normalize_path_fragment(match.group("path"))
            path = self._resolve_existing_path_hint(
                path,
                expect_file=True,
                action="read_file_line",
                payload={"position": position},
            )
            if path:
                return path, position
        return None

    def _extract_replace_request(self, message: str) -> Optional[Tuple[str, str, str]]:
        for pattern in self.REPLACE_PATTERNS:
            match = pattern.search(message)
            if not match:
                continue
            find_text = match.group("find")
            replace_text = match.group("replace")
            path = self._normalize_path_fragment(match.group("path"))
            path = self._resolve_existing_path_hint(
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
        for pattern in self.REWRITE_PATTERNS:
            match = pattern.search(message)
            if not match:
                continue
            content = match.group("content")
            path = self._normalize_path_fragment(match.group("path"))
            path = self._resolve_existing_path_hint(
                path,
                expect_file=True,
                action="preview_write_file",
                payload={"content": content},
            )
            if path:
                return path, content
        return None

    def _extract_block_rewrite_request(self, message: str) -> Optional[Tuple[str, str, str]]:
        for pattern in self.REWRITE_BLOCK_PATTERNS:
            match = pattern.search(message)
            if not match:
                continue
            target = match.group("target").strip()
            instruction = match.group("instruction").strip()
            path = self._normalize_path_fragment(match.group("path"))
            path = self._resolve_existing_path_hint(
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
        for pattern in self.APPEND_PATTERNS:
            match = pattern.search(message)
            if not match:
                continue
            content = match.group("content")
            raw_position = match.group("position")
            position = "start" if raw_position in {"开头", "前面"} else "end"
            path = self._normalize_path_fragment(match.group("path"))
            path = self._resolve_existing_path_hint(
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

    def _handle_block_rewrite(self, user_id: str, path: str, target: str, instruction: str) -> str:
        if not self.llm_service:
            return "当前未接入段落改写能力，请改用精确替换或整文件预览修改。"

        target_file = self.tool_catalog.runtime._resolve_path(path)
        if not target_file.exists():
            return f"文件不存在: {path}"
        if not target_file.is_file():
            return f"目标不是文件: {path}"

        text = target_file.read_text(encoding="utf-8", errors="ignore")
        span = self._locate_rewrite_block(text, target)
        if not span:
            return f"未找到可改写的{target}"

        start_index, end_index = span
        original_block = text[start_index:end_index]
        rewritten_block = self._rewrite_block_with_llm(path, target, instruction, original_block)
        if not rewritten_block or rewritten_block == original_block:
            return "未生成有效改写结果，请改用更明确的修改指令。"

        return self.tool_catalog.runtime.preview_replace_span(
            path,
            start_index,
            end_index,
            rewritten_block,
            owner_user_id=user_id,
        )

    def _locate_rewrite_block(self, text: str, target: str) -> Optional[Tuple[int, int]]:
        if target == "标题":
            return self._find_first_heading_span(text)
        return self._find_first_paragraph_span(text)

    def _find_first_heading_span(self, text: str) -> Optional[Tuple[int, int]]:
        match = re.search(r"^#+[ \t].+$", text, re.MULTILINE)
        if match:
            return match.start(), match.end()

        first_line_match = re.search(r"^[^\n]+", text)
        if not first_line_match:
            return None
        return first_line_match.start(), first_line_match.end()

    def _find_first_paragraph_span(self, text: str) -> Optional[Tuple[int, int]]:
        pattern = re.compile(r"(?:^|\n\n)(?P<block>(?!#)[^\n][\s\S]*?)(?=\n\n|$)")
        for match in pattern.finditer(text):
            block = match.group("block").strip()
            if not block:
                continue
            return match.start("block"), match.end("block")
        return None

    def _rewrite_block_with_llm(self, path: str, target: str, instruction: str, original_block: str) -> str:
        messages = [
            {
                "role": "system",
                "content": self.prompt_manager.build_fast_path_rewrite_prompt(),
            },
            {
                "role": "user",
                "content": (
                    f"文件路径：{path}\n"
                    f"目标块：{target}\n"
                    f"改写要求：{instruction}\n\n"
                    "请只改写下面这个已经定位好的块，不要扩写其它部分。\n\n"
                    f"原文如下：\n{original_block}"
                ),
            },
        ]
        return self.llm_service.chat(messages).strip()

    def _extract_rename_paths(self, message: str) -> Optional[Tuple[str, str]]:
        for pattern in self.RENAME_PATTERNS:
            match = pattern.search(message)
            if not match:
                continue

            source = self._normalize_path_fragment(match.group("source"))
            target = self._normalize_path_fragment(match.group("target"))
            source = self._resolve_existing_path_hint(
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
        match = self.FOLLOWUP_RENAME_PATTERN.search(message.strip())
        if not match:
            return None

        last_target = self.session_service.get_last_workspace_target(self._current_user_id)
        if str(last_target.get("type", "")).strip() != "file":
            return None

        source = str(last_target.get("path", "")).strip()
        if not source:
            return None

        target = self._normalize_path_fragment(match.group("target"))
        if not target:
            return None

        target_path = PurePosixPath(target)
        if len(target_path.parts) == 1 and source not in {".", ""}:
            source_parent = PurePosixPath(source).parent
            if str(source_parent) != ".":
                target = str(source_parent / target_path.name)
        return source, target

    def _normalize_path_fragment(self, fragment: str) -> str:
        normalized = (fragment or "").strip()
        if not normalized:
            return ""

        normalized = normalized.strip("`'\"“”‘’")
        normalized = normalized.replace("下的", "/")
        normalized = normalized.replace("下面的", "/")
        normalized = normalized.replace("目录里的", "/")
        normalized = normalized.replace("目录下的", "/")
        normalized = normalized.replace("目录", "")
        normalized = normalized.replace("文件", "")
        normalized = normalized.replace("\\", "/")
        normalized = re.sub(r"\s+", "", normalized)
        normalized = re.sub(r"/{2,}", "/", normalized)
        return normalized.strip("/")

    def _resolve_existing_path_hint(
        self,
        path: str,
        *,
        expect_file: bool = False,
        expect_dir: bool = False,
        action: str = "",
        payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        if not path:
            return path

        candidate, error = self.tool_catalog.runtime._resolve_path_or_error(path)
        if not error and candidate and candidate.exists():
            if expect_file and candidate.is_file():
                return path
            if expect_dir and candidate.is_dir():
                return path
            if not expect_file and not expect_dir:
                return path

        lowered = path.lower()
        normalized_key = self._normalize_lookup_key(path)
        if lowered == "readme":
            return "README.md"

        last_target = self.session_service.get_last_workspace_target(self._current_user_id)
        last_path = str(last_target.get("path", "")).strip()
        last_type = str(last_target.get("type", "")).strip()
        if last_path and "/" not in path:
            base_dir = PurePosixPath(last_path) if last_type == "dir" else PurePosixPath(last_path).parent
            candidate_path = str(base_dir / path) if str(base_dir) != "." else path
            candidate, candidate_error = self.tool_catalog.runtime._resolve_path_or_error(candidate_path)
            if not candidate_error and candidate and candidate.exists():
                if expect_file and candidate.is_file():
                    return candidate_path
                if expect_dir and candidate.is_dir():
                    return candidate_path
                if not expect_file and not expect_dir:
                    return candidate_path

        if "/" in path:
            self._maybe_store_path_resolution(
                path,
                expect_file=expect_file,
                expect_dir=expect_dir,
                action=action,
                payload=payload,
            )
            return None if self._pending_resolution_reply else path

        exact_file_matches: List[str] = []
        for file_path in self.tool_catalog.runtime._iter_files(self.tool_catalog.runtime.workspace_root):
            file_name = file_path.name.lower()
            if file_name != lowered and self._normalize_lookup_key(file_name) != normalized_key:
                continue
            exact_file_matches.append(self.tool_catalog.runtime._to_relative(file_path))
            if len(exact_file_matches) > 1:
                break
        if len(exact_file_matches) == 1:
            return exact_file_matches[0]

        if expect_dir:
            exact_dir_matches: List[str] = []
            for current_root, dirnames, _ in os.walk(self.tool_catalog.runtime.workspace_root):
                dirnames[:] = [name for name in dirnames if name not in self.tool_catalog.runtime.SKIP_DIRS]
                for dirname in dirnames:
                    if dirname.lower() != lowered:
                        continue
                    dir_path = PurePosixPath(os.path.relpath(os.path.join(current_root, dirname), self.tool_catalog.runtime.workspace_root))
                    exact_dir_matches.append(str(dir_path))
                    if len(exact_dir_matches) > 1:
                        break
                if len(exact_dir_matches) > 1:
                    break
            if len(exact_dir_matches) == 1:
                return exact_dir_matches[0]

        self._maybe_store_path_resolution(
            path,
            expect_file=expect_file,
            expect_dir=expect_dir,
            action=action,
            payload=payload,
        )
        return None if self._pending_resolution_reply else path

    def _normalize_lookup_key(self, value: str) -> str:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", value.lower())

    def _looks_like_existing_file(self, path: str) -> bool:
        candidate, error = self.tool_catalog.runtime._resolve_path_or_error(path)
        return not error and bool(candidate and candidate.exists() and candidate.is_file())

    def _looks_like_existing_dir(self, path: str) -> bool:
        candidate, error = self.tool_catalog.runtime._resolve_path_or_error(path)
        return not error and bool(candidate and candidate.exists() and candidate.is_dir())

    def _handle_followup_reference(self, user_id: str, message: str) -> Optional[str]:
        last_target = self.session_service.get_last_workspace_target(user_id)
        path = str(last_target.get("path", "")).strip()
        target_type = str(last_target.get("type", "")).strip()
        if not path:
            return None

        if target_type == "file" and any(token in message for token in ("它", "这个文件")):
            if any(keyword in message for keyword in ("内容", "写的什么", "写了什么", "有什么", "有哪些")):
                return self.tool_catalog.runtime.read_file(path)
            if any(keyword in message for keyword in ("最后一行", "末行", "第一行", "首行")):
                position = "first" if any(keyword in message for keyword in ("第一行", "首行")) else "last"
                return self.tool_catalog.runtime.read_file_line(path, position)

        return None

    def _maybe_store_path_resolution(
        self,
        original_input: str,
        *,
        expect_file: bool,
        expect_dir: bool,
        action: str,
        payload: Optional[Dict[str, Any]],
    ) -> None:
        candidates = self._find_path_candidates(
            original_input,
            expect_file=expect_file,
            expect_dir=expect_dir,
        )
        if not candidates:
            return

        self.session_service.set_pending_workspace_resolution(
            self._current_user_id,
            action=action,
            original_input=original_input,
            candidates=candidates,
            payload=payload or {},
        )
        self._pending_resolution_reply = self._format_resolution_prompt(
            original_input=original_input,
            candidates=candidates,
            expect_dir=expect_dir,
        )

    def _find_path_candidates(
        self,
        original_input: str,
        *,
        expect_file: bool,
        expect_dir: bool,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        query = self._normalize_path_fragment(original_input)
        query_name = PurePosixPath(query).name
        query_key = self._normalize_lookup_key(query_name)
        if not query_key:
            return []

        candidates: List[Dict[str, Any]] = []

        if not expect_dir:
            for file_path in self.tool_catalog.runtime._iter_files(self.tool_catalog.runtime.workspace_root):
                relative_path = self.tool_catalog.runtime._to_relative(file_path)
                score = self._score_path_candidate(query_key, file_path.name, relative_path)
                if score < 0.62:
                    continue
                candidates.append({"path": relative_path, "type": "file", "score": score})

        if expect_dir or not expect_file:
            for current_root, dirnames, _ in os.walk(self.tool_catalog.runtime.workspace_root):
                dirnames[:] = [name for name in dirnames if name not in self.tool_catalog.runtime.SKIP_DIRS]
                for dirname in dirnames:
                    relative_path = str(
                        PurePosixPath(
                            os.path.relpath(
                                os.path.join(current_root, dirname),
                                self.tool_catalog.runtime.workspace_root,
                            )
                        )
                    )
                    score = self._score_path_candidate(query_key, dirname, relative_path)
                    if score < 0.62:
                        continue
                    candidates.append({"path": relative_path, "type": "dir", "score": score})

        unique_by_path: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            existing = unique_by_path.get(candidate["path"])
            if existing is None or candidate["score"] > existing["score"]:
                unique_by_path[candidate["path"]] = candidate

        ordered = sorted(
            unique_by_path.values(),
            key=lambda item: (-float(item["score"]), len(str(item["path"]))),
        )
        return ordered[:limit]

    def _score_path_candidate(self, query_key: str, name: str, relative_path: str) -> float:
        name_key = self._normalize_lookup_key(name)
        path_key = self._normalize_lookup_key(relative_path)
        score = max(
            difflib.SequenceMatcher(None, query_key, name_key).ratio(),
            difflib.SequenceMatcher(None, query_key, path_key).ratio(),
        )
        if query_key == name_key:
            return 1.0
        if name_key.startswith(query_key):
            score = max(score, 0.9)
        if query_key in name_key:
            score = max(score, 0.85)
        return score

    def _format_resolution_prompt(
        self,
        *,
        original_input: str,
        candidates: List[Dict[str, Any]],
        expect_dir: bool,
    ) -> str:
        target_label = "目录" if expect_dir else "文件"
        lines = [f"没找到精确匹配的{target_label}：{original_input}", "", "你是不是指："]
        for index, candidate in enumerate(candidates, start=1):
            suffix = " [目录]" if candidate.get("type") == "dir" else ""
            lines.append(f"{index}. {candidate.get('path', '')}{suffix}")
        lines.append("")
        lines.append("回复“是”默认采用第 1 个，也可以回复序号选择，回复“不是”取消。")
        return "\n".join(lines)

    def _parse_resolution_choice(self, message: str) -> Optional[int]:
        normalized = (message or "").strip().lower()
        if normalized in {"是", "对", "嗯", "嗯嗯", "好", "好的", "确认", "同意", "就这个", "就是这个"}:
            return 1
        if normalized in {"不是", "不对", "取消", "都不是", "算了"}:
            return 0

        match = re.fullmatch(r"(?:第)?([1-9])(?:个)?", normalized)
        if match:
            return int(match.group(1))
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
            reply = self.tool_catalog.runtime.read_file(path)
        elif action == "read_file_line":
            reply = self.tool_catalog.runtime.read_file_line(path, str(payload.get("position", "last")))
        elif action == "list_directory":
            reply = self.tool_catalog.runtime.list_directory(path)
        elif action == "preview_edit_file":
            reply = self.tool_catalog.runtime.preview_edit_file(
                path,
                str(payload.get("find_text", "")),
                str(payload.get("replace_text", "")),
                owner_user_id=user_id,
            )
        elif action == "preview_write_file":
            reply = self.tool_catalog.runtime.preview_write_file(
                path,
                str(payload.get("content", "")),
                owner_user_id=user_id,
            )
        elif action == "preview_append_file":
            reply = self.tool_catalog.runtime.preview_append_file(
                path,
                str(payload.get("content", "")),
                position=str(payload.get("position", "end")),
                owner_user_id=user_id,
            )
        elif action == "rewrite_block":
            reply = self._handle_block_rewrite(
                user_id,
                path,
                str(payload.get("target", "第一段")),
                str(payload.get("instruction", "改短一点")),
            )
        elif action == "rename_path":
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
