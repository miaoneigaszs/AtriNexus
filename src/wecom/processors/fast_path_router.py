from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Optional, Tuple, TYPE_CHECKING

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
            r"(?:读一下|读取|查看|看看|打开)\s*(?P<path>[^\s，。！？]+(?:\.[A-Za-z0-9_-]+|README(?:\.md)?))",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?P<path>[^\s，。！？]+(?:\.[A-Za-z0-9_-]+|README(?:\.md)?))\s*(?:里写的什么|写了什么|内容是什么)",
            re.IGNORECASE,
        ),
    )
    LIST_DIR_PATTERNS = (
        re.compile(r"(?:看看|查看|列出)\s*(?P<path>[^\s，。！？]+)\s*目录"),
        re.compile(r"(?P<path>[^\s，。！？]+)\s*目录(?:里|下)?(?:有什么|有哪些)?"),
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

    def try_handle(self, user_id: str, message: str) -> Optional[str]:
        message = (message or "").strip()
        if not message:
            return None

        self._upgrade_tool_profile(user_id, message)

        if self.TOOL_OVERVIEW_PATTERN.search(message):
            return self._handle_tool_overview(user_id, message)

        if self.PROFILE_OVERVIEW_PATTERN.search(message):
            return self._handle_profile_overview(user_id, message)

        block_rewrite_request = self._extract_block_rewrite_request(message)
        if block_rewrite_request:
            path, target, instruction = block_rewrite_request
            return self._handle_block_rewrite(user_id, path, target, instruction)

        replace_request = self._extract_replace_request(message)
        if replace_request:
            path, find_text, replace_text = replace_request
            return self.tool_catalog.runtime.preview_edit_file(
                path,
                find_text,
                replace_text,
                owner_user_id=user_id,
            )

        rewrite_request = self._extract_rewrite_request(message)
        if rewrite_request:
            path, content = rewrite_request
            return self.tool_catalog.runtime.preview_write_file(
                path,
                content,
                owner_user_id=user_id,
            )

        append_request = self._extract_append_request(message)
        if append_request:
            path, content, position = append_request
            return self.tool_catalog.runtime.preview_append_file(
                path,
                content,
                position=position,
                owner_user_id=user_id,
            )

        rename_paths = self._extract_rename_paths(message)
        if rename_paths:
            source_path, target_path = rename_paths
            return self.tool_catalog.runtime.rename_path(source_path, target_path)

        search_request = self._extract_search_request(message)
        if search_request:
            query, path = search_request
            return self.tool_catalog.runtime.search_files(query, path)

        file_path = self._extract_read_file_path(message)
        if file_path:
            return self.tool_catalog.runtime.read_file(file_path)

        dir_path = self._extract_directory_path(message)
        if dir_path:
            return self.tool_catalog.runtime.list_directory(dir_path)

        return None

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
        avatar_name = self.session_service.get_current_avatar(user_id)
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
            f"当前人设：{avatar_name}",
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
            if normalized:
                return normalized
        return None

    def _extract_directory_path(self, message: str) -> Optional[str]:
        for pattern in self.LIST_DIR_PATTERNS:
            match = pattern.search(message)
            if not match:
                continue
            normalized = self._normalize_path_fragment(match.group("path"))
            if normalized:
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

    def _extract_replace_request(self, message: str) -> Optional[Tuple[str, str, str]]:
        for pattern in self.REPLACE_PATTERNS:
            match = pattern.search(message)
            if not match:
                continue
            path = self._normalize_path_fragment(match.group("path"))
            find_text = match.group("find")
            replace_text = match.group("replace")
            if path and find_text:
                return path, find_text, replace_text
        return None

    def _extract_rewrite_request(self, message: str) -> Optional[Tuple[str, str]]:
        for pattern in self.REWRITE_PATTERNS:
            match = pattern.search(message)
            if not match:
                continue
            path = self._normalize_path_fragment(match.group("path"))
            content = match.group("content")
            if path:
                return path, content
        return None

    def _extract_block_rewrite_request(self, message: str) -> Optional[Tuple[str, str, str]]:
        for pattern in self.REWRITE_BLOCK_PATTERNS:
            match = pattern.search(message)
            if not match:
                continue
            path = self._normalize_path_fragment(match.group("path"))
            target = match.group("target").strip()
            instruction = match.group("instruction").strip()
            if path:
                return path, target, instruction
        return None

    def _extract_append_request(self, message: str) -> Optional[Tuple[str, str, str]]:
        for pattern in self.APPEND_PATTERNS:
            match = pattern.search(message)
            if not match:
                continue
            path = self._normalize_path_fragment(match.group("path"))
            content = match.group("content")
            raw_position = match.group("position")
            position = "start" if raw_position in {"开头", "前面"} else "end"
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
            start = match.start("block")
            end = match.end("block")
            return start, end
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
            if not source or not target:
                return None

            target_path = PurePosixPath(target)
            if len(target_path.parts) == 1 and source not in {".", ""}:
                source_parent = PurePosixPath(source).parent
                if str(source_parent) != ".":
                    target = str(source_parent / target_path.name)
            return source, target
        return None

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
