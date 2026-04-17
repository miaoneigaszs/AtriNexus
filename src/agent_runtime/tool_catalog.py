"""Agent 可用工具目录（框架中立）。

每个工具登记为 `RegisteredTool`：一个 `ToolSpec`（送给模型的 JSONSchema 描述）
+ 一个异步 handler（接收已解析的 args dict，返回字符串结果）。

Phase 4 替代 LangChain 后，本文件不再依赖任何外部 agent 框架。Tool 元数据
直接以 OpenAI tool calling 协议的 JSONSchema 表达，避免任何中间层翻译。

Profile 系统不变：会话 profile 决定哪些 section 暴露；fast-path 与
before_tool_call hook 负责更细粒度的拦截 / 修复。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from src.ai.types import ToolSpec
from src.knowledge.kb_tools import build_kb_list_assets_response, build_kb_search_response
from src.agent_runtime.runtime import WorkspaceRuntime
from src.agent_runtime.tool_profiles import (
    normalize_tool_profile,
    should_enable_command,
    should_enable_web,
    should_enable_workspace_edit,
    should_enable_workspace_read,
)
from src.workspace.search_tool import SearchTool
from src.platform_core.time_tool import TimeTool


TOOL_OVERVIEW_HINTS = (
    "有哪些工具",
    "有什么工具",
    "能用什么工具",
    "可以用什么工具",
    "能做什么",
    "会什么",
    "能力有哪些",
)


# ── 工具登记结构 ────────────────────────────────────────────────────────


ToolHandler = Callable[[Dict[str, Any]], Awaitable[str]]


@dataclass(frozen=True)
class RegisteredTool:
    """供 agent loop 直接消费的工具单元。"""

    spec: ToolSpec
    handler: ToolHandler

    @property
    def name(self) -> str:
        return self.spec.name


@dataclass(frozen=True)
class ToolBundle:
    tools: List[RegisteredTool]
    profiles: List[str]
    compact_summary_lines: List[str]
    detailed_summary_lines: List[str]

    @property
    def summary(self) -> str:
        if not self.compact_summary_lines:
            return "- 当前无需额外工具。"
        return "\n".join(self.compact_summary_lines)

    @property
    def detailed_summary(self) -> str:
        if not self.detailed_summary_lines:
            return "- 当前无需额外工具。"
        return "\n".join(self.detailed_summary_lines)


@dataclass(frozen=True)
class ToolSectionDefinition:
    id: str
    label: str
    profile_tag: str
    compact_line: str
    detailed_lines: List[str]
    enabled_when: Callable[["ToolCatalog", str], bool]
    build_tools: Callable[["ToolCatalog", str], List[RegisteredTool]]


# ── JSONSchema 工厂 ─────────────────────────────────────────────────────


def _string_field(description: str, default: Optional[str] = None) -> Dict[str, Any]:
    field: Dict[str, Any] = {"type": "string", "description": description}
    if default is not None:
        field["default"] = default
    return field


def _integer_field(description: str, default: Optional[int] = None) -> Dict[str, Any]:
    field: Dict[str, Any] = {"type": "integer", "description": description}
    if default is not None:
        field["default"] = default
    return field


def _object_schema(properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# ── 同步 → 异步包装 ──────────────────────────────────────────────────────


def _sync_handler(fn: Callable[..., str]) -> ToolHandler:
    """把同步函数包成 async handler，并在线程池里跑（避免阻塞事件循环）。"""

    async def wrapper(args: Dict[str, Any]) -> str:
        return await asyncio.to_thread(fn, **args)

    return wrapper


# ── ToolCatalog ─────────────────────────────────────────────────────────


class ToolCatalog:
    """声明式工具目录。

    根据稳定的 session profile 决定当前能力边界。Profile 先稳定，再由
    before_tool_call hook 与 fast-path 做修正与拦截。
    """

    def __init__(
        self,
        workspace_root: str,
        search_api_key: Optional[str] = None,
        rag_service: Optional[object] = None,
    ) -> None:
        self.runtime = WorkspaceRuntime(workspace_root)
        self.search_api_key = search_api_key
        self.rag_service = rag_service
        self.time_tool = TimeTool()

    def build_tool_bundle(
        self,
        user_id: str,
        message: str,
        allow_tools: bool = True,
        tool_profile: Optional[str] = None,
    ) -> ToolBundle:
        if not allow_tools:
            return ToolBundle([], [], [], [])

        profile = normalize_tool_profile(tool_profile)
        tools: List[RegisteredTool] = []
        profiles: List[str] = ["core", profile]
        compact_summary_lines: List[str] = []
        detailed_summary_lines: List[str] = []

        for section in self._section_definitions():
            if not section.enabled_when(self, profile):
                continue

            section_tools = section.build_tools(self, user_id)
            if not section_tools:
                continue

            tools.extend(section_tools)
            profiles.append(section.profile_tag)
            compact_summary_lines.append(f"[{section.label}] {section.compact_line}")

            if detailed_summary_lines:
                detailed_summary_lines.append("")
            detailed_summary_lines.append(f"[{section.label}]")
            detailed_summary_lines.extend(section.detailed_lines)

        return ToolBundle(
            tools=tools,
            profiles=self._dedupe_profiles(profiles),
            compact_summary_lines=compact_summary_lines,
            detailed_summary_lines=detailed_summary_lines,
        )

    def looks_like_tool_overview(self, message: str) -> bool:
        normalized = (message or "").strip()
        return any(hint in normalized for hint in TOOL_OVERVIEW_HINTS)

    def format_tool_overview(self, tool_bundle: "ToolBundle") -> str:
        tool_names = [t.name for t in tool_bundle.tools]
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

    def _section_definitions(self) -> List[ToolSectionDefinition]:
        return [
            ToolSectionDefinition(
                id="core",
                label="基础",
                profile_tag="core",
                compact_line="读当前本地时间。",
                detailed_lines=["- get_current_time: 读当前本地时间。"],
                enabled_when=lambda _self, _profile: True,
                build_tools=lambda self, user_id: self._build_core_tools(user_id),
            ),
            ToolSectionDefinition(
                id="workspace-read",
                label="文件读取",
                profile_tag="workspace-read",
                compact_line="列目录、读文件、搜内容。",
                detailed_lines=[
                    "- list_directory: 列目录。",
                    "- read_file: 读文件。",
                    "- search_files: 按关键词搜文件内容。",
                ],
                enabled_when=lambda _self, profile: should_enable_workspace_read(profile),
                build_tools=lambda self, user_id: self._build_workspace_read_tools(user_id),
            ),
            ToolSectionDefinition(
                id="workspace-edit-preview",
                label="文件修改",
                profile_tag="workspace-edit-preview",
                compact_line="预览重写、精确替换、头尾追加。",
                detailed_lines=[
                    "- preview_write_file: 预览整文件重写。",
                    "- preview_edit_file: 预览精确替换。",
                    "- preview_append_file: 预览头尾追加。",
                ],
                enabled_when=lambda _self, profile: should_enable_workspace_edit(profile),
                build_tools=lambda self, user_id: self._build_workspace_edit_tools(user_id),
            ),
            ToolSectionDefinition(
                id="workspace-rename",
                label="文件整理",
                profile_tag="workspace-rename",
                compact_line="重命名或移动文件/目录。",
                detailed_lines=["- rename_path: 重命名或移动文件/目录。"],
                enabled_when=lambda _self, profile: should_enable_workspace_edit(profile),
                build_tools=lambda self, user_id: self._build_workspace_rename_tools(user_id),
            ),
            ToolSectionDefinition(
                id="command",
                label="运行环境",
                profile_tag="command",
                compact_line="执行命令；高风险命令先确认。",
                detailed_lines=["- run_command: 跑命令；复杂或高风险命令先确认。"],
                enabled_when=lambda _self, profile: should_enable_command(profile),
                build_tools=lambda self, user_id: self._build_command_tools(user_id),
            ),
            ToolSectionDefinition(
                id="kb",
                label="知识库",
                profile_tag="kb",
                compact_line="看资产目录、查知识库内容。",
                detailed_lines=[
                    "- kb_list_assets: 看知识库里有哪些文档和分类。",
                    "- kb_search: 查知识库内容，可限定文档或分类。",
                ],
                enabled_when=lambda self, _profile: self.rag_service is not None,
                build_tools=lambda self, user_id: self._build_kb_tools(user_id),
            ),
            ToolSectionDefinition(
                id="web",
                label="联网",
                profile_tag="web",
                compact_line="搜索互联网最新信息。",
                detailed_lines=["- web_search: 搜索互联网最新信息。"],
                enabled_when=lambda self, profile: bool(self.search_api_key and should_enable_web(profile)),
                build_tools=lambda self, user_id: self._build_web_tools(user_id),
            ),
        ]

    # ── 各 section 的工具构造 ───────────────────────────────────────────

    def _build_core_tools(self, _user_id: str) -> List[RegisteredTool]:
        time_tool = self.time_tool
        return [
            RegisteredTool(
                spec=ToolSpec(
                    name="get_current_time",
                    description="Read the current local date and time.",
                    parameters=_object_schema({}, []),
                ),
                handler=_sync_handler(lambda **_kwargs: time_tool.execute()),
            ),
        ]

    def _build_workspace_read_tools(self, _user_id: str) -> List[RegisteredTool]:
        runtime = self.runtime

        def _list_directory(path: str = ".") -> str:
            return runtime.list_directory(path)

        def _read_file(path: str) -> str:
            return runtime.read_file(path)

        def _search_files(query: str, path: str = ".") -> str:
            return runtime.search_files(query, path)

        return [
            RegisteredTool(
                spec=ToolSpec(
                    name="list_directory",
                    description="List files and directories inside the workspace.",
                    parameters=_object_schema(
                        {"path": _string_field("要列出的目录路径，相对 workspace 根目录", default=".")},
                        [],
                    ),
                ),
                handler=_sync_handler(_list_directory),
            ),
            RegisteredTool(
                spec=ToolSpec(
                    name="read_file",
                    description="Read file contents from the workspace.",
                    parameters=_object_schema(
                        {"path": _string_field("要读取的文件路径，相对 workspace 根目录")},
                        ["path"],
                    ),
                ),
                handler=_sync_handler(_read_file),
            ),
            RegisteredTool(
                spec=ToolSpec(
                    name="search_files",
                    description="Search workspace files for matching text.",
                    parameters=_object_schema(
                        {
                            "query": _string_field("要搜索的关键词"),
                            "path": _string_field("搜索起始路径，相对 workspace 根目录", default="."),
                        },
                        ["query"],
                    ),
                ),
                handler=_sync_handler(_search_files),
            ),
        ]

    def _build_workspace_edit_tools(self, user_id: str) -> List[RegisteredTool]:
        runtime = self.runtime

        def _preview_write(path: str, content: str) -> str:
            return runtime.preview_write_file(path, content, owner_user_id=user_id)

        def _preview_edit(path: str, find_text: str, replace_text: str) -> str:
            return runtime.preview_edit_file(path, find_text, replace_text, owner_user_id=user_id)

        def _preview_append(path: str, content: str, position: str = "end") -> str:
            return runtime.preview_append_file(path, content, position=position, owner_user_id=user_id)

        return [
            RegisteredTool(
                spec=ToolSpec(
                    name="preview_write_file",
                    description="Preview creating or overwriting a file.",
                    parameters=_object_schema(
                        {
                            "path": _string_field("要写入的文件路径，相对 workspace 根目录"),
                            "content": _string_field("写入后的完整文件内容"),
                        },
                        ["path", "content"],
                    ),
                ),
                handler=_sync_handler(_preview_write),
            ),
            RegisteredTool(
                spec=ToolSpec(
                    name="preview_edit_file",
                    description="Preview a precise in-file edit.",
                    parameters=_object_schema(
                        {
                            "path": _string_field("要修改的文件路径，相对 workspace 根目录"),
                            "find_text": _string_field("待替换的原始文本片段，必须足够精确"),
                            "replace_text": _string_field("替换后的文本片段"),
                        },
                        ["path", "find_text", "replace_text"],
                    ),
                ),
                handler=_sync_handler(_preview_edit),
            ),
            RegisteredTool(
                spec=ToolSpec(
                    name="preview_append_file",
                    description="Preview appending content to the start or end of a file.",
                    parameters=_object_schema(
                        {
                            "path": _string_field("要追加内容的文件路径，相对 workspace 根目录"),
                            "content": _string_field("要追加的文本内容"),
                            "position": _string_field("追加位置，只支持 start 或 end", default="end"),
                        },
                        ["path", "content"],
                    ),
                ),
                handler=_sync_handler(_preview_append),
            ),
        ]

    def _build_workspace_rename_tools(self, _user_id: str) -> List[RegisteredTool]:
        runtime = self.runtime

        def _rename(source_path: str, target_path: str) -> str:
            return runtime.rename_path(source_path, target_path)

        return [
            RegisteredTool(
                spec=ToolSpec(
                    name="rename_path",
                    description="Rename or move a file or directory inside the workspace.",
                    parameters=_object_schema(
                        {
                            "source_path": _string_field("源文件或目录路径，相对 workspace 根目录"),
                            "target_path": _string_field("目标文件或目录路径，相对 workspace 根目录"),
                        },
                        ["source_path", "target_path"],
                    ),
                ),
                handler=_sync_handler(_rename),
            ),
        ]

    def _build_command_tools(self, user_id: str) -> List[RegisteredTool]:
        runtime = self.runtime

        def _run_command(command: str, timeout_seconds: int = 20) -> str:
            return runtime.run_command(command, timeout_seconds, owner_user_id=user_id)

        return [
            RegisteredTool(
                spec=ToolSpec(
                    name="run_command",
                    description="Run a command in the workspace.",
                    parameters=_object_schema(
                        {
                            "command": _string_field("要执行的命令，默认在 workspace 根目录执行"),
                            "timeout_seconds": _integer_field(
                                "命令超时时间，单位秒，建议 5 到 20 秒",
                                default=20,
                            ),
                        },
                        ["command"],
                    ),
                ),
                handler=_sync_handler(_run_command),
            ),
        ]

    def _build_kb_tools(self, user_id: str) -> List[RegisteredTool]:
        rag_service = self.rag_service

        def _kb_list_assets() -> str:
            return build_kb_list_assets_response(rag_service, user_id)

        def _kb_search(
            query: str,
            doc_filter: str = "",
            top_k: int = 3,
            category: str = "",
        ) -> str:
            return build_kb_search_response(
                rag_service,
                user_id,
                query,
                top_k=top_k,
                doc_filter=doc_filter,
                category=category,
            )

        return [
            RegisteredTool(
                spec=ToolSpec(
                    name="kb_list_assets",
                    description="List current knowledge-base documents, categories, and heading previews.",
                    parameters=_object_schema({}, []),
                ),
                handler=_sync_handler(_kb_list_assets),
            ),
            RegisteredTool(
                spec=ToolSpec(
                    name="kb_search",
                    description="Search the knowledge base for relevant content.",
                    parameters=_object_schema(
                        {
                            "query": _string_field("知识库搜索问题或关键词"),
                            "doc_filter": _string_field("可选，限定某个文档名", default=""),
                            "top_k": _integer_field("返回结果数量，建议 1 到 5", default=3),
                            "category": _string_field("可选，限定某个分类", default=""),
                        },
                        ["query"],
                    ),
                ),
                handler=_sync_handler(_kb_search),
            ),
        ]

    def _build_web_tools(self, _user_id: str) -> List[RegisteredTool]:
        search_tool = SearchTool(api_key=self.search_api_key)

        def _web_search(query: str) -> str:
            return search_tool.execute(query=query)

        return [
            RegisteredTool(
                spec=ToolSpec(
                    name="web_search",
                    description="Search the web for up-to-date information.",
                    parameters=_object_schema(
                        {"query": _string_field("搜索关键词或问题")},
                        ["query"],
                    ),
                ),
                handler=_sync_handler(_web_search),
            ),
        ]

    def _dedupe_profiles(self, profiles: List[str]) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for item in profiles:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered
