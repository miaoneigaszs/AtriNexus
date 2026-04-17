from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Callable, List, Optional

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from src.services.agent.kb_tools import build_kb_list_assets_response, build_kb_search_response
from src.services.agent.runtime import WorkspaceRuntime
from src.services.agent.tool_profiles import (
    normalize_tool_profile,
    should_enable_command,
    should_enable_web,
    should_enable_workspace_edit,
    should_enable_workspace_read,
)
from src.services.tools.search_tool import SearchTool
from src.services.tools.time_tool import TimeTool


TOOL_OVERVIEW_HINTS = (
    "有哪些工具",
    "有什么工具",
    "能用什么工具",
    "可以用什么工具",
    "能做什么",
    "会什么",
    "能力有哪些",
)


@dataclass(frozen=True)
class ToolBundle:
    tools: List[BaseTool]
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
    build_tools: Callable[["ToolCatalog", str], List[BaseTool]]


class ToolCatalog:
    """声明式工具目录。

    这里不再根据消息正则猜测“这句像不像文件请求”，而是根据稳定的
    session profile 决定当前能力边界。这样工具治理更接近 OpenClaw：
    profile 先稳定，再由 before-tool-call 和 fast-path 去做修正与拦截。
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
        tools: List[BaseTool] = []
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

    def _build_core_tools(self, _user_id: str) -> List[BaseTool]:
        @tool
        def get_current_time() -> str:
            """Read the current local date and time."""
            return self.time_tool.execute()

        return [get_current_time]

    def _build_workspace_read_tools(self, _user_id: str) -> List[BaseTool]:
        runtime = self.runtime

        @tool
        def list_directory(
            path: Annotated[str, Field(description="要列出的目录路径，相对 workspace 根目录")] = ".",
        ) -> str:
            """List files and directories inside the workspace."""
            return runtime.list_directory(path)

        @tool
        def read_file(
            path: Annotated[str, Field(description="要读取的文件路径，相对 workspace 根目录")],
        ) -> str:
            """Read file contents from the workspace."""
            return runtime.read_file(path)

        @tool
        def search_files(
            query: Annotated[str, Field(description="要搜索的关键词")],
            path: Annotated[str, Field(description="搜索起始路径，相对 workspace 根目录")] = ".",
        ) -> str:
            """Search workspace files for matching text."""
            return runtime.search_files(query, path)

        return [list_directory, read_file, search_files]

    def _build_workspace_edit_tools(self, user_id: str) -> List[BaseTool]:
        runtime = self.runtime

        @tool
        def preview_write_file(
            path: Annotated[str, Field(description="要写入的文件路径，相对 workspace 根目录")],
            content: Annotated[str, Field(description="写入后的完整文件内容")],
        ) -> str:
            """Preview creating or overwriting a file."""
            return runtime.preview_write_file(path, content, owner_user_id=user_id)

        @tool
        def preview_edit_file(
            path: Annotated[str, Field(description="要修改的文件路径，相对 workspace 根目录")],
            find_text: Annotated[str, Field(description="待替换的原始文本片段，必须足够精确")],
            replace_text: Annotated[str, Field(description="替换后的文本片段")],
        ) -> str:
            """Preview a precise in-file edit."""
            return runtime.preview_edit_file(path, find_text, replace_text, owner_user_id=user_id)

        @tool
        def preview_append_file(
            path: Annotated[str, Field(description="要追加内容的文件路径，相对 workspace 根目录")],
            content: Annotated[str, Field(description="要追加的文本内容")],
            position: Annotated[str, Field(description="追加位置，只支持 start 或 end")] = "end",
        ) -> str:
            """Preview appending content to the start or end of a file."""
            return runtime.preview_append_file(path, content, position=position, owner_user_id=user_id)

        return [preview_write_file, preview_edit_file, preview_append_file]

    def _build_workspace_rename_tools(self, _user_id: str) -> List[BaseTool]:
        runtime = self.runtime

        @tool
        def rename_path(
            source_path: Annotated[str, Field(description="源文件或目录路径，相对 workspace 根目录")],
            target_path: Annotated[str, Field(description="目标文件或目录路径，相对 workspace 根目录")],
        ) -> str:
            """Rename or move a file or directory inside the workspace."""
            return runtime.rename_path(source_path, target_path)

        return [rename_path]

    def _build_command_tools(self, user_id: str) -> List[BaseTool]:
        runtime = self.runtime

        @tool
        def run_command(
            command: Annotated[str, Field(description="要执行的命令，默认在 workspace 根目录执行")],
            timeout_seconds: Annotated[int, Field(description="命令超时时间，单位秒，建议 5 到 20 秒")] = 20,
        ) -> str:
            """Run a command in the workspace."""
            return runtime.run_command(command, timeout_seconds, owner_user_id=user_id)

        return [run_command]

    def _build_kb_tools(self, user_id: str) -> List[BaseTool]:
        @tool
        def kb_list_assets() -> str:
            """List current knowledge-base documents, categories, and heading previews."""
            return build_kb_list_assets_response(self.rag_service, user_id)

        @tool
        def kb_search(
            query: Annotated[str, Field(description="知识库搜索问题或关键词")],
            doc_filter: Annotated[str, Field(description="可选，限定某个文档名")] = "",
            top_k: Annotated[int, Field(description="返回结果数量，建议 1 到 5")] = 3,
            category: Annotated[str, Field(description="可选，限定某个分类")] = "",
        ) -> str:
            """Search the knowledge base for relevant content."""
            return build_kb_search_response(
                self.rag_service,
                user_id,
                query,
                top_k=top_k,
                doc_filter=doc_filter,
                category=category,
            )

        return [kb_list_assets, kb_search]

    def _build_web_tools(self, _user_id: str) -> List[BaseTool]:
        search_tool = SearchTool(api_key=self.search_api_key)

        @tool
        def web_search(
            query: Annotated[str, Field(description="搜索关键词或问题")],
        ) -> str:
            """Search the web for up-to-date information."""
            return search_tool.execute(query=query)

        return [web_search]

    def _dedupe_profiles(self, profiles: List[str]) -> List[str]:
        ordered: List[str] = []
        seen = set()
        for item in profiles:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered
