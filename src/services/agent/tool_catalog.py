from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Annotated, List, Optional

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from src.services.agent.runtime import WorkspaceRuntime
from src.services.agent.kb_tools import build_kb_list_assets_response, build_kb_search_response
from src.services.agent.tool_profiles import (
    normalize_tool_profile,
    should_enable_command,
    should_enable_web,
    should_enable_workspace_edit,
    should_enable_workspace_read,
)
from src.services.tools.search_tool import SearchTool
from src.services.tools.time_tool import TimeTool


@dataclass(frozen=True)
class ToolBundle:
    tools: List[BaseTool]
    profiles: List[str]
    summary_lines: List[str]

    @property
    def summary(self) -> str:
        if not self.summary_lines:
            return "- 当前无需额外工具。"
        return "\n".join(self.summary_lines)


class ToolCatalog:
    """构建当前会话可见的工具列表，并生成工具摘要。"""

    FILE_HINT_PATTERN = re.compile(
        r"(\.py|\.md|\.txt|\.json|\.yaml|\.yml|\.toml|\.env|\.ini|txt|md|json|yaml|yml|toml|env|ini|py|/|\\|README|readme|config|日志|目录|文件|代码|文档|内容|写的什么|里面写了什么|里面有什么)",
        re.IGNORECASE,
    )
    WRITE_HINT_PATTERN = re.compile(
        r"(修改|改成|改一下|改一改|改短|缩短|精简|重写|润色|替换|追加|写入|写到|创建|新增|补充|覆盖|删除.*内容|加上|加一段|更新)",
        re.IGNORECASE,
    )
    RENAME_HINT_PATTERN = re.compile(
        r"(重命名|改名|改文件名|改成.*\.|移动到|挪到|rename)",
        re.IGNORECASE,
    )
    DOCUMENT_HINT_PATTERN = re.compile(
        r"(\.md|README|readme|文档|说明|白皮书|报告|手册|注释|提示词|prompt)",
        re.IGNORECASE,
    )
    TOOL_HINT_PATTERN = re.compile(
        r"(有哪些工具|有什么工具|能用什么工具|可以用什么工具|能做什么|会什么|能力有哪些)",
        re.IGNORECASE,
    )
    COMMAND_HINT_PATTERN = re.compile(
        r"(`[^`]+`)|(执行|运行|跑一下|命令|终端|shell|git |python |pytest|npm |pnpm |uv |pip )",
        re.IGNORECASE,
    )
    WEB_HINT_PATTERN = re.compile(
        r"(最新|实时|今天|刚刚|新闻|联网|上网|搜索一下|查一下网上|web|互联网)",
        re.IGNORECASE,
    )

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
            return ToolBundle(tools=[], profiles=[], summary_lines=[])

        runtime = self.runtime
        message = message or ""
        profile = normalize_tool_profile(tool_profile)
        include_tool_overview = self._needs_tool_overview(message)
        include_workspace_reads = include_tool_overview or should_enable_workspace_read(profile)
        include_write_preview = should_enable_workspace_edit(profile)
        include_rename = should_enable_workspace_edit(profile)
        include_command = include_tool_overview or should_enable_command(profile)
        include_web = bool(self.search_api_key and (include_tool_overview or should_enable_web(profile)))
        include_kb = self.rag_service is not None

        tools: List[BaseTool] = []
        profiles: List[str] = ["core", profile]
        summary_lines: List[str] = []

        @tool
        def get_current_time() -> str:
            """Read the current local date and time."""
            return self.time_tool.execute()

        tools.append(get_current_time)
        summary_lines.append("- get_current_time: 读取当前本地日期和时间。")

        if include_workspace_reads:
            profiles.append("workspace-read")

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

            tools.extend([list_directory, read_file, search_files])
            summary_lines.extend(
                [
                    "- list_directory: 列出目录内容。",
                    "- read_file: 读取文件内容。",
                    "- search_files: 按关键词搜索文件内容。",
                ]
            )

        if include_write_preview:
            profiles.append("workspace-edit-preview")

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

            tools.extend([preview_write_file, preview_edit_file, preview_append_file])
            summary_lines.extend(
                [
                    "- preview_write_file: 预览整文件写入或重写。",
                    "- preview_edit_file: 预览精确局部修改。",
                    "- preview_append_file: 预览在文件头部或尾部追加内容。",
                ]
            )

        if include_rename:
            profiles.append("workspace-rename")

            @tool
            def rename_path(
                source_path: Annotated[str, Field(description="源文件或目录路径，相对 workspace 根目录")],
                target_path: Annotated[str, Field(description="目标文件或目录路径，相对 workspace 根目录")],
            ) -> str:
                """Rename or move a file or directory inside the workspace."""
                return runtime.rename_path(source_path, target_path)

            tools.append(rename_path)
            summary_lines.append("- rename_path: 重命名或移动 workspace 内的文件/目录。")

        if include_command:
            profiles.append("command")

            @tool
            def run_command(
                command: Annotated[str, Field(description="要执行的命令，默认在 workspace 根目录执行")],
                timeout_seconds: Annotated[int, Field(description="命令超时时间，单位秒，建议 5 到 20 秒")] = 20,
            ) -> str:
                """Run a command in the workspace."""
                return runtime.run_command(command, timeout_seconds, owner_user_id=user_id)

            tools.append(run_command)
            summary_lines.append("- run_command: 执行命令。安全命令直接执行，复杂或高风险命令进入确认流程。")

        if include_kb:
            profiles.append("kb")

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

            tools.extend([kb_list_assets, kb_search])
            summary_lines.extend(
                [
                    "- kb_list_assets: 查看当前知识库里有哪些文档、分类和标题摘要。",
                    "- kb_search: 在知识库中搜索相关内容，可选限定文档或分类。",
                ]
            )

        if include_web:
            profiles.append("web")
            search_tool = SearchTool(api_key=self.search_api_key)

            @tool
            def web_search(
                query: Annotated[str, Field(description="搜索关键词或问题")],
            ) -> str:
                """Search the web for up-to-date information."""
                return search_tool.execute(query=query)

            tools.append(web_search)
            summary_lines.append("- web_search: 搜索互联网，获取最新信息。")

        return ToolBundle(tools=tools, profiles=profiles, summary_lines=summary_lines)

    def infer_tool_profile(self, message: str) -> str:
        message = message or ""
        if self._needs_command_tool(message):
            return "workspace_exec"
        if (
            self._needs_write_tools(message)
            or self._needs_rename_tool(message)
            or self._looks_like_document_editable_task(message)
        ):
            return "workspace_edit"
        if self._needs_workspace_tools(message):
            return "workspace_read"
        return "chat"

    def _needs_workspace_tools(self, message: str) -> bool:
        lowered = message.lower()
        return bool(
            self.FILE_HINT_PATTERN.search(message)
            or self.WRITE_HINT_PATTERN.search(message)
            or "readme" in lowered
            or "kb" in lowered
        )

    def _needs_write_tools(self, message: str) -> bool:
        return bool(self.WRITE_HINT_PATTERN.search(message))

    def _looks_like_document_editable_task(self, message: str) -> bool:
        return bool(self.DOCUMENT_HINT_PATTERN.search(message))

    def _needs_rename_tool(self, message: str) -> bool:
        return bool(self.RENAME_HINT_PATTERN.search(message))

    def _needs_command_tool(self, message: str) -> bool:
        return bool(self.COMMAND_HINT_PATTERN.search(message))

    def _needs_web_tool(self, message: str) -> bool:
        return bool(self.WEB_HINT_PATTERN.search(message))

    def _needs_tool_overview(self, message: str) -> bool:
        return bool(self.TOOL_HINT_PATTERN.search(message))
