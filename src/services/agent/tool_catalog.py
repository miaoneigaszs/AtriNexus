from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Annotated, List, Optional

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from src.services.agent.runtime import WorkspaceRuntime
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
    """负责两件事：构建工具列表，生成给模型看的工具摘要。"""

    FILE_HINT_PATTERN = re.compile(
        r"(\.py|\.md|\.txt|\.json|\.yaml|\.yml|\.toml|\.env|\.ini|/|\\|README|config|日志|目录|文件|代码)",
        re.IGNORECASE,
    )
    WRITE_HINT_PATTERN = re.compile(
        r"(修改|改成|改一下|改一改|改短|缩短|精简|重写|润色|替换|追加|写入|写到|创建|新增|补充|覆盖|删除.*内容|加上|加一段|更新)",
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

    def __init__(self, workspace_root: str, search_api_key: Optional[str] = None) -> None:
        self.runtime = WorkspaceRuntime(workspace_root)
        self.search_api_key = search_api_key
        self.time_tool = TimeTool()

    def build_tool_bundle(self, user_id: str, message: str, allow_tools: bool = True) -> ToolBundle:
        if not allow_tools:
            return ToolBundle(tools=[], profiles=[], summary_lines=[])

        runtime = self.runtime
        time_tool = self.time_tool
        message = message or ""
        include_workspace_reads = self._needs_workspace_tools(message)
        include_write_preview = self._needs_write_tools(message)
        include_command = self._needs_command_tool(message)
        include_web = bool(self.search_api_key and self._needs_web_tool(message))

        tools: List[BaseTool] = []
        profiles: List[str] = ["core"]
        summary_lines: List[str] = []

        @tool
        def get_current_time() -> str:
            """获取当前日期和时间。用户询问时间、日期或星期时使用。"""
            return time_tool.execute()

        tools.append(get_current_time)
        summary_lines.append("- get_current_time: 查询当前日期和时间。用户问时间、日期、星期时使用。")

        if include_workspace_reads:
            profiles.append("workspace-read")
            @tool
            def list_directory(
                path: Annotated[str, Field(description="要列出的目录路径，相对 workspace 根目录")] = ".",
            ) -> str:
                """列出 workspace 内指定目录的文件和子目录。用户提到目录、文件夹、项目结构时使用。"""
                return runtime.list_directory(path)

            @tool
            def read_file(
                path: Annotated[str, Field(description="要读取的文件路径，相对 workspace 根目录")],
            ) -> str:
                """读取文件内容。用户提到文件名、配置、代码片段、日志时使用。"""
                return runtime.read_file(path)

            @tool
            def search_files(
                query: Annotated[str, Field(description="要搜索的关键词")],
                path: Annotated[str, Field(description="搜索起始路径，相对 workspace 根目录")] = ".",
            ) -> str:
                """在 workspace 内按关键词搜索文件内容。用户要找代码、配置、文本片段时使用。"""
                return runtime.search_files(query, path)

            tools.extend([list_directory, read_file, search_files])
            summary_lines.extend(
                [
                    "- list_directory: 查看目录结构。用户提到目录、文件夹、项目结构时使用。",
                    "- read_file: 读取文件内容。用户提到文件名、配置、日志或代码时使用。",
                    "- search_files: 按关键词搜索文件内容。用户想找某段代码、文本或配置时使用。",
                ]
            )

        if include_write_preview:
            profiles.append("workspace-edit-preview")
            @tool
            def preview_write_file(
                path: Annotated[str, Field(description="要写入的文件路径，相对 workspace 根目录")],
                content: Annotated[str, Field(description="写入后的完整文件内容")],
            ) -> str:
                """生成文件写入预览，不直接落盘。用户要求新建文件或整体覆盖文件时使用。"""
                return runtime.preview_write_file(path, content, owner_user_id=user_id)

            @tool
            def preview_edit_file(
                path: Annotated[str, Field(description="要修改的文件路径，相对 workspace 根目录")],
                find_text: Annotated[str, Field(description="待替换的原始文本片段，必须足够精确")],
                replace_text: Annotated[str, Field(description="替换后的文本片段")],
            ) -> str:
                """生成文件局部修改预览，不直接落盘。用户要求修改、追加、替换部分内容时使用。"""
                return runtime.preview_edit_file(path, find_text, replace_text, owner_user_id=user_id)

            tools.extend([preview_write_file, preview_edit_file])
            summary_lines.extend(
                [
                    "- preview_write_file: 生成整文件写入预览。用户要创建新文件或整体覆盖文件时使用。",
                    "- preview_edit_file: 生成局部修改预览。用户要替换、追加、修改文件部分内容时使用。",
                ]
            )

        if include_command:
            profiles.append("command")
            @tool
            def run_command(
                command: Annotated[str, Field(description="要执行的命令，默认在 workspace 根目录执行")],
                timeout_seconds: Annotated[int, Field(description="命令超时时间，单位秒，建议 5 到 20 秒")] = 20,
            ) -> str:
                """执行命令。用户要求运行脚本、执行 git/python/测试命令时使用。安全命令直接执行，复杂或高风险命令进入确认流程。"""
                return runtime.run_command(command, timeout_seconds, owner_user_id=user_id)

            tools.append(run_command)
            summary_lines.append("- run_command: 执行命令。安全命令直接执行；含 shell 操作符、未知可执行文件或高风险命令时进入确认流程。")

        if include_web:
            profiles.append("web")
            search_tool = SearchTool(api_key=self.search_api_key)

            @tool
            def web_search(
                query: Annotated[str, Field(description="搜索关键词或问题")],
            ) -> str:
                """搜索互联网获取最新信息。用户询问实时新闻、最新事件或训练数据中没有的信息时调用。"""
                return search_tool.execute(query=query)

            tools.append(web_search)
            summary_lines.append("- web_search: 搜索互联网。用户询问最新信息、实时事件、新闻时使用。")

        return ToolBundle(tools=tools, profiles=profiles, summary_lines=summary_lines)

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

    def _needs_command_tool(self, message: str) -> bool:
        return bool(self.COMMAND_HINT_PATTERN.search(message))

    def _needs_web_tool(self, message: str) -> bool:
        return bool(self.WEB_HINT_PATTERN.search(message))
