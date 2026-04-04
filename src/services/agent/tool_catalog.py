from typing import Annotated, List, Optional

from langchain_core.tools import BaseTool, tool
from pydantic import Field

from src.services.agent.runtime import WorkspaceRuntime
from src.services.tools.search_tool import SearchTool
from src.services.tools.time_tool import TimeTool


class ToolCatalog:
    def __init__(self, workspace_root: str, search_api_key: Optional[str] = None) -> None:
        self.runtime = WorkspaceRuntime(workspace_root)
        self.search_api_key = search_api_key

    def build_tools(self) -> List[BaseTool]:
        runtime = self.runtime
        time_tool = TimeTool()
        tools: List[BaseTool] = []

        @tool
        def get_current_time() -> str:
            """获取当前的日期和时间。用户询问时间、日期，或需要当前时间来回答问题时调用。"""
            return time_tool.execute()

        @tool
        def list_directory(
            path: Annotated[str, Field(description="要列出的目录路径，相对 workspace 根目录")] = ".",
        ) -> str:
            """列出 workspace 内指定目录的文件和子目录。"""
            return runtime.list_directory(path)

        @tool
        def read_file(
            path: Annotated[str, Field(description="要读取的文件路径，相对 workspace 根目录")],
        ) -> str:
            """读取 workspace 内指定文本文件的内容。"""
            return runtime.read_file(path)

        @tool
        def search_files(
            query: Annotated[str, Field(description="要搜索的关键词")],
            path: Annotated[str, Field(description="搜索起始路径，相对 workspace 根目录")] = ".",
        ) -> str:
            """在 workspace 内按关键词搜索文件内容。"""
            return runtime.search_files(query, path)

        @tool
        def run_command(
            command: Annotated[str, Field(description="要执行的命令，默认在 workspace 根目录执行")],
            timeout_seconds: Annotated[int, Field(description="命令超时时间，单位秒，建议 5 到 20 秒")] = 20,
        ) -> str:
            """在 workspace 根目录执行命令并返回结果。命令有超时限制，明显危险命令会被拒绝。"""
            return runtime.run_command(command, timeout_seconds)

        @tool
        def write_file(
            path: Annotated[str, Field(description="要写入的文件路径，相对 workspace 根目录")],
            content: Annotated[str, Field(description="写入后的完整文件内容")],
        ) -> str:
            """直接写入 workspace 内文件；若文件存在则覆盖写入。"""
            return runtime.write_file(path, content)

        @tool
        def replace_in_file(
            path: Annotated[str, Field(description="要编辑的文件路径，相对 workspace 根目录")],
            old_text: Annotated[str, Field(description="待替换的原始文本片段")],
            new_text: Annotated[str, Field(description="替换后的文本片段")],
            replace_all: Annotated[bool, Field(description="是否替换全部匹配项，默认只替换 1 处")] = False,
        ) -> str:
            """在 workspace 文件中执行文本替换。默认只替换 1 处，可显式指定 replace_all=true。"""
            return runtime.replace_in_file(path, old_text, new_text, replace_all)

        tools.extend(
            [
                get_current_time,
                list_directory,
                read_file,
                search_files,
                run_command,
                write_file,
                replace_in_file,
            ]
        )

        if self.search_api_key:
            search_tool = SearchTool(api_key=self.search_api_key)

            @tool
            def web_search(
                query: Annotated[str, Field(description="搜索关键词或问题")],
            ) -> str:
                """搜索互联网获取最新信息。用户询问实时新闻、最新事件或训练数据中没有的信息时调用。"""
                return search_tool.execute(query=query)

            tools.append(web_search)

        return tools
