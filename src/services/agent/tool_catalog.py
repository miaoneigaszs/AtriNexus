from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.services.agent.runtime import WorkspaceRuntime
from src.services.tools.search_tool import SearchTool
from src.services.tools.time_tool import TimeTool


class ReadFileInput(BaseModel):
    path: str = Field(description="要读取的文件路径，相对 workspace 根目录")


class ListDirectoryInput(BaseModel):
    path: str = Field(default=".", description="要列出的目录路径，相对 workspace 根目录")


class SearchFilesInput(BaseModel):
    query: str = Field(description="要搜索的关键词")
    path: str = Field(default=".", description="搜索起始路径，相对 workspace 根目录")


class RunCommandInput(BaseModel):
    command: str = Field(description="要执行的命令。默认在 workspace 根目录执行。")
    timeout_seconds: int = Field(default=20, description="命令超时时间，单位秒，建议 5 到 20 秒")


class WriteFileInput(BaseModel):
    path: str = Field(description="要写入的文件路径，相对 workspace 根目录")
    content: str = Field(description="写入后的完整文件内容")


class ReplaceInFileInput(BaseModel):
    path: str = Field(description="要编辑的文件路径，相对 workspace 根目录")
    old_text: str = Field(description="待替换的原始文本片段")
    new_text: str = Field(description="替换后的文本片段")
    replace_all: bool = Field(default=False, description="是否替换全部匹配项，默认只替换 1 处")


class ToolCatalog:
    def __init__(self, workspace_root: str, search_api_key: Optional[str] = None) -> None:
        self.runtime = WorkspaceRuntime(workspace_root)
        self.search_api_key = search_api_key

    def build_tools(self) -> List[StructuredTool]:
        time_tool = TimeTool()

        def get_current_time() -> str:
            return time_tool.execute()

        tools: List[StructuredTool] = [
            StructuredTool.from_function(
                func=get_current_time,
                name=time_tool.name,
                description=time_tool.schema()["function"]["description"],
            ),
            StructuredTool.from_function(
                func=self.runtime.list_directory,
                name="list_directory",
                description="列出 workspace 内指定目录的文件和子目录。",
                args_schema=ListDirectoryInput,
            ),
            StructuredTool.from_function(
                func=self.runtime.read_file,
                name="read_file",
                description="读取 workspace 内指定文本文件的内容。",
                args_schema=ReadFileInput,
            ),
            StructuredTool.from_function(
                func=self.runtime.search_files,
                name="search_files",
                description="在 workspace 内按关键词搜索文件内容。",
                args_schema=SearchFilesInput,
            ),
            StructuredTool.from_function(
                func=self.runtime.run_command,
                name="run_command",
                description="在 workspace 根目录执行命令并返回结果。命令有超时限制，明显危险命令会被拒绝。",
                args_schema=RunCommandInput,
            ),
            StructuredTool.from_function(
                func=self.runtime.write_file,
                name="write_file",
                description="直接写入 workspace 内文件。若文件存在则覆盖写入。",
                args_schema=WriteFileInput,
            ),
            StructuredTool.from_function(
                func=self.runtime.replace_in_file,
                name="replace_in_file",
                description="在 workspace 文件中执行文本替换。默认只替换 1 处，可显式指定 replace_all=true。",
                args_schema=ReplaceInFileInput,
            ),
        ]

        if self.search_api_key:
            search_tool = SearchTool(api_key=self.search_api_key)

            def web_search(query: str) -> str:
                return search_tool.execute(query=query)

            tools.append(
                StructuredTool.from_function(
                    func=web_search,
                    name=search_tool.name,
                    description=search_tool.schema()["function"]["description"],
                )
            )

        return tools
