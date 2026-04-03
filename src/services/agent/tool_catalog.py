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
