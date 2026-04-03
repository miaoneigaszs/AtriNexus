from __future__ import annotations

from typing import List, Optional

from langchain_core.tools import StructuredTool

from src.services.tools.search_tool import SearchTool
from src.services.tools.time_tool import TimeTool


def build_langchain_tools(search_api_key: Optional[str] = None) -> List[StructuredTool]:
    time_tool = TimeTool()

    def get_current_time() -> str:
        return time_tool.execute()

    tools: List[StructuredTool] = [
        StructuredTool.from_function(
            func=get_current_time,
            name=time_tool.name,
            description=time_tool.schema()["function"]["description"],
        )
    ]

    if search_api_key:
        search_tool = SearchTool(api_key=search_api_key)

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
