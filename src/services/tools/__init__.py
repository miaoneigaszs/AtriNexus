"""
Tool Layer — 工具注册与调度

用法示例：

    from src.services.tools import ToolRegistry
    from src.services.tools.time_tool import TimeTool
    from src.services.tools.search_tool import SearchTool

    registry = ToolRegistry()
    registry.register(TimeTool())
    registry.register(SearchTool(api_key="..."))

    # 获取 OpenAI function calling schemas
    schemas = registry.get_schemas()

    # 执行 LLM 选中的工具
    result = registry.execute("get_current_time")
    result = registry.execute("web_search", query="今天新闻")
"""

import json
import logging
from typing import Optional

from src.services.tools.base_tool import BaseTool

logger = logging.getLogger('wecom')

__all__ = ['ToolRegistry', 'BaseTool']


class ToolRegistry:
    """
    工具注册表。
    
    负责：
    - 注册/注销工具
    - 向 LLM 提供 function calling schemas
    - 按名称分发工具调用
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """注册一个工具"""
        self._tools[tool.name] = tool
        logger.debug(f"[ToolRegistry] 注册工具: {tool.name}")

    def unregister(self, name: str) -> None:
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"[ToolRegistry] 注销工具: {name}")

    def has(self, name: str) -> bool:
        """是否注册了某工具"""
        return name in self._tools

    def get_schemas(self) -> list[dict]:
        """
        返回所有已注册工具的 OpenAI function calling 格式 schema 列表。
        传给 chat.completions.create(tools=...) 参数。
        """
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments_json: str = "{}") -> str:
        """
        执行工具调用。
        
        Args:
            name:            工具名称（来自 tool_call.function.name）
            arguments_json:  JSON 字符串参数（来自 tool_call.function.arguments）
        
        Returns:
            工具执行结果字符串，用于构造 role=tool 的 message content
        """
        tool = self._tools.get(name)
        if not tool:
            logger.warning(f"[ToolRegistry] 未找到工具: {name}，已注册: {list(self._tools.keys())}")
            return f"未知工具: {name}"

        try:
            kwargs = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError as e:
            logger.error(f"[ToolRegistry] 工具参数解析失败: {e}, raw={arguments_json!r}")
            kwargs = {}

        try:
            logger.info(f"[ToolRegistry] 执行工具: {name}，参数: {kwargs}")
            return tool.execute(**kwargs)
        except Exception as e:
            logger.error(f"[ToolRegistry] 工具 {name} 执行异常: {e}", exc_info=True)
            return f"工具执行出错: {str(e)}"

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry({list(self._tools.keys())})"
