"""
BaseTool 抽象基类

所有工具（Tool）必须继承此类并实现三个方法：
- name:    工具的唯一名称（字符串属性）
- schema:  返回 OpenAI function calling 格式的 dict
- execute: 执行工具逻辑，接收 kwargs，返回字符串结果

未来集成 MCP / skills 时，只需实现同一接口即可接入。
"""

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """所有 LLM 工具的抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具的唯一名称，对应 function calling 中的 function.name"""
        ...

    @abstractmethod
    def schema(self) -> dict:
        """
        返回 OpenAI function calling 格式的工具描述。

        Returns:
            {
                "type": "function",
                "function": {
                    "name": ...,
                    "description": ...,
                    "parameters": { "type": "object", "properties": {...}, "required": [...] }
                }
            }
        """
        ...

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """
        执行工具，返回结果字符串（直接注入 LLM 的 tool message content）。

        Args:
            **kwargs: 由 LLM 的 tool_call.function.arguments 解析得来
        """
        ...
