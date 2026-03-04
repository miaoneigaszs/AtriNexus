"""
网络搜索工具

包装 NetworkSearchService，作为 LLM function calling 的搜索工具。
"""

import logging

from src.services.tools.base_tool import BaseTool

logger = logging.getLogger('wecom')


class SearchTool(BaseTool):
    """基于 Tavily 的网络搜索工具"""

    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "web_search"

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "搜索互联网获取最新信息。"
                    "当用户询问实时新闻、最新事件、或你的训练数据中没有的信息时调用此工具。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "搜索关键词或问题"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    def execute(self, **kwargs) -> str:
        """执行网络搜索，返回格式化结果字符串"""
        query = kwargs.get("query", "").strip()
        if not query:
            return "错误：缺少搜索关键词"

        try:
            from src.services.ai.network_search_service import NetworkSearchService
            search_service = NetworkSearchService(api_key=self._api_key)
            result = search_service.search(query)

            if result.get('original'):
                logger.info(f"[SearchTool] 搜索成功: {query}")
                return result['original']
            else:
                return f"未找到关于「{query}」的相关信息"

        except Exception as e:
            logger.error(f"[SearchTool] 搜索失败: {e}", exc_info=True)
            return f"网络搜索出错: {str(e)}"
