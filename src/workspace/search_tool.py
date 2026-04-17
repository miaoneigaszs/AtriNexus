"""网络搜索工具。"""

import logging

logger = logging.getLogger('wecom')


class SearchTool:
    """基于 Tavily 的网络搜索工具。"""

    def __init__(self, api_key: str):
        self._api_key = api_key

    def execute(self, **kwargs) -> str:
        """执行网络搜索，返回格式化结果字符串"""
        query = kwargs.get("query", "").strip()
        if not query:
            return "错误：缺少搜索关键词"

        try:
            from src.ai.network_search_service import NetworkSearchService
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
