"""
网络搜索服务（简化版）
使用 Tavily 进行网络搜索
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger('wecom')


class NetworkSearchService:
    """网络搜索服务 - 基于 Tavily"""
    
    def __init__(self, api_key: str = None):
        """
        初始化网络搜索服务
        
        Args:
            api_key: Tavily API Key
        """
        self.api_key = api_key
    
    def search(self, query: str) -> Dict[str, str]:
        """
        执行网络搜索
        
        Args:
            query: 搜索查询
        
        Returns:
            {'original': 搜索结果文本} 或 {'original': None}（失败时）
        """
        result = {'original': None}
        
        if not self.api_key:
            logger.warning("[网络搜索] 未配置 API Key")
            return result
        
        try:
            from tavily import TavilyClient
            
            client = TavilyClient(api_key=self.api_key)
            response = client.search(query=query, search_depth="advanced")
            
            # 格式化结果
            context_list = []
            nsfw_keywords = ["porn", "sex", "erotic", "xxx", "hentai", "nude"]
            
            for r in response.get('results', []):
                content = r.get('content', '')
                title = r.get('title', '')
                url = r.get('url', '')
                
                # NSFW 过滤
                if any(k in title.lower() or k in content.lower() for k in nsfw_keywords):
                    continue
                
                content = content[:500]
                if len(content) < 20:
                    continue
                
                entry = f"【{title}】\n{content}\n来源: {url}"
                context_list.append(entry)
                
                if sum(len(e) for e in context_list) > 3000:
                    break
            
            if context_list:
                result['original'] = f"以下是关于「{query}」的网络搜索结果：\n\n" + "\n\n---\n\n".join(context_list)
                logger.info(f"[网络搜索] 成功获取 {len(context_list)} 条结果")
            
        except ImportError:
            logger.error("[网络搜索] 未安装 tavily-python，请运行 pip install tavily-python")
        except Exception as e:
            logger.error(f"[网络搜索] 失败: {e}")
        
        return result
