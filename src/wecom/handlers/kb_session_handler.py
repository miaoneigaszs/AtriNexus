"""
KB 会话处理器
负责处理知识库检索会话的状态管理
"""

import json
import logging
import re
from typing import Dict, Any, Optional

from src.services.session_service import SessionService

logger = logging.getLogger('wecom')


class KBSessionHandler:
    """知识库检索会话处理器"""

    def __init__(self, session_service: SessionService):
        """
        初始化 KB 会话处理器

        Args:
            session_service: 会话服务
        """
        self.session_service = session_service

    def handle_kb_session_response(self, user_id: str, content: str) -> Optional[Dict[str, Any]]:
        """
        处理用户在 KB 检索会话中的回复

        Args:
            user_id: 用户ID
            content: 消息内容

        Returns:
            Optional[Dict]: 返回 None 表示不在 KB 会话中，否则返回操作指令
                - {"action": "search_all", "original_query": str}: 用户选择搜索全部
                - {"action": "search_category", "category": str, "original_query": str}: 用户选择了分类
                - {"action": "clarify", "message": str}: 需要继续询问用户
        """
        kb_session = self.session_service.get_kb_search_session(user_id)
        if not kb_session:
            return None

        # 解析会话数据
        try:
            candidates = json.loads(kb_session.candidates)
        except Exception:
            self.session_service.clear_kb_search_session(user_id)
            return None

        original_query = kb_session.original_query or content
        content_trim = content.strip()

        # 处理分类选择
        if kb_session.waiting_for == "category":
            # 检查是否选择全部
            if "全部" in content_trim or "所有" in content_trim:
                self.session_service.clear_kb_search_session(user_id)
                return {"action": "search_all", "original_query": original_query}

            # 尝试解析数字选择
            try:
                numbers = re.findall(r'\d+', content_trim)
                if numbers:
                    choice = int(numbers[0]) - 1
                    if 0 <= choice < len(candidates):
                        selected_category = candidates[choice]
                        self.session_service.clear_kb_search_session(user_id)
                        return {"action": "search_category", "category": selected_category, "original_query": original_query}
            except Exception:
                pass

            # 尝试直接匹配分类名称
            for cat in candidates:
                if cat in content_trim:
                    self.session_service.clear_kb_search_session(user_id)
                    return {"action": "search_category", "category": cat, "original_query": original_query}

            # 无法识别，重新提示
            return {"action": "clarify", "message": f"未能理解您的选择，请回复数字 1-{len(candidates)} 或分类名称。\n\n"}

        # 未知状态，清除会话
        self.session_service.clear_kb_search_session(user_id)
        return None
