"""
去重中间件
负责检查消息是否重复处理
"""

import logging
from src.platform_core.database import ChatMessage
from src.platform_core.db_session import new_session

logger = logging.getLogger('wecom')


class DedupMiddleware:
    """消息去重中间件"""

    @staticmethod
    def is_duplicate_message(msg_id: str) -> bool:
        """
        检查消息是否已处理过

        Args:
            msg_id: 消息ID

        Returns:
            bool: 是否重复
        """
        with new_session() as session:
            exists = session.query(ChatMessage).filter_by(wecom_msg_id=msg_id).first()
            return exists is not None
