"""
去重中间件
负责检查消息是否重复处理
"""

import logging
from src.services.database import Session, ChatMessage

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
        session = Session()
        try:
            exists = session.query(ChatMessage).filter_by(wecom_msg_id=msg_id).first()
            return exists is not None
        finally:
            session.close()
