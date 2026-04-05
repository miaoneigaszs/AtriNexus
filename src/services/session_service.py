"""
会话状态管理服务
统一管理用户会话状态和KB检索会话
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from src.services.database import Session, SessionState, KBSearchSession

logger = logging.getLogger('wecom')


class SessionService:
    """会话状态管理服务"""
    
    def __init__(self, kb_session_timeout: int = 5):
        """
        初始化会话服务
        
        Args:
            kb_session_timeout: KB检索会话超时时间（分钟）
        """
        self.kb_session_timeout = kb_session_timeout
    
    # ---------- 用户会话状态管理 ----------
    
    def get_session(self, user_id: str) -> SessionState:
        """获取或创建用户会话状态"""
        with Session(expire_on_commit=False) as session:
            state = session.query(SessionState).filter_by(user_id=user_id).first()
            if not state:
                state = SessionState(user_id=user_id, mode='work')
                session.add(state)
                session.commit()
                logger.info(f"为用户 {user_id} 创建新会话 (mode=work)")
            else:
                session.expire(state)
                session.refresh(state)
            return state
    
    def update_session_variables(self, user_id: str, variables: dict):
        """更新用户会话上下文变量"""
        with Session() as session:
            state = session.query(SessionState).filter_by(user_id=user_id).first()
            if state:
                state.variables = json.dumps(variables, ensure_ascii=False)
                session.commit()
    
    def update_session_mode(self, user_id: str, mode: str):
        """更新用户模式"""
        with Session() as session:
            state = session.query(SessionState).filter_by(user_id=user_id).first()
            if state:
                state.mode = mode
                state.last_active = datetime.now()
                session.commit()
                logger.info(f"用户 {user_id} 模式切换为 {mode}")
    
    # ---------- KB检索会话管理 ----------
    
    def get_kb_search_session(self, user_id: str) -> Optional[KBSearchSession]:
        """获取用户当前的知识库检索会话"""
        with Session() as session:
            kb_session = session.query(KBSearchSession).filter_by(user_id=user_id).first()
            if kb_session and kb_session.expires_at and kb_session.expires_at < datetime.now():
                session.delete(kb_session)
                session.commit()
                return None
            return kb_session
    
    def create_kb_search_session(self, user_id: str, original_query: str,
                                  waiting_for: str, candidates: List[Dict],
                                  current_filter: str = ""):
        """创建知识库检索会话"""
        with Session() as session:
            try:
                old_session = session.query(KBSearchSession).filter_by(user_id=user_id).first()
                if old_session:
                    session.delete(old_session)

                kb_session = KBSearchSession(
                    user_id=user_id,
                    original_query=original_query,
                    current_filter=current_filter,
                    waiting_for=waiting_for,
                    candidates=json.dumps(candidates, ensure_ascii=False),
                    expires_at=datetime.now() + timedelta(minutes=self.kb_session_timeout)
                )
                session.add(kb_session)
                session.commit()
            except Exception as e:
                logger.error(f"创建KB检索会话失败: {e}")
                session.rollback()
    
    def clear_kb_search_session(self, user_id: str):
        """清除知识库检索会话"""
        with Session() as session:
            try:
                kb_session = session.query(KBSearchSession).filter_by(user_id=user_id).first()
                if kb_session:
                    session.delete(kb_session)
                    session.commit()
            except Exception as e:
                logger.error(f"清除KB检索会话失败: {e}")
                session.rollback()
