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

    def get_session_variables(self, user_id: str) -> Dict[str, Any]:
        """读取会话扩展变量。"""
        state = self.get_session(user_id)
        raw = state.variables or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("用户 %s 的 session variables 不是合法 JSON，已忽略", user_id)
            return {}
        return data if isinstance(data, dict) else {}

    def get_tool_profile(self, user_id: str) -> str:
        """读取当前会话绑定的工具 profile。"""
        variables = self.get_session_variables(user_id)
        return str(variables.get("tool_profile", "chat"))

    def get_current_mode(self, user_id: str) -> str:
        """读取当前会话模式。"""
        state = self.get_session(user_id)
        return str(state.mode or "work")

    def get_current_avatar(self, user_id: str) -> str:
        """读取当前会话绑定的人设名称。"""
        state = self.get_session(user_id)
        return str(state.avatar_name or "ATRI")

    def set_tool_profile(self, user_id: str, tool_profile: str) -> None:
        """更新当前会话绑定的工具 profile。"""
        with Session() as session:
            state = session.query(SessionState).filter_by(user_id=user_id).first()
            if not state:
                return

            try:
                variables = json.loads(state.variables or "{}")
            except json.JSONDecodeError:
                variables = {}

            variables["tool_profile"] = tool_profile
            state.variables = json.dumps(variables, ensure_ascii=False)
            state.last_active = datetime.now()
            session.commit()

    def get_last_workspace_target(self, user_id: str) -> Dict[str, str]:
        """读取最近一次文件系统快路径命中的目标。"""
        variables = self.get_session_variables(user_id)
        value = variables.get("last_workspace_target", {})
        return value if isinstance(value, dict) else {}

    def set_last_workspace_target(self, user_id: str, path: str, target_type: str) -> None:
        """记录最近一次命中的文件或目录，便于承接“它/这个文件”这类后续追问。"""
        if target_type not in {"file", "dir"}:
            return

        with Session() as session:
            state = session.query(SessionState).filter_by(user_id=user_id).first()
            if not state:
                return

            try:
                variables = json.loads(state.variables or "{}")
            except json.JSONDecodeError:
                variables = {}

            variables["last_workspace_target"] = {
                "path": path,
                "type": target_type,
            }
            state.variables = json.dumps(variables, ensure_ascii=False)
            state.last_active = datetime.now()
            session.commit()

    def get_pending_workspace_resolution(self, user_id: str) -> Dict[str, Any]:
        """读取待确认的路径候选。"""
        variables = self.get_session_variables(user_id)
        value = variables.get("pending_workspace_resolution", {})
        return value if isinstance(value, dict) else {}

    def set_pending_workspace_resolution(
        self,
        user_id: str,
        *,
        action: str,
        original_input: str,
        candidates: List[Dict[str, Any]],
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """保存待确认的路径候选，等待用户回复“是/不是/序号”。"""
        with Session() as session:
            state = session.query(SessionState).filter_by(user_id=user_id).first()
            if not state:
                return

            try:
                variables = json.loads(state.variables or "{}")
            except json.JSONDecodeError:
                variables = {}

            variables["pending_workspace_resolution"] = {
                "action": action,
                "original_input": original_input,
                "candidates": candidates,
                "payload": payload or {},
            }
            state.variables = json.dumps(variables, ensure_ascii=False)
            state.last_active = datetime.now()
            session.commit()

    def clear_pending_workspace_resolution(self, user_id: str) -> None:
        """清除待确认的路径候选。"""
        with Session() as session:
            state = session.query(SessionState).filter_by(user_id=user_id).first()
            if not state:
                return

            try:
                variables = json.loads(state.variables or "{}")
            except json.JSONDecodeError:
                variables = {}

            if "pending_workspace_resolution" not in variables:
                return

            variables.pop("pending_workspace_resolution", None)
            state.variables = json.dumps(variables, ensure_ascii=False)
            state.last_active = datetime.now()
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
