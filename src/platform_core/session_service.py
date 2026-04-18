"""
会话状态管理服务
统一管理用户会话状态
"""

import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.agent_runtime.tool_profiles import default_tool_profile_for_mode
from src.platform_core.database import SessionState
from src.platform_core.db_session import new_session

logger = logging.getLogger('wecom')

_WORKSPACE_BROWSER_STATE_KEY = "workspace_browser_state"


class SessionService:
    """会话状态管理服务"""

    def __init__(self):
        """初始化会话服务。"""

    # ---------- 用户会话状态管理 ----------

    def get_session(self, user_id: str) -> SessionState:
        """获取或创建用户会话状态"""
        with new_session(expire_on_commit=False) as session:
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

    def _parse_variables(self, raw: str, user_id: str) -> Dict[str, Any]:
        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            logger.warning("用户 %s 的 session variables 不是合法 JSON，已忽略", user_id)
            return {}
        return data if isinstance(data, dict) else {}

    def _mutate_session_variables(
        self,
        user_id: str,
        mutator: Callable[[Dict[str, Any]], bool],
    ) -> None:
        with new_session() as session:
            state = session.query(SessionState).filter_by(user_id=user_id).first()
            if not state:
                return

            variables = self._parse_variables(state.variables or "{}", user_id)
            if not mutator(variables):
                return

            state.variables = json.dumps(variables, ensure_ascii=False)
            state.last_active = datetime.now()
            session.commit()

    def _normalize_workspace_browser_state(self, value: Any) -> Dict[str, Dict[str, Any]]:
        data = value if isinstance(value, dict) else {}
        focus = data.get("focus") if isinstance(data.get("focus"), dict) else {}
        last_action = data.get("last_action") if isinstance(data.get("last_action"), dict) else {}
        pending = data.get("pending") if isinstance(data.get("pending"), dict) else {}
        return {
            "focus": dict(focus),
            "last_action": dict(last_action),
            "pending": dict(pending),
        }

    def update_session_variables(self, user_id: str, variables: dict):
        """更新用户会话上下文变量"""
        with new_session() as session:
            state = session.query(SessionState).filter_by(user_id=user_id).first()
            if state:
                state.variables = json.dumps(variables, ensure_ascii=False)
                session.commit()

    def get_session_variables(self, user_id: str) -> Dict[str, Any]:
        """读取会话扩展变量。"""
        state = self.get_session(user_id)
        return self._parse_variables(state.variables or "{}", user_id)

    def get_tool_profile(self, user_id: str) -> str:
        """读取当前会话绑定的工具 profile。"""
        state = self.get_session(user_id)
        variables = self._parse_variables(state.variables or "{}", user_id)

        value = str(variables.get("tool_profile", "")).strip()
        if value:
            return value
        return default_tool_profile_for_mode(state.mode)

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

        def mutator(variables: Dict[str, Any]) -> bool:
            variables["tool_profile"] = tool_profile
            return True

        self._mutate_session_variables(user_id, mutator)

    def get_workspace_browser_state(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """读取结构化的 workspace 浏览状态。"""
        variables = self.get_session_variables(user_id)
        return self._normalize_workspace_browser_state(variables.get(_WORKSPACE_BROWSER_STATE_KEY, {}))

    def set_workspace_browser_state(self, user_id: str, state: Dict[str, Any]) -> None:
        """覆盖结构化的 workspace 浏览状态。"""
        normalized = self._normalize_workspace_browser_state(state)

        def mutator(variables: Dict[str, Any]) -> bool:
            variables[_WORKSPACE_BROWSER_STATE_KEY] = normalized
            return True

        self._mutate_session_variables(user_id, mutator)

    def update_workspace_browser_state(
        self,
        user_id: str,
        *,
        focus: Optional[Dict[str, Any]] = None,
        last_action: Optional[Dict[str, Any]] = None,
        pending: Optional[Dict[str, Any]] = None,
    ) -> None:
        """局部更新 workspace 浏览状态。"""

        def mutator(variables: Dict[str, Any]) -> bool:
            browser_state = self._normalize_workspace_browser_state(
                variables.get(_WORKSPACE_BROWSER_STATE_KEY, {})
            )
            if focus is not None:
                browser_state["focus"] = dict(focus)
            if last_action is not None:
                browser_state["last_action"] = dict(last_action)
            if pending is not None:
                browser_state["pending"] = dict(pending)
            variables[_WORKSPACE_BROWSER_STATE_KEY] = browser_state
            return True

        self._mutate_session_variables(user_id, mutator)

    def set_workspace_browser_focus(
        self,
        user_id: str,
        *,
        path: str,
        target_type: str,
        intent: str = "",
        query: str = "",
        line_position: str = "",
    ) -> None:
        """记录当前 workspace 浏览焦点，并兼容旧的 last_workspace_target。"""
        if target_type not in {"file", "dir"}:
            return

        def mutator(variables: Dict[str, Any]) -> bool:
            browser_state = self._normalize_workspace_browser_state(
                variables.get(_WORKSPACE_BROWSER_STATE_KEY, {})
            )
            browser_state["focus"] = {"path": path, "type": target_type}
            if intent:
                browser_state["last_action"] = {
                    "intent": intent,
                    "path": path,
                    "query": query,
                    "line_position": line_position,
                }
            variables[_WORKSPACE_BROWSER_STATE_KEY] = browser_state
            variables["last_workspace_target"] = {"path": path, "type": target_type}
            return True

        self._mutate_session_variables(user_id, mutator)

    def get_last_workspace_target(self, user_id: str) -> Dict[str, str]:
        """读取最近一次文件系统快路径命中的目标。"""
        browser_state = self.get_workspace_browser_state(user_id)
        focus = browser_state.get("focus", {})
        path = str(focus.get("path", "")).strip()
        target_type = str(focus.get("type", "")).strip()
        if path and target_type in {"file", "dir"}:
            return {"path": path, "type": target_type}

        variables = self.get_session_variables(user_id)
        value = variables.get("last_workspace_target", {})
        return value if isinstance(value, dict) else {}

    def set_last_workspace_target(self, user_id: str, path: str, target_type: str) -> None:
        """记录最近一次命中的文件或目录，便于承接后续追问。"""
        self.set_workspace_browser_focus(
            user_id,
            path=path,
            target_type=target_type,
        )

    def get_workspace_browser_pending(self, user_id: str) -> Dict[str, Any]:
        """读取结构化的 workspace 浏览待处理状态。"""
        browser_state = self.get_workspace_browser_state(user_id)
        pending = browser_state.get("pending", {})
        return pending if isinstance(pending, dict) else {}

    def set_workspace_browser_pending(self, user_id: str, pending: Dict[str, Any]) -> None:
        """保存结构化的 workspace 浏览待处理状态。"""
        normalized_pending = dict(pending) if isinstance(pending, dict) else {}

        def mutator(variables: Dict[str, Any]) -> bool:
            browser_state = self._normalize_workspace_browser_state(
                variables.get(_WORKSPACE_BROWSER_STATE_KEY, {})
            )
            browser_state["pending"] = normalized_pending
            variables[_WORKSPACE_BROWSER_STATE_KEY] = browser_state
            return True

        self._mutate_session_variables(user_id, mutator)

    def clear_workspace_browser_pending(self, user_id: str) -> None:
        """清空结构化的 workspace 浏览待处理状态。"""

        def mutator(variables: Dict[str, Any]) -> bool:
            browser_state = self._normalize_workspace_browser_state(
                variables.get(_WORKSPACE_BROWSER_STATE_KEY, {})
            )
            if not browser_state.get("pending"):
                return False
            browser_state["pending"] = {}
            variables[_WORKSPACE_BROWSER_STATE_KEY] = browser_state
            return True

        self._mutate_session_variables(user_id, mutator)

    def get_pending_workspace_resolution(self, user_id: str) -> Dict[str, Any]:
        """读取待确认的路径候选。"""
        pending = self.get_workspace_browser_pending(user_id)
        if pending:
            return {
                "action": str(pending.get("action", "")).strip(),
                "original_input": str(pending.get("original_input", "")).strip(),
                "candidates": pending.get("candidates", []),
                "payload": pending.get("payload", {}),
            }

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
        pending = {
            "kind": "path_resolution",
            "action": action,
            "original_input": original_input,
            "candidates": candidates,
            "payload": payload or {},
        }

        def mutator(variables: Dict[str, Any]) -> bool:
            browser_state = self._normalize_workspace_browser_state(
                variables.get(_WORKSPACE_BROWSER_STATE_KEY, {})
            )
            browser_state["pending"] = pending
            variables[_WORKSPACE_BROWSER_STATE_KEY] = browser_state
            variables["pending_workspace_resolution"] = {
                "action": action,
                "original_input": original_input,
                "candidates": candidates,
                "payload": payload or {},
            }
            return True

        self._mutate_session_variables(user_id, mutator)

    def clear_pending_workspace_resolution(self, user_id: str) -> None:
        """清除待确认的路径候选。"""

        def mutator(variables: Dict[str, Any]) -> bool:
            changed = False
            browser_state = self._normalize_workspace_browser_state(
                variables.get(_WORKSPACE_BROWSER_STATE_KEY, {})
            )
            if browser_state.get("pending"):
                browser_state["pending"] = {}
                variables[_WORKSPACE_BROWSER_STATE_KEY] = browser_state
                changed = True
            if "pending_workspace_resolution" in variables:
                variables.pop("pending_workspace_resolution", None)
                changed = True
            return changed

        self._mutate_session_variables(user_id, mutator)

    def update_session_mode(self, user_id: str, mode: str):
        """更新用户模式"""
        with new_session() as session:
            state = session.query(SessionState).filter_by(user_id=user_id).first()
            if state:
                state.mode = mode
                state.last_active = datetime.now()
                session.commit()
                logger.info(f"用户 {user_id} 模式切换为 {mode}")
