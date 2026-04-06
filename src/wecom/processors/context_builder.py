"""
上下文构建器
负责构建主聊天链路所需的记忆与模式上下文
"""

import logging
from typing import Dict, Any

from src.services.memory_manager import MemoryManager
from src.services.prompt_manager import PromptManager
from src.services.session_service import SessionService
from data.config import config

logger = logging.getLogger('wecom')


class ContextBuilder:
    """上下文构建器"""

    MAX_RELEVANT_MEMORY_ITEMS = 2
    MAX_RELEVANT_MEMORY_ITEM_CHARS = 180

    def __init__(self, memory_manager: MemoryManager, session_service: SessionService, root_dir: str):
        """
        初始化上下文构建器

        Args:
            memory_manager: 记忆管理器
            session_service: 会话服务
            root_dir: 项目根目录
        """
        self.memory = memory_manager
        self.session_service = session_service
        self.root_dir = root_dir
        self.avatar_dir = config.behavior.context.avatar_dir
        self.prompt_manager = PromptManager(root_dir)

    def build_search_context(self, user_id: str, content: str) -> Dict[str, Any]:
        """
        构建检索上下文

        Args:
            user_id: 用户ID
            content: 消息内容

        Returns:
            Dict: 包含会话状态、人设、模式、记忆等上下文信息
        """
        state = self.session_service.get_session(user_id)
        avatar_name = state.avatar_name or 'ATRI'
        current_mode = self._resolve_mode(user_id, content, state.mode or 'work')
        mem_ctx = self._build_memory_context(user_id, avatar_name, content)

        return {
            "state": state,
            "avatar_name": avatar_name,
            "current_mode": current_mode,
            "mem_ctx": mem_ctx,
            "previous_context": mem_ctx["previous_context"],
        }

    def _resolve_mode(self, user_id: str, content: str, current_mode: str) -> str:
        if not self._check_companion_trigger(content):
            return current_mode

        self.session_service.update_session_mode(user_id, 'companion')
        logger.info(f"用户 {user_id} 切换至陪伴模式")
        return 'companion'

    def _build_memory_context(self, user_id: str, avatar_name: str, content: str) -> Dict[str, Any]:
        return self.memory.build_full_context(user_id, avatar_name, content)

    def build_merged_memory_context(self, mem_ctx: Dict[str, Any]) -> str:
        """
        构建合并后的记忆上下文

        Args:
            mem_ctx: 记忆上下文

        Returns:
            str: 合并后的记忆文本
        """
        core_memory = mem_ctx["core_memory"]

        if mem_ctx["relevant_memories"]:
            trimmed_memories = [
                self._truncate_text(m, self.MAX_RELEVANT_MEMORY_ITEM_CHARS)
                for m in mem_ctx["relevant_memories"][: self.MAX_RELEVANT_MEMORY_ITEMS]
            ]
            relevant_text = "\n".join(f"- {m}" for m in trimmed_memories)
            if core_memory:
                core_memory = f"{core_memory}\n\n【相关历史对话记忆】\n{relevant_text}"
            else:
                core_memory = f"【相关历史对话记忆】\n{relevant_text}"

        return core_memory

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        normalized = (text or "").strip()
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 17].rstrip() + " [内容已截断]"

    def build_system_prompt(self, avatar_name: str, current_mode: str) -> str:
        """构建当前轮的模式提示词。"""
        return self.prompt_manager.build_persona_prompt(avatar_name, current_mode)

    def _check_companion_trigger(self, content: str) -> bool:
        """检查是否触发陪伴模式"""
        triggers = config.companion_mode.triggers
        for trigger in triggers:
            if trigger.lower() in content.lower():
                return True
        return False

