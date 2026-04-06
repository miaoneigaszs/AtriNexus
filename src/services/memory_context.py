from __future__ import annotations

import asyncio
from typing import Callable, List

from src.utils.async_utils import run_sync


_ACTION_HINTS = (
    "readme",
    "目录",
    "文件",
    "重命名",
    "改成",
    "改为",
    "追加",
    "执行",
    "git ",
    "知识库",
    "文档",
    "资料",
)

_MEMORY_HINTS = (
    "还记得",
    "记得",
    "记不记得",
    "之前",
    "上次",
    "刚才",
    "前面",
    "提过",
    "说过",
    "聊过",
    "以前",
    "喜欢",
    "不喜欢",
    "偏好",
    "习惯",
)


class MemoryContextBuilder:
    """封装记忆上下文组装，避免 MemoryManager 同时承担太多职责。"""

    def __init__(
        self,
        *,
        get_core_memory: Callable[[str, str], str],
        get_short_memory: Callable[[str, str], list],
        search_relevant_memories: Callable[[str, str, str, int], List[str]],
        build_context_from_memory: Callable[[list], list],
    ) -> None:
        self.get_core_memory = get_core_memory
        self.get_short_memory = get_short_memory
        self.search_relevant_memories = search_relevant_memories
        self.build_context_from_memory = build_context_from_memory

    def should_recall_vector_memories(self, current_message: str) -> bool:
        """只在明显需要跨轮回忆时才做向量记忆召回。"""
        normalized = (current_message or "").strip().lower()
        if len(normalized) < 4:
            return False

        if any(hint in normalized for hint in _ACTION_HINTS):
            return False

        return any(hint in normalized for hint in _MEMORY_HINTS)

    def build_full_context(self, user_id: str, avatar_name: str, current_message: str) -> dict:
        """同步构建完整上下文。"""
        core_memory = self.get_core_memory(user_id, avatar_name)

        relevant_memories: List[str] = []
        if self.should_recall_vector_memories(current_message):
            relevant_memories = self.search_relevant_memories(
                user_id,
                avatar_name,
                current_message,
                top_k=3,
            )

        short_memory = self.get_short_memory(user_id, avatar_name)
        previous_context = self.build_context_from_memory(short_memory)

        return {
            "core_memory": core_memory,
            "relevant_memories": relevant_memories,
            "previous_context": previous_context,
        }

    async def build_full_context_async(
        self,
        user_id: str,
        avatar_name: str,
        current_message: str,
    ) -> dict:
        """异步构建完整上下文。"""
        core_memory_task = run_sync(self.get_core_memory, user_id, avatar_name)
        relevant_memories_task = (
            run_sync(self.search_relevant_memories, user_id, avatar_name, current_message, 3)
            if self.should_recall_vector_memories(current_message)
            else asyncio.sleep(0, result=[])
        )
        short_memory_task = run_sync(self.get_short_memory, user_id, avatar_name)

        core_memory, relevant_memories, short_memory = await asyncio.gather(
            core_memory_task,
            relevant_memories_task,
            short_memory_task,
        )

        previous_context = self.build_context_from_memory(short_memory)
        return {
            "core_memory": core_memory,
            "relevant_memories": relevant_memories,
            "previous_context": previous_context,
        }
