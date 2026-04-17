from __future__ import annotations

import json
import logging
from typing import Callable, Optional

from src.utils.async_utils import run_sync


class MemoryUpdateCoordinator:
    """封装记忆更新编排，避免 MemoryManager 同时承担所有更新细节。"""

    def __init__(
        self,
        *,
        logger: logging.Logger,
        llm_service,
        memory_prompt_path: str,
        zone_stable: str,
        zone_recent: str,
        get_short_memory: Callable[[str, str], list],
        build_context_from_memory: Callable[[list], list],
        get_core_memory: Callable[[str, str], str],
        save_core_memory: Callable[[str, str, str], None],
        add_to_vector_memory: Callable[[str, str, str], None],
        extract_zone: Callable[[str, str], str],
        increment_core_count: Callable[[str, str], int],
        reset_core_count: Callable[[str, str], None],
        increment_vector_count: Callable[[str, str], int],
        reset_vector_count: Callable[[str, str], None],
    ) -> None:
        self.logger = logger
        self.llm_service = llm_service
        self.memory_prompt_path = memory_prompt_path
        self.zone_stable = zone_stable
        self.zone_recent = zone_recent
        self.get_short_memory = get_short_memory
        self.build_context_from_memory = build_context_from_memory
        self.get_core_memory = get_core_memory
        self.save_core_memory = save_core_memory
        self.add_to_vector_memory = add_to_vector_memory
        self.extract_zone = extract_zone
        self.increment_core_count = increment_core_count
        self.reset_core_count = reset_core_count
        self.increment_vector_count = increment_vector_count
        self.reset_vector_count = reset_vector_count

    def update_memories_if_needed(
        self,
        user_id: str,
        avatar_name: str,
        *,
        vector_store_available: bool = True,
    ) -> None:
        """根据计数器触发核心记忆和向量记忆更新。"""
        vector_count = self.increment_vector_count(user_id, avatar_name)
        if vector_count >= 10:
            self.reset_vector_count(user_id, avatar_name)
            self.update_vector_memory(
                user_id,
                avatar_name,
                vector_store_available=vector_store_available,
            )

        core_count = self.increment_core_count(user_id, avatar_name)
        if core_count >= 15:
            self.reset_core_count(user_id, avatar_name)
            self.update_core_memory(user_id, avatar_name)

    async def update_memories_if_needed_async(
        self,
        user_id: str,
        avatar_name: str,
        *,
        vector_store_available: bool = True,
    ) -> None:
        """异步版本的触发链。"""
        vector_count = await run_sync(self.increment_vector_count, user_id, avatar_name)
        if vector_count >= 10:
            await run_sync(self.reset_vector_count, user_id, avatar_name)
            await self.update_vector_memory_async(
                user_id,
                avatar_name,
                vector_store_available=vector_store_available,
            )

        core_count = await run_sync(self.increment_core_count, user_id, avatar_name)
        if core_count >= 15:
            await run_sync(self.reset_core_count, user_id, avatar_name)
            await self.update_core_memory_async(user_id, avatar_name)

    def update_vector_memory(
        self,
        user_id: str,
        avatar_name: str,
        *,
        vector_store_available: bool = True,
    ) -> None:
        """同步更新向量记忆。"""
        if not vector_store_available:
            return
        if not self.llm_service:
            self.logger.warning("[VectorMemory] 未注入 LLMService，跳过向量记忆更新")
            return

        try:
            short_memory = self.get_short_memory(user_id, avatar_name)
            context = self.build_context_from_memory(short_memory)
            summary_messages = [
                {"role": "system", "content": (
                    "请用2-3句话，以第三人称简洁总结以下对话的主要内容和关键信息，"
                    "使用对话中出现的具体称呼（如人名、昵称），而不是泛称'用户'或'AI'。"
                    "只返回总结，不要任何解释。"
                )},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ]
            summary = self.llm_service.chat(summary_messages)
            if summary:
                self.add_to_vector_memory(user_id, avatar_name, summary)
                self.logger.info(f"[VectorMemory] 对话摘要已写入，user={user_id}: {summary[:50]}...")
        except Exception as e:
            self.logger.error(f"[VectorMemory] 生成对话摘要失败: {e}")

    def update_core_memory(self, user_id: str, avatar_name: str) -> None:
        """同步更新核心记忆。"""
        if not self.llm_service:
            self.logger.warning("[CoreMemory] 未注入 LLMService，跳过核心记忆更新")
            return

        try:
            short_memory = self.get_short_memory(user_id, avatar_name)
            context = self.build_context_from_memory(short_memory)
            existing_core = self.get_core_memory(user_id, avatar_name)
            memory_prompt = self._load_memory_prompt()

            messages = [
                {"role": "system", "content": memory_prompt},
                {"role": "user", "content": (
                    f"旧核心记忆：\n{existing_core}\n\n"
                    f"最新对话：\n{json.dumps(context, ensure_ascii=False)}"
                )},
            ]
            new_core = self.llm_service.chat(messages)
            if new_core:
                self.save_core_memory(user_id, avatar_name, new_core)
                stable = self.extract_zone(new_core, self.zone_stable)
                recent = self.extract_zone(new_core, self.zone_recent)
                self.logger.info(
                    f"[CoreMemory] 核心记忆已更新 稳定区={len(stable)}字 近期区={len(recent)}字"
                )
        except Exception as e:
            self.logger.error(f"[CoreMemory] 更新核心记忆失败: {e}", exc_info=True)

    async def update_vector_memory_async(
        self,
        user_id: str,
        avatar_name: str,
        vector_store_available: bool = True,
    ) -> None:
        """异步更新向量记忆。"""
        if not vector_store_available:
            return
        if not self.llm_service:
            self.logger.warning("[VectorMemory] 未注入 LLMService，跳过向量记忆更新")
            return

        try:
            short_memory = await run_sync(self.get_short_memory, user_id, avatar_name)
            context = self.build_context_from_memory(short_memory)
            summary_messages = [
                {"role": "system", "content": (
                    "请用2-3句话，以第三人称简洁总结以下对话的主要内容和关键信息，"
                    "使用对话中出现的具体称呼（如人名、昵称），而不是泛称'用户'或'AI'。"
                    "只返回总结，不要任何解释。"
                )},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ]
            summary = await self.llm_service.chat_async(summary_messages)
            if summary and summary != "抱歉，我暂时无法处理你的消息，请稍后再试。":
                await run_sync(self.add_to_vector_memory, user_id, avatar_name, summary)
                self.logger.info(f"[VectorMemory] 对话摘要已写入，user={user_id}: {summary[:50]}...")
        except Exception as e:
            self.logger.error(f"[VectorMemory] 异步生成对话摘要失败: {e}", exc_info=True)

    async def update_core_memory_async(self, user_id: str, avatar_name: str) -> None:
        """异步更新核心记忆。"""
        if not self.llm_service:
            self.logger.warning("[CoreMemory] 未注入 LLMService，跳过核心记忆更新")
            return

        try:
            short_memory = await run_sync(self.get_short_memory, user_id, avatar_name)
            context = self.build_context_from_memory(short_memory)
            existing_core = await run_sync(self.get_core_memory, user_id, avatar_name)
            memory_prompt = self._load_memory_prompt()

            messages = [
                {"role": "system", "content": memory_prompt},
                {"role": "user", "content": (
                    f"旧核心记忆：\n{existing_core}\n\n"
                    f"最新对话：\n{json.dumps(context, ensure_ascii=False)}"
                )},
            ]
            new_core = await self.llm_service.chat_async(messages)
            if new_core and new_core != "抱歉，我暂时无法处理你的消息，请稍后再试。":
                await run_sync(self.save_core_memory, user_id, avatar_name, new_core)
                stable = self.extract_zone(new_core, self.zone_stable)
                recent = self.extract_zone(new_core, self.zone_recent)
                self.logger.info(
                    f"[CoreMemory] 核心记忆已更新 稳定区={len(stable)}字 近期区={len(recent)}字"
                )
        except Exception as e:
            self.logger.error(f"[CoreMemory] 异步更新核心记忆失败: {e}", exc_info=True)

    def _load_memory_prompt(self) -> str:
        """读取核心记忆重写提示词，缺失时回退到内置模板。"""
        try:
            with open(self.memory_prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return (
                "请根据旧核心记忆和最新对话，按以下格式重写核心记忆：\n"
                f"{self.zone_stable}\n（稳定信息，200字内，只改变明确更新的内容）\n\n"
                f"{self.zone_recent}\n（近期信息，200字内，根据最新对话重新提炼）\n\n"
                "直接输出双区格式文本，不要任何解释。"
            )
