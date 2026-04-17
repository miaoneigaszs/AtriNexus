"""
记忆管理器模块
三层记忆架构：
- 短期记忆（最近 N 轮对话，数据库 JSON Blob）
- 中期记忆（向量检索）— 语义相关的历史对话摘要
- 核心记忆（LLM 自动摘要的永久关键信息，数据库）
"""

import logging
import os
import hashlib
import asyncio
from typing import List, Optional

from src.ai.embedding_service import EmbeddingService
from src.memory.memory_context import MemoryContextBuilder
from src.memory.memory_updates import MemoryUpdateCoordinator
from src.memory.memory_store import (
    append_short_memory_entry,
    build_context_from_short_memory,
    increment_memory_counter,
    load_core_memory,
    load_short_memory,
    reset_memory_counter,
    save_core_memory as persist_core_memory,
    save_short_memory as persist_short_memory,
)
from src.memory.memory_vector import MemoryVectorManager
from src.platform_core.vector_store import QdrantVectorStore, VectorCollection, VectorStore
from src.platform_core.async_utils import run_sync
from src.platform_core.metrics import Metrics, PROMETHEUS_AVAILABLE
from data.config import config

logger = logging.getLogger('wecom')

# 核心记忆两区标记
_ZONE_STABLE = '【稳定信息】'
_ZONE_RECENT = '【近期信息】'


class MemoryManager:
    """
    三层记忆管理器

    架构：
    1. 短期记忆（数据库）：最近 max_groups 轮对话原文
    2. 中期记忆（向量存储）：对话摘要的向量检索，按语义相关度召回
    3. 核心记忆（数据库）：LLM 定期摘要的永久关键信息（用户身份、偏好等）

    上下文构建优先级：核心记忆 > 语义相关中期记忆 > 最近短期记忆
    """

    def __init__(self, llm_service=None, vector_store: Optional[VectorStore] = None):
        """
        初始化记忆管理器

        Args:
            llm_service: LLMService 实例（用于核心记忆摘要和对话摘要）
            vector_store: 可选外部传入的向量存储实例，若无则内部创建 Qdrant 实例
        """
        self.llm_service = llm_service
        self._vector_store = vector_store

        # 项目根目录
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # 记忆提示词路径
        self._memory_prompt_path = os.path.join(self.root_dir, 'src', 'prompting', 'memory.md')


        # 初始化 Embedding 服务（使用独立的 embedding 配置）
        self._embedding_service = EmbeddingService()
        # 优先使用 embedding_settings，回退到 llm_settings
        if hasattr(config, 'embedding') and config.embedding and config.embedding.api_key:
            api_key = config.embedding.api_key
            base_url = config.embedding.base_url or 'https://api.siliconflow.cn/v1'
            logger.info(f"使用 embedding_settings 配置: base_url={base_url}")
        else:
            api_key = config.llm.api_key
            base_url = config.llm.base_url
            logger.warning("embedding_settings 未配置，回退使用 llm_settings（可能不支持 embedding API）")
        if api_key:
            self._embedding_service.configure(api_key=api_key, base_url=base_url)

        # 初始化向量存储
        self._init_vector_store()

        # 时间衰减配置（可从配置文件读取）
        if hasattr(config.behavior, 'memory_decay'):
            self._decay_rate = config.behavior.memory_decay.daily_decay_rate
            self._decay_min = config.behavior.memory_decay.min_weight
        else:
            self._decay_rate = 0.98
            self._decay_min = 0.5

        self.context_builder = MemoryContextBuilder(
            get_core_memory=self.get_core_memory,
            get_short_memory=self.get_short_memory,
            search_relevant_memories=self.search_relevant_memories,
            build_context_from_memory=self.build_context_from_memory,
        )
        self.vector_manager = MemoryVectorManager(
            logger=logger,
            vector_store=self._vector_store,
            vector_store_available=self._vector_store_available,
            decay_rate=self._decay_rate,
            decay_min=self._decay_min,
            get_collection=self._get_collection,
        )
        self.update_coordinator = MemoryUpdateCoordinator(
            logger=logger,
            llm_service=self.llm_service,
            memory_prompt_path=self._memory_prompt_path,
            zone_stable=_ZONE_STABLE,
            zone_recent=_ZONE_RECENT,
            get_short_memory=self.get_short_memory,
            build_context_from_memory=self.build_context_from_memory,
            get_core_memory=self.get_core_memory,
            save_core_memory=self.save_core_memory,
            add_to_vector_memory=self.add_to_vector_memory,
            extract_zone=self._extract_zone,
            increment_core_count=self._increment_core_count,
            reset_core_count=self._reset_core_count,
            increment_vector_count=self._increment_vector_count,
            reset_vector_count=self._reset_vector_count,
        )

        logger.info("MemoryManager 初始化完成（三层记忆架构）")

    def _init_vector_store(self):
        """初始化向量存储"""
        try:
            # 使用共享的 Embedding 服务
            if self._embedding_service.is_available():
                if self._vector_store and hasattr(self._vector_store, "set_embedding_function"):
                    self._vector_store.set_embedding_function(self._embedding_service.embedding_function)
                if not self._vector_store:
                    qdrant_url = os.getenv("ATRINEXUS_QDRANT_URL", "").strip() or None
                    qdrant_api_key = os.getenv("ATRINEXUS_QDRANT_API_KEY", "").strip() or None
                    qdrant_path = os.getenv(
                        "ATRINEXUS_QDRANT_PATH",
                        os.path.join(self.root_dir, "data", "vectordb_qdrant"),
                    )
                    if qdrant_url:
                        self._vector_store = QdrantVectorStore(
                            url=qdrant_url,
                            api_key=qdrant_api_key,
                            embedding_function=self._embedding_service.embedding_function,
                        )
                    else:
                        os.makedirs(qdrant_path, exist_ok=True)
                        self._vector_store = QdrantVectorStore(
                            path=qdrant_path,
                            embedding_function=self._embedding_service.embedding_function,
                        )
                self._embedding_fn = self._embedding_service.embedding_function
                self._vector_store_available = True
                logger.info("向量存储初始化成功")
            else:
                logger.warning("Embedding 服务未配置，向量检索功能不可用")
                self._embedding_fn = None
                self._vector_store_available = False

        except ImportError:
            logger.warning("向量存储依赖未安装，向量检索功能不可用。")
            self._vector_store = None
            self._embedding_fn = None
            self._vector_store_available = False
        except Exception as e:
            logger.error(f"向量存储初始化失败: {e}")
            self._vector_store = None
            self._embedding_fn = None
            self._vector_store_available = False

    def _get_collection(self, user_id: str, avatar_name: str) -> Optional[VectorCollection]:
        """获取用户 + 人设隔离的向量集合"""
        if not self._vector_store_available or not self._vector_store:
            return None
        # Collection 名称：字母数字下划线，3-63字符
        safe_name = f"mem_{hashlib.md5(f'{user_id}_{avatar_name}'.encode()).hexdigest()[:16]}"
        try:
            return self._vector_store.get_or_create_collection(
                name=safe_name,
                metadata={"user_id": user_id, "avatar_name": avatar_name}
            )
        except Exception as e:
            logger.error(f"获取向量集合失败: {e}")
            return None

    # ---------- 短期记忆 ----------

    def get_short_memory(self, user_id: str, avatar_name: str) -> list:
        """从数据库加载短期记忆"""
        return load_short_memory(user_id, avatar_name)

    def save_short_memory(self, user_id: str, avatar_name: str, memory: list):
        """保存短期记忆到数据库"""
        ok = persist_short_memory(user_id, avatar_name, memory)
        if not ok:
            return

        if PROMETHEUS_AVAILABLE and Metrics.memory_short_entries:
            Metrics.memory_short_entries.labels(
                user_id=user_id, avatar_name=avatar_name
            ).set(len(memory))

        if PROMETHEUS_AVAILABLE and Metrics.memory_operations_total:
            Metrics.memory_operations_total.labels(
                operation='save', memory_type='short'
            ).inc()

    # ---------- 中期记忆（向量检索）----------

    def add_to_vector_memory(self, user_id: str, avatar_name: str, summary: str):
        """将对话摘要写入向量库。"""
        self.vector_manager.add_summary(user_id, avatar_name, summary)

    def search_relevant_memories(self, user_id: str, avatar_name: str,
                                  query: str, top_k: int = 3) -> List[str]:
        """语义搜索相关记忆（支持时间衰减重排）。"""
        return self.vector_manager.search(user_id, avatar_name, query, top_k)

    # ---------- 向量记忆公共接口 ----------

    def get_vector_memories(self, user_id: str, avatar_name: str, limit: int = 20) -> dict:
        """获取用户的向量记忆（中期记忆）。"""
        return self.vector_manager.get_memories(user_id, avatar_name, limit)

    def get_vector_store_stats(self) -> List[dict]:
        """返回向量存储中的记忆集合统计。"""
        return self.vector_manager.get_store_stats()

    def delete_vector_memory(self, user_id: str, avatar_name: str, memory_id: str = None) -> bool:
        """删除向量记忆。"""
        return self.vector_manager.delete_memory(user_id, avatar_name, memory_id)

    # ---------- 核心记忆 ----------

    def get_core_memory(self, user_id: str, avatar_name: str) -> str:
        """从数据库加载核心记忆"""
        return load_core_memory(user_id, avatar_name)

    def save_core_memory(self, user_id: str, avatar_name: str, content: str):
        """保存核心记忆到 SQLite"""
        # 格式校验（非强制，仅 warning）
        if _ZONE_STABLE not in content or _ZONE_RECENT not in content:
            logger.warning(
                "[CoreMemory] 保存的核心记忆缺少双区标记，格式可能不正确："
                f"{content[:80]}..."
            )
        persist_core_memory(user_id, avatar_name, content)

    def _extract_zone(self, content: str, zone_name: str) -> str:
        """
        从双区格式核心记忆中提取指定区的内容。

        Args:
            content:   完整核心记忆文本
            zone_name: 区标记，如 '【稳定信息】' 或 '【近期信息】'

        Returns:
            该区的文本内容；若解析失败返回空字符串
        """
        if zone_name not in content:
            return ''
        after = content.split(zone_name, 1)[1]
        # 找到下一个区的开始位置（如果有的话）
        other_zone = _ZONE_RECENT if zone_name == _ZONE_STABLE else _ZONE_STABLE
        if other_zone in after:
            after = after.split(other_zone, 1)[0]
        return after.strip()

    # ---------- 对话计数（持久化）----------
    # 两套独立计数器：
    #   count        -> 核心记忆触发计数（每15轮重置）
    #   vector_count -> 向量记忆触发计数（每10轮重置）

    def _increment_core_count(self, user_id: str, avatar_name: str) -> int:
        """增加核心记忆计数并返回新值（每15轮触发）"""
        return increment_memory_counter(user_id, avatar_name, "count")

    def _reset_core_count(self, user_id: str, avatar_name: str):
        """重置核心记忆计数"""
        reset_memory_counter(user_id, avatar_name, "count")

    def _increment_vector_count(self, user_id: str, avatar_name: str) -> int:
        """增加向量记忆计数并返回新值（每10轮触发）"""
        return increment_memory_counter(user_id, avatar_name, "vector_count")

    def _reset_vector_count(self, user_id: str, avatar_name: str):
        """重置向量记忆计数"""
        reset_memory_counter(user_id, avatar_name, "vector_count")


    # ---------- 上下文构建（三层融合）----------

    def build_context_from_memory(self, short_memory: list) -> list:
        """将短期记忆转换为 LLM 上下文格式"""
        return build_context_from_short_memory(short_memory)

    def build_full_context(self, user_id: str, avatar_name: str,
                           current_message: str) -> dict:
        """构建包含三层记忆的完整上下文。"""
        return self.context_builder.build_full_context(user_id, avatar_name, current_message)

    # ---------- 对话后记忆更新 ----------

    def after_reply(self, user_id: str, avatar_name: str, user_msg: str, bot_reply: str):
        """
        回复后更新所有记忆层

        Args:
            user_id: 用户ID
            avatar_name: 人设名称
            user_msg: 用户消息
            bot_reply: 机器人回复
        """
        # 1. 更新短期记忆
        append_short_memory_entry(user_id, avatar_name, user_msg, bot_reply)
        short_memory = self.get_short_memory(user_id, avatar_name)

        # 2. 检查是否需要更新核心记忆 + 写入向量库
        self._update_memories_if_needed(user_id, avatar_name)

    def _update_memories_if_needed(self, user_id: str, avatar_name: str):
        """根据计数器触发核心记忆和向量记忆更新。"""
        self.update_coordinator.update_memories_if_needed(
            user_id,
            avatar_name,
            vector_store_available=self._vector_store_available,
        )

    def _do_update_vector_memory(self, user_id: str, avatar_name: str):
        """将本批对话摘要写入向量库（每10轮触发）。"""
        self.update_coordinator.update_vector_memory(
            user_id,
            avatar_name,
            vector_store_available=self._vector_store_available,
        )

    def _do_update_core_memory(self, user_id: str, avatar_name: str):
        """LLM重写核心记忆双区格式（每15轮触发）。"""
        self.update_coordinator.update_core_memory(user_id, avatar_name)

    # ========== 异步方法（性能优化）==========


    async def search_relevant_memories_async(
        self, 
        user_id: str, 
        avatar_name: str,
        query: str, 
        top_k: int = 3
    ) -> List[str]:
        """
        异步语义搜索相关记忆（不阻塞事件循环）
        
        使用线程池执行同步向量检索，适用于高并发场景。
        """
        return await run_sync(
            self.search_relevant_memories,
            user_id, avatar_name, query, top_k
        )

    async def build_full_context_async(
        self, 
        user_id: str, 
        avatar_name: str,
        current_message: str
    ) -> dict:
        """异步构建完整上下文。"""
        return await self.context_builder.build_full_context_async(
            user_id,
            avatar_name,
            current_message,
        )

    async def after_reply_async(
        self, 
        user_id: str, 
        avatar_name: str, 
        user_msg: str, 
        bot_reply: str
    ):
        """
        异步更新记忆（不阻塞主流程）
        
        适用于"即发即忘"场景，不等待写入完成。
        """
        asyncio.create_task(
            self._after_reply_background(user_id, avatar_name, user_msg, bot_reply)
        )

    async def _after_reply_background(
        self,
        user_id: str,
        avatar_name: str,
        user_msg: str,
        bot_reply: str,
    ) -> None:
        try:
            await run_sync(append_short_memory_entry, user_id, avatar_name, user_msg, bot_reply)
            await self._update_memories_if_needed_async(user_id, avatar_name)
        except Exception as e:
            logger.error(f"异步更新记忆失败: {e}", exc_info=True)

    async def _update_memories_if_needed_async(self, user_id: str, avatar_name: str) -> None:
        await self.update_coordinator.update_memories_if_needed_async(
            user_id,
            avatar_name,
            vector_store_available=self._vector_store_available,
        )

    async def _do_update_vector_memory_async(self, user_id: str, avatar_name: str) -> None:
        await self.update_coordinator.update_vector_memory_async(
            user_id,
            avatar_name,
            vector_store_available=self._vector_store_available,
        )

    async def _do_update_core_memory_async(self, user_id: str, avatar_name: str) -> None:
        await self.update_coordinator.update_core_memory_async(user_id, avatar_name)
