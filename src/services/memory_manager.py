"""
记忆管理器模块（纯异步版本）
三层记忆架构：
- 短期记忆（最近 N 轮对话，SQLite JSON Blob）
- 中期记忆（向量检索，ChromaDB）— 语义相关的历史对话摘要
- 核心记忆（LLM 自动摘要的永久关键信息，SQLite）

所有方法均为异步，通过线程池执行同步 I/O 操作，不阻塞事件循环。
"""

import json
import logging
import os
import hashlib
import asyncio
from datetime import datetime
from typing import List, Optional

from src.services.database import Session, MemorySnapshot, ConversationCounter
from src.services.ai.embedding_service import EmbeddingService
from src.utils.async_utils import run_sync
from src.utils.metrics import Metrics, PROMETHEUS_AVAILABLE
from data.config import config

logger = logging.getLogger('wecom')

# 核心记忆两区标记
_ZONE_STABLE = '【稳定信息】'
_ZONE_RECENT = '【近期信息】'


class MemoryManager:
    """
    三层记忆管理器（纯异步版本）

    架构：
    1. 短期记忆（SQLite）：最近 max_groups 轮对话原文
    2. 中期记忆（ChromaDB）：对话摘要的向量检索，按语义相关度召回
    3. 核心记忆（SQLite）：LLM 定期摘要的永久关键信息（用户身份、偏好等）

    上下文构建优先级：核心记忆 > 语义相关中期记忆 > 最近短期记忆

    异步实现：
    - 所有数据库操作通过 run_sync 在线程池执行
    - build_full_context 使用 asyncio.gather 并行加载三层记忆
    """

    def __init__(self, llm_service=None, chroma_client=None):
        """
        初始化记忆管理器

        Args:
            llm_service: LLMService 实例（用于核心记忆摘要和对话摘要）
            chroma_client: 可选外部传入的 ChromaDB 单例，若无则内部创建
        """
        self.llm_service = llm_service
        self._chroma_client = chroma_client

        # 项目根目录
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # 记忆提示词路径
        self._memory_prompt_path = os.path.join(self.root_dir, 'src', 'base', 'memory.md')


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

        # 初始化 ChromaDB 向量存储（同步初始化，仅在启动时执行一次）
        self._init_vector_store()

        # 时间衰减配置（可从配置文件读取）
        if hasattr(config.behavior, 'memory_decay'):
            self._decay_rate = config.behavior.memory_decay.daily_decay_rate
            self._decay_min = config.behavior.memory_decay.min_weight
        else:
            self._decay_rate = 0.98
            self._decay_min = 0.5

        logger.info("MemoryManager 初始化完成（三层记忆架构 - 纯异步版本）")

    def _init_vector_store(self):
        """初始化 ChromaDB 向量存储（同步方法，仅在 __init__ 中调用）"""
        try:
            import chromadb
            from chromadb.config import Settings

            # 如果外部没有传，才新起一个 client
            if not self._chroma_client:
                vectordb_path = os.path.join(self.root_dir, 'data', 'vectordb')
                os.makedirs(vectordb_path, exist_ok=True)

                self._chroma_client = chromadb.PersistentClient(
                    path=vectordb_path,
                    settings=Settings(anonymized_telemetry=False)
                )

            # 使用共享的 Embedding 服务
            if self._embedding_service.is_available():
                self._embedding_fn = self._embedding_service.embedding_function
                self._vector_store_available = True
                logger.info("ChromaDB 向量存储初始化成功")
            else:
                logger.warning("Embedding 服务未配置，向量检索功能不可用")
                self._embedding_fn = None
                self._vector_store_available = False

        except ImportError:
            logger.warning("chromadb 未安装，向量检索功能不可用。请运行: pip install chromadb")
            self._chroma_client = None
            self._embedding_fn = None
            self._vector_store_available = False
        except Exception as e:
            logger.error(f"ChromaDB 初始化失败: {e}")
            self._chroma_client = None
            self._embedding_fn = None
            self._vector_store_available = False

    def _get_collection(self, user_id: str, avatar_name: str):
        """获取用户 + 人设隔离的 ChromaDB Collection（同步方法，内部使用）"""
        if not self._vector_store_available:
            return None
        # Collection 名称：字母数字下划线，3-63字符
        safe_name = f"mem_{hashlib.md5(f'{user_id}_{avatar_name}'.encode()).hexdigest()[:16]}"
        try:
            return self._chroma_client.get_or_create_collection(
                name=safe_name,
                embedding_function=self._embedding_fn,
                metadata={"user_id": user_id, "avatar_name": avatar_name}
            )
        except Exception as e:
            logger.error(f"获取 ChromaDB Collection 失败: {e}")
            return None

    # ========== 短期记忆（异步）==========

    async def get_short_memory(self, user_id: str, avatar_name: str) -> list:
        """从 SQLite 加载短期记忆（异步）"""
        def _load():
            session = Session()
            try:
                snapshot = session.query(MemorySnapshot).filter_by(
                    user_id=user_id, avatar_name=avatar_name, memory_type='short'
                ).first()
                if snapshot and snapshot.content:
                    return json.loads(snapshot.content)
                return []
            except Exception as e:
                logger.error(f"加载短期记忆失败: {e}")
                return []
            finally:
                session.close()

        return await run_sync(_load)

    async def save_short_memory(self, user_id: str, avatar_name: str, memory: list):
        """保存短期记忆到 SQLite（异步）"""
        def _save():
            session = Session()
            try:
                snapshot = session.query(MemorySnapshot).filter_by(
                    user_id=user_id, avatar_name=avatar_name, memory_type='short'
                ).first()
                if snapshot:
                    snapshot.content = json.dumps(memory, ensure_ascii=False)
                    snapshot.updated_at = datetime.now()
                else:
                    snapshot = MemorySnapshot(
                        user_id=user_id, avatar_name=avatar_name,
                        memory_type='short', content=json.dumps(memory, ensure_ascii=False)
                    )
                    session.add(snapshot)
                session.commit()
                
                # 更新 Prometheus 指标
                if PROMETHEUS_AVAILABLE and Metrics.memory_short_entries:
                    Metrics.memory_short_entries.labels(
                        user_id=user_id, avatar_name=avatar_name
                    ).set(len(memory))
                
                if PROMETHEUS_AVAILABLE and Metrics.memory_operations_total:
                    Metrics.memory_operations_total.labels(
                        operation='save', memory_type='short'
                    ).inc()
                    
            except Exception as e:
                logger.error(f"保存短期记忆失败: {e}")
                session.rollback()
            finally:
                session.close()

        await run_sync(_save)

    # ========== 中期记忆（ChromaDB 向量检索 - 异步）==========

    async def add_to_vector_memory(self, user_id: str, avatar_name: str, summary: str):
        """
        将对话摘要写入向量库（异步）

        Args:
            user_id: 用户ID
            avatar_name: 人设名称
            summary: 对话摘要文本
        """
        def _add():
            collection = self._get_collection(user_id, avatar_name)
            if not collection:
                return

            try:
                # 生成唯一 ID
                doc_id = f"mem_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hashlib.md5(summary.encode()).hexdigest()[:8]}"

                collection.add(
                    ids=[doc_id],
                    documents=[summary],
                    metadatas=[{
                        "user_id": user_id,
                        "avatar_name": avatar_name,
                        "timestamp": datetime.now().isoformat(),
                        "type": "conversation_summary"
                    }]
                )
                
                # 更新 Prometheus 指标
                if PROMETHEUS_AVAILABLE and Metrics.memory_vector_entries:
                    count = collection.count()
                    Metrics.memory_vector_entries.labels(
                        user_id=user_id, avatar_name=avatar_name
                    ).set(count)
                
                if PROMETHEUS_AVAILABLE and Metrics.memory_operations_total:
                    Metrics.memory_operations_total.labels(
                        operation='add', memory_type='vector'
                    ).inc()
                
                logger.info(f"对话摘要已写入向量库: {summary[:50]}...")
            except Exception as e:
                logger.error(f"写入向量库失败: {e}")

        await run_sync(_add)

    async def search_relevant_memories(
        self, 
        user_id: str, 
        avatar_name: str,
        query: str, 
        top_k: int = 3
    ) -> List[str]:
        """
        语义搜索相关记忆（支持时间衰减重排）- 异步版本

        Args:
            user_id: 用户ID
            avatar_name: 人设名称
            query: 搜索查询（当前用户消息）
            top_k: 返回最相关的 N 条记忆

        Returns:
            List[str]: 相关记忆文本列表
        """
        def _search():
            collection = self._get_collection(user_id, avatar_name)
            if not collection:
                return []

            try:
                # 检查 collection 是否有数据
                count = collection.count()
                if count == 0:
                    return []

                # 查询更多候选以便时间衰减重排 (取 top_k * 3 或至少 10 条)
                candidate_k = min(max(top_k * 3, 10), count)

                results = collection.query(
                    query_texts=[query],
                    n_results=candidate_k
                )

                if results and results['documents'] and results['documents'][0]:
                    memories = results['documents'][0]
                    metadatas = results['metadatas'][0] if results.get('metadatas') else [{}] * len(memories)
                    distances = results['distances'][0] if results.get('distances') else [0] * len(memories)

                    # 开始重排（时间衰减 + 相关度结合）
                    scored_memories = []
                    now = datetime.now()
                    for mem, meta, dist in zip(memories, metadatas, distances):
                        # 基准相似度：L2距离越小越好
                        sim_score = 1.0 / (1.0 + dist)

                        # 时间衰减：越久远的记忆权重越低（使用配置化参数）
                        time_str = meta.get('timestamp')
                        decay = 1.0
                        if time_str:
                            try:
                                mem_time = datetime.fromisoformat(time_str)
                                hours_diff = (now - mem_time).total_seconds() / 3600
                                # 使用配置的衰减率和最低权重
                                decay = max(self._decay_min, self._decay_rate ** (hours_diff / 24.0))
                            except Exception:
                                pass

                        # 综合得分
                        final_score = sim_score * decay
                        scored_memories.append((final_score, mem, dist, decay))

                    # 按综合得分降序排序
                    scored_memories.sort(key=lambda x: x[0], reverse=True)
                    
                    # 取前 top_k
                    top_memories = scored_memories[:top_k]

                    logger.info(f"向量检索: 查询='{query[:30]}...', 候选={len(memories)}条，返回={len(top_memories)}条")
                    for i, (score, mem, dist, decay) in enumerate(top_memories):
                        logger.debug(f"  [{i}] 综合={score:.3f} (距={dist:.3f}, 衰减={decay:.3f}): {mem[:50]}...")
                        
                    return [item[1] for item in top_memories]

                return []
            except Exception as e:
                logger.error(f"向量检索失败: {e}")
                return []

        return await run_sync(_search)

    # ---------- 向量记忆公共接口（异步）----------

    async def get_vector_memories(self, user_id: str, avatar_name: str, limit: int = 20) -> dict:
        """
        获取用户的向量记忆（中期记忆）- 异步版本

        Args:
            user_id: 用户ID
            avatar_name: 人设名称
            limit: 返回条数

        Returns:
            dict: {
                'collection': collection 对象或 None,
                'memories': [{'id': ..., 'content': ..., 'metadata': ...}],
                'total': 总数
            }
        """
        def _get():
            collection = self._get_collection(user_id, avatar_name)
            if not collection:
                return {'collection': None, 'memories': [], 'total': 0}

            try:
                results = collection.get(limit=limit, include=["documents", "metadatas"])

                memories = []
                if results and results['ids']:
                    for i, doc_id in enumerate(results['ids']):
                        memories.append({
                            "id": doc_id,
                            "content": results['documents'][i] if results.get('documents') else None,
                            "metadata": results['metadatas'][i] if results.get('metadatas') else {}
                        })

                return {
                    'collection': collection,
                    'memories': memories,
                    'total': collection.count()
                }
            except Exception as e:
                logger.error(f"获取向量记忆失败: {e}")
                return {'collection': None, 'memories': [], 'total': 0}

        return await run_sync(_get)

    async def delete_vector_memory(self, user_id: str, avatar_name: str, memory_id: str = None) -> bool:
        """
        删除向量记忆 - 异步版本

        Args:
            user_id: 用户ID
            avatar_name: 人设名称
            memory_id: 记忆ID，为空则清空全部

        Returns:
            bool: 是否成功
        """
        def _delete():
            collection = self._get_collection(user_id, avatar_name)
            if not collection:
                return True  # 空集合视为已删除

            try:
                if memory_id:
                    # 删除单条记忆
                    collection.delete(ids=[memory_id])
                    logger.info(f"向量记忆已删除: user={user_id}, id={memory_id}")
                else:
                    # 清空全部（删除整个 collection）
                    self._chroma_client.delete_collection(collection.name)
                    logger.info(f"向量记忆已清空: user={user_id}, avatar={avatar_name}")

                return True
            except Exception as e:
                logger.error(f"删除向量记忆失败: {e}")
                return False

        return await run_sync(_delete)

    # ========== 核心记忆（异步）==========

    async def get_core_memory(self, user_id: str, avatar_name: str) -> str:
        """从 SQLite 加载核心记忆（异步）"""
        def _load():
            session = Session()
            try:
                snapshot = session.query(MemorySnapshot).filter_by(
                    user_id=user_id, avatar_name=avatar_name, memory_type='core'
                ).first()
                if snapshot and snapshot.content:
                    data = json.loads(snapshot.content)
                    return data.get('content', '') if isinstance(data, dict) else str(data)
                return ''
            except Exception as e:
                logger.error(f"加载核心记忆失败: {e}")
                return ''
            finally:
                session.close()

        return await run_sync(_load)

    async def save_core_memory(self, user_id: str, avatar_name: str, content: str):
        """保存核心记忆到 SQLite（异步）"""
        # 格式校验（非强制，仅 warning）
        if _ZONE_STABLE not in content or _ZONE_RECENT not in content:
            logger.warning(
                "[CoreMemory] 保存的核心记忆缺少双区标记，格式可能不正确："
                f"{content[:80]}..."
            )

        def _save():
            session = Session()
            try:
                snapshot = session.query(MemorySnapshot).filter_by(
                    user_id=user_id, avatar_name=avatar_name, memory_type='core'
                ).first()
                data = json.dumps({
                    'content': content,
                    'updated_at': datetime.now().isoformat()
                }, ensure_ascii=False)
                if snapshot:
                    snapshot.content = data
                    snapshot.updated_at = datetime.now()
                else:
                    snapshot = MemorySnapshot(
                        user_id=user_id, avatar_name=avatar_name,
                        memory_type='core', content=data
                    )
                    session.add(snapshot)
                session.commit()
            except Exception as e:
                logger.error(f"保存核心记忆失败: {e}")
                session.rollback()
            finally:
                session.close()

        await run_sync(_save)

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

    # ========== 对话计数（持久化 - 异步）==========
    # 两套独立计数器：
    #   count        -> 核心记忆触发计数（每15轮重置）
    #   vector_count -> 向量记忆触发计数（每10轮重置）

    async def _get_or_create_counter(self, session, user_id: str, avatar_name: str):
        """获取或创建计数器记录（内部方法，需要在外部管理 session）"""
        counter = session.query(ConversationCounter).filter_by(
            user_id=user_id, avatar_name=avatar_name
        ).first()
        if not counter:
            counter = ConversationCounter(
                user_id=user_id,
                avatar_name=avatar_name,
                count=0,
                vector_count=0
            )
            session.add(counter)
            session.flush()
        # 兼容旧数据库：若 vector_count 列不存在，动态添加
        if not hasattr(counter, 'vector_count') or counter.vector_count is None:
            try:
                from sqlalchemy import text
                session.execute(text(
                    "ALTER TABLE conversation_counters ADD COLUMN vector_count INTEGER DEFAULT 0"
                ))
                session.commit()
                counter.vector_count = 0
            except Exception:
                counter.vector_count = 0
        return counter

    async def _increment_core_count(self, user_id: str, avatar_name: str) -> int:
        """增加核心记忆计数并返回新值（每15轮触发）- 异步"""
        def _inc():
            session = Session()
            try:
                counter = session.query(ConversationCounter).filter_by(
                    user_id=user_id, avatar_name=avatar_name
                ).first()
                if not counter:
                    counter = ConversationCounter(
                        user_id=user_id,
                        avatar_name=avatar_name,
                        count=0,
                        vector_count=0
                    )
                    session.add(counter)
                counter.count += 1
                session.commit()
                return counter.count
            except Exception as e:
                logger.error(f"更新核心记忆计数失败: {e}")
                session.rollback()
                return 0
            finally:
                session.close()

        return await run_sync(_inc)

    async def _reset_core_count(self, user_id: str, avatar_name: str):
        """重置核心记忆计数 - 异步"""
        def _reset():
            session = Session()
            try:
                counter = session.query(ConversationCounter).filter_by(
                    user_id=user_id, avatar_name=avatar_name
                ).first()
                if counter:
                    counter.count = 0
                    session.commit()
            except Exception as e:
                logger.error(f"重置核心记忆计数失败: {e}")
                session.rollback()
            finally:
                session.close()

        await run_sync(_reset)

    async def _increment_vector_count(self, user_id: str, avatar_name: str) -> int:
        """增加向量记忆计数并返回新值（每10轮触发）- 异步"""
        def _inc():
            session = Session()
            try:
                counter = session.query(ConversationCounter).filter_by(
                    user_id=user_id, avatar_name=avatar_name
                ).first()
                if not counter:
                    counter = ConversationCounter(
                        user_id=user_id,
                        avatar_name=avatar_name,
                        count=0,
                        vector_count=0
                    )
                    session.add(counter)
                counter.vector_count += 1
                session.commit()
                return counter.vector_count
            except Exception as e:
                logger.error(f"更新向量记忆计数失败: {e}")
                session.rollback()
                return 0
            finally:
                session.close()

        return await run_sync(_inc)

    async def _reset_vector_count(self, user_id: str, avatar_name: str):
        """重置向量记忆计数 - 异步"""
        def _reset():
            session = Session()
            try:
                counter = session.query(ConversationCounter).filter_by(
                    user_id=user_id, avatar_name=avatar_name
                ).first()
                if counter:
                    counter.vector_count = 0
                    session.commit()
            except Exception as e:
                logger.error(f"重置向量记忆计数失败: {e}")
                session.rollback()
            finally:
                session.close()

        await run_sync(_reset)


    # ========== 上下文构建（三层融合 - 异步）==========

    def _build_context_from_memory(self, short_memory: list) -> list:
        """将短期记忆转换为 LLM 上下文格式（纯 CPU 操作，无需异步）"""
        context = []
        max_groups = config.behavior.context.max_groups
        for conv in short_memory[-max_groups:]:
            if 'user' in conv:
                context.append({"role": "user", "content": conv["user"]})
            if 'bot' in conv:
                context.append({"role": "assistant", "content": conv["bot"]})
        return context

    async def build_full_context(
        self, 
        user_id: str, 
        avatar_name: str,
        current_message: str
    ) -> dict:
        """
        构建包含三层记忆的完整上下文（异步，并行优化）

        使用 asyncio.gather 并行加载三层记忆：
        - 核心记忆（SQLite）
        - 中期记忆（ChromaDB 向量检索）
        - 短期记忆（SQLite）
        
        理论上可降低上下文构建耗时约 50%-67%

        返回:
            dict: {
                "core_memory": str,          # 核心记忆文本
                "relevant_memories": list,   # 语义相关的中期记忆
                "previous_context": list,    # 最近对话历史（LLM message 格式）
            }
        """
        # 并行执行三个独立的 I/O 操作
        core_memory_task = self.get_core_memory(user_id, avatar_name)
        relevant_memories_task = self.search_relevant_memories(
            user_id, avatar_name, current_message, top_k=3
        )
        short_memory_task = self.get_short_memory(user_id, avatar_name)
        
        # 等待所有任务完成
        core_memory, relevant_memories, short_memory = await asyncio.gather(
            core_memory_task, relevant_memories_task, short_memory_task
        )
        
        # 构建上下文（CPU 操作，无需异步）
        previous_context = self._build_context_from_memory(short_memory)
        
        return {
            "core_memory": core_memory,
            "relevant_memories": relevant_memories,
            "previous_context": previous_context,
        }

    # ========== 对话后记忆更新（异步）==========

    async def after_reply(
        self, 
        user_id: str, 
        avatar_name: str, 
        user_msg: str, 
        bot_reply: str
    ):
        """
        回复后更新所有记忆层（异步）

        Args:
            user_id: 用户ID
            avatar_name: 人设名称
            user_msg: 用户消息
            bot_reply: 机器人回复
        """
        # 1. 更新短期记忆
        short_memory = await self.get_short_memory(user_id, avatar_name)
        short_memory.append({
            "user": user_msg,
            "bot": bot_reply,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        # 只保留最近的对话轮次
        max_memory = config.behavior.context.max_groups * 2
        if len(short_memory) > max_memory:
            short_memory = short_memory[-max_memory:]
        await self.save_short_memory(user_id, avatar_name, short_memory)

        # 2. 检查是否需要更新核心记忆 + 写入向量库
        await self._update_memories_if_needed(user_id, avatar_name)

    async def _update_memories_if_needed(self, user_id: str, avatar_name: str):
        """
        两套独立触发链（互不干扰，计数器分离）：
          每10轮 -> 对话摘要写入向量库（中期记忆）
          每15轮 -> LLM重写核心记忆双区（稳定区 + 近期区）
        步骤A（归档近期区到向量库）已移除，消除与中期记忆的重叠。
        """
        # ===== 向量记忆：每10轮 =====
        vector_count = await self._increment_vector_count(user_id, avatar_name)
        if vector_count >= 10:
            await self._reset_vector_count(user_id, avatar_name)
            await self._do_update_vector_memory(user_id, avatar_name)

        # ===== 核心记忆：每15轮 =====
        core_count = await self._increment_core_count(user_id, avatar_name)
        if core_count >= 15:
            await self._reset_core_count(user_id, avatar_name)
            await self._do_update_core_memory(user_id, avatar_name)

    async def _do_update_vector_memory(self, user_id: str, avatar_name: str):
        """将本批对话摘要写入向量库（每10轮触发）- 异步"""
        if not self._vector_store_available:
            return
        if not self.llm_service:
            logger.warning("[VectorMemory] 未注入 LLMService，跳过向量记忆更新")
            return
        try:
            short_memory = await self.get_short_memory(user_id, avatar_name)
            context = self._build_context_from_memory(short_memory)
            summary_messages = [
                {"role": "system", "content": (
                    "请用2-3句话，以第三人称简洁总结以下对话的主要内容和关键信息，"
                    "使用对话中出现的具体称呼（如人名、昵称），而不是泛称'用户'或'AI'。"
                    "只返回总结，不要任何解释。"
                )},
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)}
            ]
            summary = await self.llm_service.chat(summary_messages)
            if summary and not summary.strip().lower().startswith("error"):
                await self.add_to_vector_memory(user_id, avatar_name, summary)
                logger.info(f"[VectorMemory] 对话摘要已写入，user={user_id}: {summary[:50]}...")
            else:
                logger.error(f"[VectorMemory] LLM 生成对话摘要失败或返回错误内容: {summary}")
        except Exception as e:
            logger.error(f"[VectorMemory] 生成对话摘要失败: {e}")

    async def _do_update_core_memory(self, user_id: str, avatar_name: str):
        """LLM重写核心记忆双区格式（每15轮触发）- 异步"""
        if not self.llm_service:
            logger.warning("[CoreMemory] 未注入 LLMService，跳过核心记忆更新")
            return
        try:
            short_memory = await self.get_short_memory(user_id, avatar_name)
            context = self._build_context_from_memory(short_memory)
            existing_core = await self.get_core_memory(user_id, avatar_name)

            try:
                with open(self._memory_prompt_path, 'r', encoding='utf-8') as f:
                    memory_prompt = f.read()
            except FileNotFoundError:
                memory_prompt = (
                    f"请根据旧核心记忆和最新对话，按以下格式重写核心记忆：\n"
                    f"{_ZONE_STABLE}\n（稳定信息，200字内，只改变明确更新的内容）\n\n"
                    f"{_ZONE_RECENT}\n（近期信息，200字内，根据最新对话重新提炼）\n\n"
                    "直接输出双区格式文本，不要任何解释。"
                )

            messages = [
                {"role": "system", "content": memory_prompt},
                {"role": "user", "content": (
                    f"旧核心记忆：\n{existing_core}\n\n"
                    f"最新对话：\n{json.dumps(context, ensure_ascii=False)}"
                )}
            ]
            new_core = await self.llm_service.chat(messages)
            
            # 强化校验：确保不是错误信息，并且必须包含双区结构，才覆盖写入数据库
            if new_core and not new_core.strip().lower().startswith("error"):
                if _ZONE_STABLE in new_core and _ZONE_RECENT in new_core:
                    await self.save_core_memory(user_id, avatar_name, new_core)
                    stable = self._extract_zone(new_core, _ZONE_STABLE)
                    recent = self._extract_zone(new_core, _ZONE_RECENT)
                    logger.info(
                        f"[CoreMemory] 核心记忆已更新 "
                        f"稳定区={len(stable)}字 近期区={len(recent)}字"
                    )
                else:
                    logger.error(f"[CoreMemory] LLM 返回的内容缺少双区标记({_ZONE_STABLE}/{_ZONE_RECENT})，拒绝覆盖: {new_core[:100]}...")
            else:
                logger.error(f"[CoreMemory] LLM 调用失败或为空，拒绝覆盖核心记忆: {new_core}")
        except Exception as e:
            logger.error(f"[CoreMemory] 更新核心记忆失败: {e}", exc_info=True)
