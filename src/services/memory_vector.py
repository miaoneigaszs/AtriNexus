from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from typing import Callable, List, Optional

from src.services.vector_store import VectorCollection, VectorStore
from src.utils.metrics import Metrics, PROMETHEUS_AVAILABLE


class MemoryVectorManager:
    """封装向量记忆相关操作，减轻 MemoryManager 主文件压力。"""

    def __init__(
        self,
        *,
        logger: logging.Logger,
        vector_store: Optional[VectorStore],
        vector_store_available: bool,
        decay_rate: float,
        decay_min: float,
        get_collection: Callable[[str, str], Optional[VectorCollection]],
    ) -> None:
        self.logger = logger
        self.vector_store = vector_store
        self.vector_store_available = vector_store_available
        self.decay_rate = decay_rate
        self.decay_min = decay_min
        self.get_collection = get_collection

    def add_summary(self, user_id: str, avatar_name: str, summary: str) -> None:
        """将一段对话摘要写入向量记忆。"""
        collection = self.get_collection(user_id, avatar_name)
        if not collection:
            return

        try:
            doc_id = (
                f"mem_{datetime.now().strftime('%Y%m%d%H%M%S')}_"
                f"{hashlib.md5(summary.encode()).hexdigest()[:8]}"
            )
            collection.add(
                ids=[doc_id],
                documents=[summary],
                metadatas=[{
                    "user_id": user_id,
                    "avatar_name": avatar_name,
                    "timestamp": datetime.now().isoformat(),
                    "type": "conversation_summary",
                }],
            )

            if PROMETHEUS_AVAILABLE and Metrics.memory_vector_entries:
                Metrics.memory_vector_entries.labels(
                    user_id=user_id,
                    avatar_name=avatar_name,
                ).set(collection.count())

            if PROMETHEUS_AVAILABLE and Metrics.memory_operations_total:
                Metrics.memory_operations_total.labels(
                    operation="add",
                    memory_type="vector",
                ).inc()

            self.logger.info(f"对话摘要已写入向量库: {summary[:50]}...")
        except Exception as e:
            self.logger.error(f"写入向量库失败: {e}")

    def search(self, user_id: str, avatar_name: str, query: str, top_k: int = 3) -> List[str]:
        """语义搜索相关记忆，并做时间衰减重排。"""
        collection = self.get_collection(user_id, avatar_name)
        if not collection:
            return []

        try:
            count = collection.count()
            if count == 0:
                return []

            candidate_k = min(max(top_k * 3, 10), count)
            results = collection.query(query_texts=[query], n_results=candidate_k)

            if not results or not results.get("documents") or not results["documents"][0]:
                return []

            memories = results["documents"][0]
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(memories)
            distances = results["distances"][0] if results.get("distances") else [0] * len(memories)

            scored_memories = []
            now = datetime.now()
            for mem, meta, dist in zip(memories, metadatas, distances):
                sim_score = 1.0 / (1.0 + dist)
                decay = 1.0
                time_str = meta.get("timestamp")
                if time_str:
                    try:
                        mem_time = datetime.fromisoformat(time_str)
                        hours_diff = (now - mem_time).total_seconds() / 3600
                        decay = max(self.decay_min, self.decay_rate ** (hours_diff / 24.0))
                    except Exception:
                        pass

                scored_memories.append((sim_score * decay, mem, dist, decay))

            scored_memories.sort(key=lambda item: item[0], reverse=True)
            top_memories = scored_memories[:top_k]

            self.logger.info(
                f"向量检索: 查询='{query[:30]}...', 候选={len(memories)}条，返回={len(top_memories)}条"
            )
            for index, (score, mem, dist, decay) in enumerate(top_memories):
                self.logger.debug(
                    f"  [{index}] 综合={score:.3f} (距={dist:.3f}, 衰减={decay:.3f}): {mem[:50]}..."
                )

            return [item[1] for item in top_memories]
        except Exception as e:
            self.logger.error(f"向量检索失败: {e}")
            return []

    def get_memories(self, user_id: str, avatar_name: str, limit: int = 20) -> dict:
        """获取向量记忆列表。"""
        collection = self.get_collection(user_id, avatar_name)
        if not collection:
            return {"collection": None, "memories": [], "total": 0}

        try:
            results = collection.get(limit=limit, include=["documents", "metadatas"])
            memories = []
            if results and results["ids"]:
                for index, doc_id in enumerate(results["ids"]):
                    memories.append({
                        "id": doc_id,
                        "content": results["documents"][index] if results.get("documents") else None,
                        "metadata": results["metadatas"][index] if results.get("metadatas") else {},
                    })

            return {
                "collection": collection,
                "memories": memories,
                "total": collection.count(),
            }
        except Exception as e:
            self.logger.error(f"获取向量记忆失败: {e}")
            return {"collection": None, "memories": [], "total": 0}

    def get_store_stats(self) -> List[dict]:
        """返回向量存储中的记忆集合统计。"""
        if not self.vector_store_available or not self.vector_store:
            return []

        client = getattr(self.vector_store, "_client", None)
        if client is None or not hasattr(client, "get_collections"):
            return []

        try:
            collections = client.get_collections().collections
            stats: List[dict] = []
            for collection in collections:
                name = getattr(collection, "name", "")
                if not str(name).startswith("mem_"):
                    continue
                qdrant_collection = self.vector_store.get_or_create_collection(name)
                stats.append({
                    "name": name,
                    "count": qdrant_collection.count(),
                })
            return stats
        except Exception as e:
            self.logger.error(f"获取向量存储统计失败: {e}")
            return []

    def delete_memory(self, user_id: str, avatar_name: str, memory_id: str | None = None) -> bool:
        """删除单条或整组向量记忆。"""
        collection = self.get_collection(user_id, avatar_name)
        if not collection:
            return True

        try:
            if memory_id:
                collection.delete(ids=[memory_id])
                self.logger.info(f"向量记忆已删除: user={user_id}, id={memory_id}")
            else:
                if self.vector_store:
                    self.vector_store.delete_collection(collection.name)
                self.logger.info(f"向量记忆已清空: user={user_id}, avatar={avatar_name}")
            return True
        except Exception as e:
            self.logger.error(f"删除向量记忆失败: {e}")
            return False
