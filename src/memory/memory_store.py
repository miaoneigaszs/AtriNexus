"""MemoryManager 使用的数据库存储与上下文辅助函数。"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Tuple

from data.config import config
from src.platform_core.database import ConversationCounter, MemorySnapshot
from src.platform_core.db_session import new_session

logger = logging.getLogger("wecom")


def load_short_memory(user_id: str, avatar_name: str) -> list:
    with new_session() as session:
        try:
            snapshot = session.query(MemorySnapshot).filter_by(
                user_id=user_id,
                avatar_name=avatar_name,
                memory_type="short",
            ).first()
            if snapshot and snapshot.content:
                return json.loads(snapshot.content)
            return []
        except Exception as exc:
            logger.error(f"加载短期记忆失败: {exc}")
            return []


def save_short_memory(user_id: str, avatar_name: str, memory: list) -> bool:
    with new_session() as session:
        try:
            snapshot = session.query(MemorySnapshot).filter_by(
                user_id=user_id,
                avatar_name=avatar_name,
                memory_type="short",
            ).first()
            content = json.dumps(memory, ensure_ascii=False)
            if snapshot:
                snapshot.content = content
                snapshot.updated_at = datetime.now()
            else:
                snapshot = MemorySnapshot(
                    user_id=user_id,
                    avatar_name=avatar_name,
                    memory_type="short",
                    content=content,
                )
                session.add(snapshot)
            session.commit()
            return True
        except Exception as exc:
            logger.error(f"保存短期记忆失败: {exc}")
            session.rollback()
            return False


def load_core_memory(user_id: str, avatar_name: str) -> str:
    with new_session() as session:
        try:
            snapshot = session.query(MemorySnapshot).filter_by(
                user_id=user_id,
                avatar_name=avatar_name,
                memory_type="core",
            ).first()
            if snapshot and snapshot.content:
                data = json.loads(snapshot.content)
                return data.get("content", "") if isinstance(data, dict) else str(data)
            return ""
        except Exception as exc:
            logger.error(f"加载核心记忆失败: {exc}")
            return ""


def save_core_memory(user_id: str, avatar_name: str, content: str) -> bool:
    with new_session() as session:
        try:
            snapshot = session.query(MemorySnapshot).filter_by(
                user_id=user_id,
                avatar_name=avatar_name,
                memory_type="core",
            ).first()
            data = json.dumps(
                {
                    "content": content,
                    "updated_at": datetime.now().isoformat(),
                },
                ensure_ascii=False,
            )
            if snapshot:
                snapshot.content = data
                snapshot.updated_at = datetime.now()
            else:
                snapshot = MemorySnapshot(
                    user_id=user_id,
                    avatar_name=avatar_name,
                    memory_type="core",
                    content=data,
                )
                session.add(snapshot)
            session.commit()
            return True
        except Exception as exc:
            logger.error(f"保存核心记忆失败: {exc}")
            session.rollback()
            return False


def increment_memory_counter(user_id: str, avatar_name: str, field: str) -> int:
    with new_session() as session:
        try:
            counter = _get_or_create_counter(session, user_id, avatar_name)
            current = getattr(counter, field, 0) or 0
            setattr(counter, field, current + 1)
            session.commit()
            return getattr(counter, field, 0) or 0
        except Exception as exc:
            logger.error(f"更新记忆计数失败({field}): {exc}")
            session.rollback()
            return 0


def reset_memory_counter(user_id: str, avatar_name: str, field: str) -> None:
    with new_session() as session:
        try:
            counter = _get_or_create_counter(session, user_id, avatar_name)
            setattr(counter, field, 0)
            session.commit()
        except Exception as exc:
            logger.error(f"重置记忆计数失败({field}): {exc}")
            session.rollback()


def append_short_memory_entry(user_id: str, avatar_name: str, user_msg: str, bot_reply: str) -> list:
    short_memory = load_short_memory(user_id, avatar_name)
    short_memory.append(
        {
            "user": user_msg,
            "bot": bot_reply,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    max_memory = config.behavior.context.max_groups * 2
    if len(short_memory) > max_memory:
        short_memory = short_memory[-max_memory:]
    save_short_memory(user_id, avatar_name, short_memory)
    return short_memory


def build_context_from_short_memory(short_memory: list) -> list:
    context = []
    max_groups = config.behavior.context.max_groups
    for conv in short_memory[-max_groups:]:
        if "user" in conv:
            context.append({"role": "user", "content": conv["user"]})
        if "bot" in conv:
            context.append({"role": "assistant", "content": conv["bot"]})
    return context


def get_memory_counters(user_id: str, avatar_name: str) -> Tuple[int, int]:
    with new_session() as session:
        try:
            counter = _get_or_create_counter(session, user_id, avatar_name)
            return (counter.count or 0, getattr(counter, "vector_count", 0) or 0)
        except Exception as exc:
            logger.error(f"获取记忆计数失败: {exc}")
            return (0, 0)


def _get_or_create_counter(session, user_id: str, avatar_name: str):
    counter = session.query(ConversationCounter).filter_by(
        user_id=user_id,
        avatar_name=avatar_name,
    ).first()
    if not counter:
        counter = ConversationCounter(
            user_id=user_id,
            avatar_name=avatar_name,
            count=0,
            vector_count=0,
        )
        session.add(counter)
        session.flush()
    return counter
