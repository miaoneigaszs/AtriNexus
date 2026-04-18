"""Per-user todo state, used by the `todo` agent tool + system prompt snapshot.

存储为 in-memory dict，随进程生命周期存活，不持久化 —— 参考 hermes-agent
`tools/todo_tool.py`：todo 的角色是"把一轮对话里要走的多步骤记在 agent 看得见
的地方"，会话重启即清空是预期行为。

每项 `{id, content, status}`，status ∈ {pending, in_progress, completed, cancelled}。
通过模块级 `todo_store` 单例访问。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional


VALID_STATUSES = ("pending", "in_progress", "completed", "cancelled")

_STATUS_LABELS = {
    "pending": "未开始",
    "in_progress": "进行中",
    "completed": "已完成",
    "cancelled": "已取消",
}


@dataclass(frozen=True)
class TodoItem:
    id: str
    content: str
    status: str

    @classmethod
    def from_raw(cls, raw: object) -> Optional["TodoItem"]:
        if not isinstance(raw, dict):
            return None
        item_id = str(raw.get("id", "")).strip()
        content = str(raw.get("content", "")).strip()
        status = str(raw.get("status", "pending")).strip() or "pending"
        if not item_id or not content:
            return None
        if status not in VALID_STATUSES:
            status = "pending"
        return cls(id=item_id, content=content, status=status)

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


class TodoStore:
    """Per-user todo list，进程内状态。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: Dict[str, List[TodoItem]] = {}

    def get(self, user_id: str) -> List[TodoItem]:
        with self._lock:
            return list(self._items.get(user_id, []))

    def set(self, user_id: str, items: Iterable[TodoItem]) -> List[TodoItem]:
        cleaned = _dedupe_by_id(items)
        with self._lock:
            self._items[user_id] = cleaned
            return list(cleaned)

    def merge(self, user_id: str, updates: Iterable[TodoItem]) -> List[TodoItem]:
        """Upsert by id：已有 id 更新 content/status，新 id 追加，其余保留。"""
        update_list = list(updates)
        with self._lock:
            existing = list(self._items.get(user_id, []))
            by_id: Dict[str, TodoItem] = {item.id: item for item in existing}
            order: List[str] = [item.id for item in existing]
            for item in update_list:
                if item.id in by_id:
                    by_id[item.id] = item
                else:
                    by_id[item.id] = item
                    order.append(item.id)
            merged = [by_id[item_id] for item_id in order]
            self._items[user_id] = merged
            return list(merged)

    def clear(self, user_id: str) -> None:
        with self._lock:
            self._items.pop(user_id, None)

    def render(self, user_id: str) -> str:
        items = self.get(user_id)
        if not items:
            return ""
        lines = []
        for item in items:
            label = _STATUS_LABELS.get(item.status, item.status)
            lines.append(f"- [{label}] ({item.id}) {item.content}")
        return "\n".join(lines)


def _dedupe_by_id(items: Iterable[TodoItem]) -> List[TodoItem]:
    seen: Dict[str, TodoItem] = {}
    order: List[str] = []
    for item in items:
        if item.id not in seen:
            order.append(item.id)
        seen[item.id] = item
    return [seen[item_id] for item_id in order]


# 模块级单例 —— 和 user_runtime 一样的全局访问方式。
todo_store = TodoStore()
