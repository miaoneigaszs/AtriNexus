"""Clarify tool 的跨层信号。

`clarify` 工具 handler 调用 `mark_clarify(...)` 把澄清问题写进 contextvar；
agent_loop 在处理完一轮 tool call 后读取，一旦拿到非空问题，立即返回
`LoopResult(text=question, stop_reason="clarify", ...)` 终止本轮 run。

用户下一条消息经过正常入口（`process_message`）再次进入 agent loop，
带着包含澄清问答的历史上下文——不需要特殊的 enqueue。
"""

from __future__ import annotations

import contextvars
from typing import Optional


CLARIFY_PENDING: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "clarify_pending",
    default=None,
)


def mark_clarify(question: str) -> None:
    """Clarify 工具 handler 调用这个写入澄清文本。"""
    CLARIFY_PENDING.set(question or None)


def take_clarify() -> Optional[str]:
    """Agent loop 每轮结束时调一次。有值就返回并清空，让下一轮从干净状态开始。"""
    value = CLARIFY_PENDING.get()
    if value is not None:
        CLARIFY_PENDING.set(None)
    return value


def reset_clarify():
    """Run 开始时调一次，清掉上一轮残留。返回 token，配合 `restore_clarify`。"""
    return CLARIFY_PENDING.set(None)


def restore_clarify(token) -> None:
    """Run 结束时调一次。"""
    CLARIFY_PENDING.reset(token)
