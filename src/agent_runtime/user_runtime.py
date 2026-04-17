"""Per-user agent runtime 状态：活跃 run 跟踪 + follow-up 队列 + 取消信号。

对标 pi-mono 的 `Agent` 状态容器：一个用户在任何时刻最多只有一个活跃 run；
用户在 run 期间发的消息会进 follow-up 队列，agent 本轮结束后再消化；用户
主动要求取消时，通过 abort event 触发 `asyncio.CancelledError` 在下一个 hook
边界停止。

WeCom 场景下**不做**"mid-run steering 注入"——那需要把 LangChain 的 ainvoke
切开重放；留到 Phase 4 自建 loop 时实现。

**线程安全**：所有公开方法持 `asyncio.Lock`，保证同一 user_id 的并发操作看到
一致视图。跨 event loop 共用同一 registry 不安全；WeCom 场景下 FastAPI 的
单 event loop 模型足够。
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional


logger = logging.getLogger("wecom")


# ── 队列 ─────────────────────────────────────────────────────────────────


class PendingMessageQueue:
    """简单的先进先出消息队列，支持 all / one-at-a-time 两种 drain 模式。

    对标 pi-mono 的 PendingMessageQueue：
    - "all" —— 每次 drain 把队列清空，返回所有已 enqueue 的消息
    - "one-at-a-time" —— 每次 drain 只返回首条，其余留在队列
    """

    def __init__(self, mode: str = "all") -> None:
        if mode not in {"all", "one-at-a-time"}:
            raise ValueError(f"unknown drain mode: {mode}")
        self.mode = mode
        self._items: List[str] = []

    def enqueue(self, message: str) -> None:
        self._items.append(message)

    def has_items(self) -> bool:
        return bool(self._items)

    def drain(self) -> List[str]:
        if not self._items:
            return []
        if self.mode == "all":
            drained = list(self._items)
            self._items.clear()
            return drained
        first = self._items[0]
        self._items = self._items[1:]
        return [first]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


# ── Per-user 运行状态 ───────────────────────────────────────────────────


@dataclass
class UserRunState:
    """单个用户的 agent run 状态。一个 user_id 同一时刻至多一个活跃 run。"""

    abort_event: asyncio.Event = field(default_factory=asyncio.Event)
    follow_up: PendingMessageQueue = field(default_factory=lambda: PendingMessageQueue("all"))
    is_running: bool = False

    def reset_abort(self) -> None:
        self.abort_event = asyncio.Event()


# ── Abort 信号穿透：contextvar ──────────────────────────────────────────


# hook 里可以通过 CURRENT_ABORT_EVENT.get() 取到当前 run 的取消信号。
# 没处于 run 里时值为 None。
CURRENT_ABORT_EVENT: contextvars.ContextVar[Optional[asyncio.Event]] = contextvars.ContextVar(
    "atrinexus_current_abort_event",
    default=None,
)


def abort_requested() -> bool:
    """hook 或工具内部的便捷查询：当前 run 是否被要求取消。"""
    event = CURRENT_ABORT_EVENT.get()
    return event is not None and event.is_set()


# ── Registry ────────────────────────────────────────────────────────────


class UserRuntimeRegistry:
    """进程级 per-user runtime 状态表。

    用法：
    - `claim_run(user_id)` 是异步上下文管理器；agent service 在 generate_reply_async
      外层用它圈定一次活跃 run，进入时 set contextvar 方便 hook 读取取消信号，
      退出时清状态、尝试消耗 follow-up。
    - `is_running(user_id)` 供 WeCom 入口判断是否要入队 follow-up。
    - `abort(user_id)` 由 message_handler 在用户发 "取消" 时调用。
    """

    def __init__(self) -> None:
        self._states: Dict[str, UserRunState] = {}
        self._lock = asyncio.Lock()

    async def _get_or_create(self, user_id: str) -> UserRunState:
        async with self._lock:
            state = self._states.get(user_id)
            if state is None:
                state = UserRunState()
                self._states[user_id] = state
            return state

    async def is_running(self, user_id: str) -> bool:
        state = self._states.get(user_id)
        return bool(state and state.is_running)

    async def abort(self, user_id: str) -> bool:
        """返回是否真的有活跃 run 被置位为 abort。无活跃 run 时返回 False。"""
        state = self._states.get(user_id)
        if state is None or not state.is_running:
            return False
        state.abort_event.set()
        logger.info("Agent run abort requested: user=%s", user_id)
        return True

    async def queue_follow_up(self, user_id: str, message: str) -> int:
        """把用户消息推进 follow-up 队列，返回当前队列长度。"""
        state = await self._get_or_create(user_id)
        state.follow_up.enqueue(message)
        return len(state.follow_up)

    async def drain_follow_up(self, user_id: str) -> List[str]:
        state = self._states.get(user_id)
        if state is None:
            return []
        return state.follow_up.drain()

    async def clear_follow_up(self, user_id: str) -> None:
        state = self._states.get(user_id)
        if state is not None:
            state.follow_up.clear()

    @asynccontextmanager
    async def claim_run(self, user_id: str) -> AsyncIterator[UserRunState]:
        """进入 agent 活跃 run。必须用 `async with` 包住 generate_reply_async。"""
        state = await self._get_or_create(user_id)
        async with self._lock:
            if state.is_running:
                raise RuntimeError(
                    f"user {user_id} 已有活跃 agent run；调用方应先 is_running 检查并入队"
                )
            state.is_running = True
            state.reset_abort()
        token = CURRENT_ABORT_EVENT.set(state.abort_event)
        try:
            yield state
        finally:
            CURRENT_ABORT_EVENT.reset(token)
            async with self._lock:
                state.is_running = False


# ── 进程级单例 ───────────────────────────────────────────────────────────


user_runtime = UserRuntimeRegistry()
