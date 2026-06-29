"""Agent runtime hook interfaces.

Hooks keep cross-cutting concerns outside the core loop: tool permission checks,
result shaping, outbound context transforms, and response observation. Returning
None means "leave the current value unchanged"; returning a result object applies
the fields set on that object.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Dict, List, Optional, Protocol, runtime_checkable


# ── Tool call hooks ─────────────────────────────────────────────────────


@dataclass
class BeforeToolCallContext:
    """工具开始执行前的上下文。"""

    tool_name: str
    args: Dict[str, Any]
    call_id: str


@dataclass
class BeforeToolCallResult:
    """before_tool_call 的返回值。字段全部可选，只有被设置的才生效。"""

    block: bool = False
    """True 则拒绝执行，用 reason 作为 tool 错误消息返回给 agent。"""

    reason: str = ""
    """block=True 时的用户/agent 可见解释。"""

    repaired_args: Optional[Dict[str, Any]] = None
    """不为 None 则替换原 args。支持轻量修正（路径规范化、去空格等）。"""


@dataclass
class AfterToolCallContext:
    """工具执行完但结果还没返回给 agent 前的上下文。"""

    tool_name: str
    args: Dict[str, Any]
    call_id: str
    result_content: str
    is_error: bool


@dataclass
class AfterToolCallResult:
    """after_tool_call 的返回值；字段都可选。"""

    content: Optional[str] = None
    """不为 None 则替换结果文本（例如截断、整形）。"""

    is_error: Optional[bool] = None
    """不为 None 则覆盖错误标志。"""


# ── Context / response hooks ────────────────────────────────────────────


@dataclass
class TransformContextContext:
    """LLM 请求发送前的消息列表；hook 可以改写再送出。"""

    messages: List[Dict[str, Any]]
    model: str


@dataclass
class TransformContextResult:
    """transform_context 的返回值。"""

    messages: Optional[List[Dict[str, Any]]] = None
    """不为 None 则替换消息列表。"""


@dataclass
class OnResponseContext:
    """LLM 响应回来后的观测上下文；hook 不改响应，只收集信号。"""

    model: str
    response: Any
    """Provider 原始响应对象或响应摘要。"""

    response_metadata: Optional[Dict[str, Any]] = None
    """若 provider 暴露了 metadata / headers，提前抽出来传入。"""

    duration_ms: float = 0.0


# ── Protocol ────────────────────────────────────────────────────────────


@runtime_checkable
class AgentHooks(Protocol):
    """Agent 运行时 hook 协议。实现方按需覆盖方法；未实现的按"不改"处理。"""

    async def before_tool_call(
        self, ctx: BeforeToolCallContext
    ) -> Optional[BeforeToolCallResult]:
        ...

    async def after_tool_call(
        self, ctx: AfterToolCallContext
    ) -> Optional[AfterToolCallResult]:
        ...

    def transform_context(
        self, ctx: TransformContextContext
    ) -> Optional[TransformContextResult]:
        ...

    def on_response(self, ctx: OnResponseContext) -> None:
        ...


# ── No-op baseline ──────────────────────────────────────────────────────


class NoopAgentHooks:
    """所有 hook 都返回 None / 无操作，方便在 Service 里做安全回退。"""

    async def before_tool_call(
        self, ctx: BeforeToolCallContext
    ) -> Optional[BeforeToolCallResult]:
        return None

    async def after_tool_call(
        self, ctx: AfterToolCallContext
    ) -> Optional[AfterToolCallResult]:
        return None

    def transform_context(
        self, ctx: TransformContextContext
    ) -> Optional[TransformContextResult]:
        return None

    def on_response(self, ctx: OnResponseContext) -> None:
        return None


# 便捷导出
__all__ = [
    "AgentHooks",
    "NoopAgentHooks",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "AfterToolCallContext",
    "AfterToolCallResult",
    "TransformContextContext",
    "TransformContextResult",
    "OnResponseContext",
]
