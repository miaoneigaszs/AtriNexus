"""默认 AgentHooks 组合：tool guard + prompt caching + rate limit capture。

把 PR7 落地的 prompt_cache / rate_limit 工具和 PR8 的 AgentToolGuard 合并成
一个满足 AgentHooks 协议的复合对象。业务侧只需要一个 `DefaultAgentHooks(...)`
就把四个 hook 全部打开。

注意：`transform_context` 当前依赖 provider 接收 dict 形式的消息。LangChain 的
`ChatOpenAI` 在发送前会把 dict 转 `BaseMessage`，额外字段在这个过程中可能被
丢弃。所以在 LangChain 路径下该 hook 实际生效取决于 provider；等 Phase 4 自建
provider 层后会成为可靠路径。
"""

from __future__ import annotations

import logging
from typing import Optional

from src.agent_runtime.agent_tool_guard import AgentToolGuard
from src.agent_runtime.hooks import (
    AfterToolCallContext,
    AfterToolCallResult,
    BeforeToolCallContext,
    BeforeToolCallResult,
    OnResponseContext,
    TransformContextContext,
    TransformContextResult,
)
from src.agent_runtime.prompt_cache import (
    apply_anthropic_cache_control,
    model_supports_cache_control,
)
from src.platform_core.rate_limit import parse_rate_limit_headers, record_latest_state


logger = logging.getLogger("wecom")


class DefaultAgentHooks:
    """默认 hook 实现：委托 tool_guard + prompt 缓存 + 速率限制抓取。"""

    def __init__(self, tool_guard: AgentToolGuard) -> None:
        self.tool_guard = tool_guard

    async def before_tool_call(
        self, ctx: BeforeToolCallContext
    ) -> Optional[BeforeToolCallResult]:
        return await self.tool_guard.before_tool_call(ctx)

    async def after_tool_call(
        self, ctx: AfterToolCallContext
    ) -> Optional[AfterToolCallResult]:
        return await self.tool_guard.after_tool_call(ctx)

    def transform_context(
        self, ctx: TransformContextContext
    ) -> Optional[TransformContextResult]:
        if not model_supports_cache_control(ctx.model):
            return None
        transformed = apply_anthropic_cache_control(ctx.messages)
        if transformed is ctx.messages:
            return None
        return TransformContextResult(messages=transformed)

    def on_response(self, ctx: OnResponseContext) -> None:
        metadata = ctx.response_metadata or {}
        if not metadata:
            return
        state = parse_rate_limit_headers(metadata, provider=ctx.model)
        if state:
            record_latest_state(state)
