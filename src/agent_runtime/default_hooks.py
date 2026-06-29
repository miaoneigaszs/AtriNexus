"""Default AgentHooks implementation.

The default hook set combines tool permission checks, Anthropic prompt-cache
markers, rate-limit header capture, and context-engine usage updates. Keeping
these concerns behind AgentHooks keeps the main loop explicit and testable.
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
