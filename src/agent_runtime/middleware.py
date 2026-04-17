"""LangChain 中间件装配层。

这里是 AtriNexus **唯一** 的 LangChain 集成点。`agent_runtime.hooks` 定义的
`AgentHooks` 协议是框架中立的业务接口；本模块负责把 hook 调用翻译成 LangChain 的
`wrap_tool_call` / `wrap_model_call` / `dynamic_prompt` 原语。

Phase 4 替代 LangChain 后，本文件会被删除，agent_loop 将直接调用 hook。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Mapping, Optional

from langchain.agents.middleware import dynamic_prompt, wrap_model_call, wrap_tool_call
from langchain_core.messages import ToolMessage

from src.agent_runtime.hooks import (
    AfterToolCallContext,
    AgentHooks,
    BeforeToolCallContext,
    OnResponseContext,
)
from src.prompting.prompt_manager import PromptManager
from src.platform_core.metrics import Metrics, PROMETHEUS_AVAILABLE


logger = logging.getLogger("wecom")


# ── Prompt middleware ───────────────────────────────────────────────────


def build_dynamic_prompt_middleware(prompt_manager: PromptManager) -> Callable:
    """把每轮动态 runtime prompt 接到 LangChain 的 dynamic_prompt。"""

    @dynamic_prompt
    def runtime_prompt(request) -> str:
        context = request.runtime.context
        return prompt_manager.build_runtime_prompt(
            persona_prompt=context.persona_prompt,
            tool_profile=context.tool_profile,
            tool_profiles=context.tool_profiles,
            tool_summary=context.tool_summary,
            core_memory=context.core_memory,
        )

    return runtime_prompt


# ── Model middleware（含指标 + on_response hook） ───────────────────────


def build_model_middleware(
    default_model_name: str,
    hooks: Optional[AgentHooks] = None,
) -> Callable:
    """为 model 调用加上指标、结构化日志，并在响应回来后调 on_response hook。"""

    @wrap_model_call
    async def managed_model_call(request, handler):
        model_name = (
            getattr(request.model, "model_name", None)
            or getattr(request.model, "model", None)
            or default_model_name
        )
        start = time.perf_counter()
        if PROMETHEUS_AVAILABLE and Metrics.active_llm_requests:
            Metrics.active_llm_requests.labels(model=model_name, request_type="agent").inc()
        logger.info(
            "Model call start: model=%s messages=%s tools=%s",
            model_name,
            len(request.messages),
            len(request.tools),
        )
        try:
            response = await handler(request)
        except Exception as exc:
            duration = time.perf_counter() - start
            logger.warning(
                "Model call failed: model=%s duration_ms=%.2f error=%s",
                model_name,
                duration * 1000,
                exc,
            )
            if PROMETHEUS_AVAILABLE and Metrics.llm_errors_total:
                Metrics.llm_errors_total.labels(model=model_name, error_type=type(exc).__name__).inc()
            raise
        finally:
            if PROMETHEUS_AVAILABLE and Metrics.active_llm_requests:
                Metrics.active_llm_requests.labels(model=model_name, request_type="agent").dec()

        duration = time.perf_counter() - start
        logger.info("Model call end: model=%s duration_ms=%.2f", model_name, duration * 1000)
        if PROMETHEUS_AVAILABLE and Metrics.llm_request_duration:
            Metrics.llm_request_duration.labels(model=model_name).observe(duration)

        if hooks is not None:
            try:
                hooks.on_response(
                    OnResponseContext(
                        model=model_name,
                        response=response,
                        response_metadata=_extract_metadata(response),
                        duration_ms=duration * 1000,
                    )
                )
            except Exception as exc:
                logger.debug("on_response hook 失败: %s", exc)

        return response

    return managed_model_call


def _extract_metadata(response: Any) -> Optional[Mapping[str, Any]]:
    """尽力从 LangChain 响应里抓元信息（含 provider 头）。

    不同 provider 把原始头放在不同位置；依次尝试，都没有就返回 None。
    """
    messages = getattr(response, "messages", None)
    if messages:
        last_ai = next(
            (m for m in reversed(messages) if getattr(m, "type", "") == "ai"),
            None,
        )
        if last_ai is not None:
            metadata = getattr(last_ai, "response_metadata", None)
            if isinstance(metadata, Mapping):
                return metadata

    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, Mapping):
        return metadata
    return None


# ── Tool middleware（hook 翻译层） ──────────────────────────────────────


def build_tool_middleware_from_hooks(hooks: AgentHooks) -> Callable:
    """把 AgentHooks.before_tool_call / after_tool_call 包成 LangChain wrap_tool_call。

    语义：
    - before_tool_call 返回 block=True → 直接返回错误 ToolMessage，不执行工具
    - before_tool_call 返回 repaired_args → 替换 request.tool_call["args"]
    - 工具执行过程里抛异常 → 封装成错误 ToolMessage
    - after_tool_call 返回 content 覆盖 → 替换 ToolMessage content
    - 超长仍走兜底截断（保留与 PR4-PR7 一致的边界）
    """

    @wrap_tool_call
    async def managed_tool_call(request, handler):
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "<unknown>")
        tool_args = tool_call.get("args", {}) or {}
        call_id = tool_call.get("id", "")

        before = None
        try:
            before = await hooks.before_tool_call(
                BeforeToolCallContext(tool_name=tool_name, args=tool_args, call_id=call_id)
            )
        except Exception as exc:
            logger.debug("before_tool_call hook 失败: %s", exc)

        if before is not None:
            if before.block:
                return ToolMessage(
                    content=before.reason or f"工具 {tool_name} 被拒绝执行",
                    tool_call_id=call_id,
                    status="error",
                )
            if before.repaired_args is not None and before.repaired_args != tool_args:
                request.tool_call["args"] = before.repaired_args
                tool_args = before.repaired_args

        try:
            response = await handler(request)
        except Exception as exc:
            logger.warning("Tool call failed: name=%s error=%s", tool_name, exc)
            return ToolMessage(
                content=f"工具 {tool_name} 执行失败：{exc}",
                tool_call_id=call_id,
                status="error",
            )

        if not isinstance(response, ToolMessage):
            logger.info("Tool call end: name=%s result_type=%s", tool_name, type(response).__name__)
            return response

        content = _extract_tool_message_text(response)
        is_error = response.status == "error"

        try:
            after = await hooks.after_tool_call(
                AfterToolCallContext(
                    tool_name=tool_name,
                    args=tool_args,
                    call_id=call_id,
                    result_content=content,
                    is_error=is_error,
                )
            )
        except Exception as exc:
            logger.debug("after_tool_call hook 失败: %s", exc)
            after = None

        final_content = content
        final_is_error = is_error
        if after is not None:
            if after.content is not None:
                final_content = after.content
            if after.is_error is not None:
                final_is_error = after.is_error

        if final_content == content and final_is_error == is_error:
            return response

        return ToolMessage(
            content=final_content,
            tool_call_id=response.tool_call_id,
            status="error" if final_is_error else response.status,
            artifact=response.artifact,
            name=response.name,
            id=response.id,
        )

    return managed_tool_call


def _extract_tool_message_text(message: ToolMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)
