"""自建 agent loop，替代 langchain.agents.create_agent。

把 PR7-PR11 的所有横切件串起来：
- provider.stream() 走 PR11 的 OpenAICompatProvider
- 工具调用前后跑 PR8 的 hook（before_tool_call / after_tool_call）
- transform_context hook 在每轮发请求前对消息做最后修饰（如 Anthropic prompt cache）
- on_response hook 在每轮响应后跑（如 rate-limit 头抓取、context_engine 喂 usage）
- abort 信号通过 PR9 的 contextvar 在 hook 边界即时生效
- context_engine 在每轮 invoke 前 preflight 压缩消息

Loop 的形态对标 pi-mono 的 agent-loop.ts，但保持显式与可读：
1. 取消？stop_reason="cancelled"。
2. 压缩历史。
3. transform_context hook。
4. provider.stream → consume_stream → 拿 text + tool_calls + usage。
5. 没工具调用：写入最终 assistant 消息，结束。
6. 有工具调用：把 assistant 消息写入历史，依次执行每个工具（受 hook 控制），
   把 tool 结果消息写入历史，进入下一轮。
7. 超过 max_iterations：写一条系统警示 + 结束。
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agent_runtime.context_engine import ContextEngine
from src.agent_runtime.clarify_store import reset_clarify, take_clarify
from src.agent_runtime.hooks import (
    AfterToolCallContext,
    AgentHooks,
    BeforeToolCallContext,
    OnResponseContext,
    TransformContextContext,
)
from src.agent_runtime.tool_catalog import RegisteredTool
from src.agent_runtime.user_runtime import abort_requested
from src.ai.providers.base import ProviderAdapter, ProviderRequest
from src.ai.stream import consume_stream
from src.ai.types import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
    Usage,
    messages_to_openai,
)


logger = logging.getLogger("wecom")


# ── 输出结构 ────────────────────────────────────────────────────────────


@dataclass
class ToolEvent:
    """供 trajectory 落盘的单次工具调用记录。"""

    name: str
    args: Dict[str, Any]
    result: str
    status: str  # "ok" / "error" / "blocked"


@dataclass
class LoopResult:
    """run_agent_loop 的最终输出。"""

    text: str
    usage: Usage
    stop_reason: str
    iterations: int
    tool_events: List[ToolEvent] = field(default_factory=list)
    cancelled: bool = False


# ── 主循环 ──────────────────────────────────────────────────────────────


async def run_agent_loop(
    *,
    provider: ProviderAdapter,
    model: str,
    system_prompt: str,
    initial_messages: List[Message],
    tools: List[RegisteredTool],
    hooks: AgentHooks,
    context_engine: Optional[ContextEngine] = None,
    max_iterations: int = 12,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> LoopResult:
    """同步 agent loop。返回最终回复文本和元信息。

    initial_messages 是除 system 外的对话历史 + 当前 user 消息（按时间顺序）。
    system_prompt 会自动作为消息列表第一条 system 消息送出。
    """
    tool_index: Dict[str, RegisteredTool] = {t.spec.name: t for t in tools}
    tool_specs: List[ToolSpec] = [t.spec for t in tools]

    messages: List[Message] = [SystemMessage(content=system_prompt)] if system_prompt else []
    messages.extend(initial_messages)

    tool_events: List[ToolEvent] = []
    total_usage = Usage()
    last_text = ""
    stop_reason = ""

    # 每个 run 清一次 clarify 信号；上一次 run 留下的残留都当作无效。
    reset_clarify()

    for iteration in range(1, max_iterations + 1):
        if abort_requested():
            return LoopResult(
                text="",
                usage=total_usage,
                stop_reason="cancelled",
                iterations=iteration - 1,
                tool_events=tool_events,
                cancelled=True,
            )

        messages = _maybe_compress(messages, context_engine)
        messages = _apply_transform_context(messages, model, hooks)

        request = ProviderRequest(
            model=model,
            messages=list(messages),
            tools=list(tool_specs),
            temperature=temperature,
            max_tokens=max_tokens,
        )

        import time
        started = time.perf_counter()
        summary = await consume_stream(provider.stream(request))
        duration_ms = (time.perf_counter() - started) * 1000.0

        _fire_on_response(hooks, model, summary, duration_ms)

        if context_engine is not None and summary.usage.total_tokens > 0:
            try:
                context_engine.update_from_response({
                    "prompt_tokens": summary.usage.prompt_tokens,
                    "completion_tokens": summary.usage.completion_tokens,
                })
            except Exception as exc:
                logger.debug("context_engine.update_from_response 失败: %s", exc)

        total_usage = total_usage.merge(summary.usage)
        last_text = summary.text
        stop_reason = summary.stop_reason

        if summary.error:
            logger.warning("Provider 流报错，终止 loop: %s", summary.error)
            return LoopResult(
                text=last_text or "",
                usage=total_usage,
                stop_reason="error",
                iterations=iteration,
                tool_events=tool_events,
            )

        if not summary.tool_calls:
            return LoopResult(
                text=last_text,
                usage=total_usage,
                stop_reason=stop_reason or "stop",
                iterations=iteration,
                tool_events=tool_events,
            )

        assistant_msg = AssistantMessage(
            content=summary.text,
            tool_calls=[
                ToolCall(id=call_id, name=name, args=args)
                for call_id, name, args in summary.tool_calls
            ],
        )
        messages.append(assistant_msg)

        for call_id, tool_name, raw_args in summary.tool_calls:
            if abort_requested():
                return LoopResult(
                    text="",
                    usage=total_usage,
                    stop_reason="cancelled",
                    iterations=iteration,
                    tool_events=tool_events,
                    cancelled=True,
                )

            registered = tool_index.get(tool_name)
            if registered is None:
                err_text = f"工具 {tool_name} 不存在或当前 profile 未启用。"
                messages.append(ToolResultMessage(
                    tool_call_id=call_id,
                    content=err_text,
                    is_error=True,
                    name=tool_name,
                ))
                tool_events.append(ToolEvent(name=tool_name, args=raw_args, result=err_text, status="error"))
                continue

            args, before_result = await _run_before_hook(hooks, tool_name, raw_args, call_id)
            if before_result is not None and before_result.block:
                blocked_text = before_result.reason or f"工具 {tool_name} 被拒绝执行。"
                messages.append(ToolResultMessage(
                    tool_call_id=call_id,
                    content=blocked_text,
                    is_error=True,
                    name=tool_name,
                ))
                tool_events.append(ToolEvent(name=tool_name, args=args, result=blocked_text, status="blocked"))
                continue

            result_text, is_error = await _execute_tool(registered, args)
            after_result = await _run_after_hook(hooks, tool_name, args, call_id, result_text, is_error)
            if after_result is not None:
                if after_result.content is not None:
                    result_text = after_result.content
                if after_result.is_error is not None:
                    is_error = after_result.is_error

            messages.append(ToolResultMessage(
                tool_call_id=call_id,
                content=result_text,
                is_error=is_error,
                name=tool_name,
            ))
            tool_events.append(ToolEvent(
                name=tool_name,
                args=args,
                result=result_text,
                status="error" if is_error else "ok",
            ))

        clarify_question = take_clarify()
        if clarify_question:
            return LoopResult(
                text=clarify_question,
                usage=total_usage,
                stop_reason="clarify",
                iterations=iteration,
                tool_events=tool_events,
            )

    logger.warning("Agent loop 触达 max_iterations=%s，强制结束", max_iterations)
    messages.append(SystemMessage(content=f"[已达到最大轮数 {max_iterations}，停止继续工具调用。请直接给出当前结论。]"))
    # 给模型最后一次发言机会，让它把当前结果整理成回复
    try:
        request = ProviderRequest(
            model=model,
            messages=list(messages),
            tools=[],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        final = await consume_stream(provider.stream(request))
        if final.text:
            last_text = final.text
        total_usage = total_usage.merge(final.usage)
    except Exception as exc:
        logger.warning("收尾调用失败: %s", exc)

    return LoopResult(
        text=last_text,
        usage=total_usage,
        stop_reason="max_iterations",
        iterations=max_iterations,
        tool_events=tool_events,
    )


# ── 内部 helper ─────────────────────────────────────────────────────────


def _maybe_compress(messages: List[Message], engine: Optional[ContextEngine]) -> List[Message]:
    if engine is None:
        return messages
    try:
        openai_msgs = messages_to_openai(messages)
        if not engine.should_compress_preflight(openai_msgs):
            return messages
        compressed = engine.compress(openai_msgs)
        return _from_openai_dicts(compressed)
    except Exception as exc:
        logger.debug("context_engine 压缩失败，按原样发送: %s", exc)
        return messages


def _from_openai_dicts(items: List[Dict[str, Any]]) -> List[Message]:
    """把 dict 形式的消息列表转回 typed Message。压缩器吐的是 dict，agent loop 用 typed。

    暂只覆盖 system / user / assistant / tool 这四种。tool_calls 字段保留。
    """
    from src.ai.types import SystemMessage as _S, UserMessage as _U, ToolResultMessage as _TR

    converted: List[Message] = []
    for item in items:
        role = item.get("role")
        content = item.get("content", "")
        if role == "system":
            converted.append(_S(content=str(content) if isinstance(content, str) else json.dumps(content, ensure_ascii=False)))
        elif role == "user":
            converted.append(_U(content=str(content) if isinstance(content, str) else json.dumps(content, ensure_ascii=False)))
        elif role == "assistant":
            tool_calls_raw = item.get("tool_calls") or []
            tool_calls = []
            for tc in tool_calls_raw:
                func = tc.get("function") or {}
                name = func.get("name", "")
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(id=str(tc.get("id", "")), name=str(name), args=args if isinstance(args, dict) else {}))
            converted.append(AssistantMessage(content=str(content) if isinstance(content, str) else "", tool_calls=tool_calls))
        elif role == "tool":
            converted.append(_TR(
                tool_call_id=str(item.get("tool_call_id", "")),
                content=str(content),
                name=item.get("name"),
            ))
        # 未识别 role 直接忽略——压缩器只产合法消息
    return converted


def _apply_transform_context(
    messages: List[Message],
    model: str,
    hooks: AgentHooks,
) -> List[Message]:
    transform = getattr(hooks, "transform_context", None)
    if transform is None:
        return messages
    try:
        ctx = TransformContextContext(messages=messages_to_openai(messages), model=model)
        result = transform(ctx)
    except Exception as exc:
        logger.debug("transform_context hook 失败: %s", exc)
        return messages
    if result is None or result.messages is None:
        return messages
    return _from_openai_dicts(result.messages)


def _fire_on_response(
    hooks: AgentHooks,
    model: str,
    summary: Any,
    duration_ms: float,
) -> None:
    on_response = getattr(hooks, "on_response", None)
    if on_response is None:
        return
    try:
        on_response(OnResponseContext(
            model=model,
            response=summary,
            response_metadata=None,  # 自建 provider 当前不暴露原始头；后续接 Anthropic native 时可补
            duration_ms=duration_ms,
        ))
    except Exception as exc:
        logger.debug("on_response hook 失败: %s", exc)


async def _run_before_hook(
    hooks: AgentHooks,
    tool_name: str,
    args: Dict[str, Any],
    call_id: str,
):
    before = getattr(hooks, "before_tool_call", None)
    if before is None:
        return args, None
    try:
        result = await before(BeforeToolCallContext(tool_name=tool_name, args=args, call_id=call_id))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.debug("before_tool_call hook 失败: %s", exc)
        return args, None
    if result is None:
        return args, None
    if result.repaired_args is not None:
        return result.repaired_args, result
    return args, result


async def _run_after_hook(
    hooks: AgentHooks,
    tool_name: str,
    args: Dict[str, Any],
    call_id: str,
    result_text: str,
    is_error: bool,
):
    after = getattr(hooks, "after_tool_call", None)
    if after is None:
        return None
    try:
        return await after(AfterToolCallContext(
            tool_name=tool_name,
            args=args,
            call_id=call_id,
            result_content=result_text,
            is_error=is_error,
        ))
    except Exception as exc:
        logger.debug("after_tool_call hook 失败: %s", exc)
        return None


async def _execute_tool(registered: RegisteredTool, args: Dict[str, Any]):
    try:
        text = await registered.handler(args)
        return text, False
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Tool %s 执行抛异常: %s", registered.name, exc)
        return f"工具 {registered.name} 执行失败：{exc}", True


__all__ = ["run_agent_loop", "LoopResult", "ToolEvent"]
