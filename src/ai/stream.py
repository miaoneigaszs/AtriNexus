"""SSE 行解析与 OpenAI chat-completions 流式 chunk 累积。

单独成模块是因为：解析逻辑跟具体 provider 无关；OpenAI、DeepSeek、Moonshot、
LiteLLM 等所有 OpenAI 兼容代理共享同一份。Anthropic native（PR13）会有自己
不同的事件流（content_block_start / content_block_delta / message_delta），
另写一个解析器。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Tuple

from src.ai.types import (
    StreamDone,
    StreamError,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    Usage,
)


logger = logging.getLogger("wecom")


# ── 通用 SSE 行解析 ─────────────────────────────────────────────────────


async def iter_sse_lines(byte_iter: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """把 chunked 字节流切成 SSE 行（每条 `data: ...\\n\\n`）。

    对 OpenAI / DeepSeek 等的标准 `text/event-stream` 响应通用。
    """
    buffer = b""
    async for chunk in byte_iter:
        if not chunk:
            continue
        buffer += chunk
        while b"\n" in buffer:
            line_bytes, buffer = buffer.split(b"\n", 1)
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\r")
            if not line:
                continue
            if line.startswith(":"):  # 注释行
                continue
            yield line


# ── OpenAI 兼容 chunk 解析 ──────────────────────────────────────────────


@dataclass
class _ToolCallAccumulator:
    """一次 stream 中累积一个 tool call 的全量信息。"""

    index: int
    id: Optional[str] = None
    name: Optional[str] = None
    args_buffer: str = ""

    def to_call(self) -> Optional[Tuple[str, str, dict]]:
        """流结束后把累积的 args JSON 解析成 dict；解析失败返回 None。"""
        if not self.id or not self.name:
            return None
        try:
            args = json.loads(self.args_buffer or "{}")
        except json.JSONDecodeError as exc:
            logger.warning(
                "tool_call args JSON 解析失败 id=%s name=%s err=%s buffer=%r",
                self.id,
                self.name,
                exc,
                self.args_buffer[:200],
            )
            return None
        if not isinstance(args, dict):
            return None
        return (self.id, self.name, args)


@dataclass
class StreamAccumulator:
    """流式 OpenAI chunk 的累积器。在 stream 完成后给出最终 assistant message 与 usage。"""

    text_parts: List[str] = field(default_factory=list)
    tool_calls: Dict[int, _ToolCallAccumulator] = field(default_factory=dict)
    stop_reason: str = ""
    usage: Optional[Usage] = None

    def feed_chunk(self, chunk: dict) -> List[StreamEvent]:
        """接收一条 OpenAI chunk dict，返回 0~多条 StreamEvent。"""
        events: List[StreamEvent] = []
        choices = chunk.get("choices") or []
        if choices:
            choice = choices[0]
            delta = choice.get("delta") or {}

            text = delta.get("content")
            if isinstance(text, str) and text:
                self.text_parts.append(text)
                events.append(TextDelta(text=text))

            for tool_delta in delta.get("tool_calls") or []:
                index = int(tool_delta.get("index", 0))
                acc = self.tool_calls.get(index)
                if acc is None:
                    acc = _ToolCallAccumulator(index=index)
                    self.tool_calls[index] = acc

                if "id" in tool_delta and tool_delta["id"]:
                    acc.id = str(tool_delta["id"])
                func = tool_delta.get("function") or {}
                if "name" in func and func["name"]:
                    acc.name = str(func["name"])
                if "arguments" in func and func["arguments"]:
                    fragment = str(func["arguments"])
                    acc.args_buffer += fragment
                    events.append(ToolCallDelta(
                        index=index,
                        id=acc.id,
                        name=acc.name,
                        args_delta=fragment,
                    ))

            finish = choice.get("finish_reason")
            if finish:
                self.stop_reason = str(finish)

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.usage = Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                cache_read_tokens=int(
                    usage.get("prompt_cache_hit_tokens", 0)  # DeepSeek
                    or (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)  # OpenAI
                    or 0
                ),
                cache_write_tokens=0,
            )

        return events

    def finalize(self) -> StreamDone:
        return StreamDone(stop_reason=self.stop_reason, usage=self.usage or Usage())

    def collected_text(self) -> str:
        return "".join(self.text_parts)

    def collected_tool_calls(self) -> List[Tuple[str, str, dict]]:
        results = []
        for index in sorted(self.tool_calls.keys()):
            call = self.tool_calls[index].to_call()
            if call is not None:
                results.append(call)
        return results


# ── 顶层 driver：把 SSE 字节流变成 StreamEvent ─────────────────────────


async def stream_openai_chunks(
    byte_iter: AsyncIterator[bytes],
    accumulator: Optional[StreamAccumulator] = None,
) -> AsyncIterator[StreamEvent]:
    """主入口：消费 SSE 字节流，吐 StreamEvent；最后一个事件是 StreamDone。

    `accumulator` 可由调用方传入（用于流后查最终 text + tool_calls）。不传就内部建一个。
    """
    acc = accumulator or StreamAccumulator()

    async for line in iter_sse_lines(byte_iter):
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload:
            continue
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError as exc:
            logger.warning("流响应 chunk JSON 解析失败: %s line=%r", exc, payload[:200])
            continue

        if not isinstance(chunk, dict):
            continue

        # provider 把错误以 chunk 形式塞回来时（如限流）这里转成 StreamError
        if "error" in chunk:
            err = chunk["error"]
            message = err.get("message") if isinstance(err, dict) else str(err)
            yield StreamError(message=str(message or "unknown provider error"))
            yield acc.finalize()
            return

        for event in acc.feed_chunk(chunk):
            yield event

    yield acc.finalize()


__all__ = [
    "iter_sse_lines",
    "StreamAccumulator",
    "stream_openai_chunks",
]
