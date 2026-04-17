"""OpenAI 兼容 provider：覆盖 OpenAI / DeepSeek / Moonshot / vLLM / LiteLLM 等。

依赖 httpx 异步客户端。请求体走 chat-completions 标准；流式响应通过
`src.ai.stream.stream_openai_chunks` 解析成 StreamEvent。

错误处理约定（与 base.ProviderAdapter 契约一致）：
- 网络/连接错误 → 一条 StreamError + 一条空 StreamDone
- HTTP 4xx/5xx → 同上
- provider 在流中 yield 错误 chunk → 一条 StreamError + StreamDone
- StreamDone.usage 永远存在（缺数据时是 Usage(0,0,0,0)）
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Optional

import httpx

from src.ai.providers.base import ProviderAdapter, ProviderRequest
from src.ai.registry import get_capabilities
from src.ai.stream import StreamAccumulator, stream_openai_chunks
from src.ai.types import (
    StreamDone,
    StreamError,
    StreamEvent,
    Usage,
    messages_to_openai,
    tools_to_openai,
)


logger = logging.getLogger("wecom")


class OpenAICompatProvider(ProviderAdapter):
    """OpenAI 兼容协议的最小 provider 实现。"""

    name = "openai_compat"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float = 120.0,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAICompatProvider 需要非空 api_key")
        if not base_url:
            raise ValueError("OpenAICompatProvider 需要非空 base_url")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = client  # 可选注入用于测试
        self._owns_client = client is None

    def _build_payload(self, request: ProviderRequest) -> dict:
        payload: dict = {
            "model": request.model,
            "messages": messages_to_openai(request.messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if request.tools and self.supports_tools(request.model):
            payload["tools"] = tools_to_openai(request.tools)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        return payload

    def _build_headers(self, request: ProviderRequest) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if request.extra_headers:
            headers.update(request.extra_headers)
        return headers

    def supports_tools(self, model: str) -> bool:
        return get_capabilities(model).supports_tools

    async def stream(self, request: ProviderRequest) -> AsyncIterator[StreamEvent]:
        url = f"{self.base_url}/chat/completions"
        payload = self._build_payload(request)
        headers = self._build_headers(request)
        accumulator = StreamAccumulator()

        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            try:
                async with client.stream(
                    "POST", url, json=payload, headers=headers
                ) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        message = self._format_http_error(response.status_code, body)
                        logger.warning("OpenAI compat HTTP %s: %s", response.status_code, message)
                        yield StreamError(message=message)
                        yield StreamDone(stop_reason="error", usage=Usage())
                        return

                    async for event in stream_openai_chunks(response.aiter_bytes(), accumulator):
                        yield event
            except httpx.HTTPError as exc:
                logger.warning("OpenAI compat 请求失败: %s", exc)
                yield StreamError(message=str(exc) or exc.__class__.__name__)
                yield StreamDone(stop_reason="error", usage=Usage())
        finally:
            if self._owns_client:
                await client.aclose()

    def _format_http_error(self, status: int, body: bytes) -> str:
        text = body.decode("utf-8", errors="replace") if body else ""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict) and err.get("message"):
                return f"HTTP {status}: {err['message']}"
        return f"HTTP {status}: {text[:300]}"


__all__ = ["OpenAICompatProvider"]
