from __future__ import annotations

import logging
import time
from typing import Any, Callable, Mapping, Optional

from langchain.agents.middleware import dynamic_prompt, wrap_model_call

from src.prompting.prompt_manager import PromptManager
from src.platform_core.metrics import Metrics, PROMETHEUS_AVAILABLE
from src.platform_core.rate_limit import parse_rate_limit_headers, record_latest_state


logger = logging.getLogger("wecom")


def build_dynamic_prompt_middleware(prompt_manager: PromptManager) -> Callable:
    """把每轮动态 runtime prompt 接到 langchain agent 的 dynamic_prompt。"""

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


def build_model_middleware(default_model_name: str) -> Callable:
    """为 model 调用加上指标与结构化日志。"""

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
        _capture_rate_limit(response, model_name)
        return response

    return managed_model_call


def _capture_rate_limit(response: Any, model_name: str) -> None:
    """尽力从响应里抓 x-ratelimit-* 头；provider 没给就无声跳过。

    LangChain ChatOpenAI 把 provider 响应元信息放在 message.response_metadata 里，
    不同 provider 键名差异大：有的直接塞头 dict，有的叫 headers，有的一概没有。
    这里按常见位置依次尝试，找不到就不记录。
    """
    headers = _extract_headers(response)
    if not headers:
        return
    state = parse_rate_limit_headers(headers, provider=model_name)
    if state:
        record_latest_state(state)


def _extract_headers(response: Any) -> Optional[Mapping[str, str]]:
    metadata = _coerce_metadata(response)
    if not metadata:
        return None

    for key in ("headers", "response_headers", "raw_headers"):
        candidate = metadata.get(key)
        if isinstance(candidate, Mapping) and candidate:
            return candidate

    rate_keys = {k: v for k, v in metadata.items() if isinstance(k, str) and k.lower().startswith("x-ratelimit-")}
    return rate_keys or None


def _coerce_metadata(response: Any) -> Optional[Mapping[str, Any]]:
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
