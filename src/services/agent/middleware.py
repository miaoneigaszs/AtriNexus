from __future__ import annotations

import logging
import time
from typing import Callable

from langchain.agents.middleware import dynamic_prompt, wrap_model_call

from src.services.prompt_manager import PromptManager
from src.utils.metrics import Metrics, PROMETHEUS_AVAILABLE


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
        return response

    return managed_model_call
