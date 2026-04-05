from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, wrap_model_call, wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI

from data.config import config
from src.services.agent.tool_catalog import ToolCatalog
from src.services.prompt_manager import PromptManager
from src.services.token_monitor import token_monitor
from src.utils.metrics import Metrics, PROMETHEUS_AVAILABLE

logger = logging.getLogger("wecom")

USER_VISIBLE_AGENT_ERROR = "抱歉，我暂时无法处理你的消息，请稍后再试。"
MODELS_WITHOUT_TOOL_SUPPORT = {"deepseek-reasoner", "deepseek-r1"}
TOOL_OVERVIEW_HINTS = (
    "有哪些工具",
    "有什么工具",
    "能用什么工具",
    "可以用什么工具",
    "能做什么",
    "会什么",
    "能力有哪些",
)
MAX_TOOL_RESULT_CHARS = 4000
MAX_TOOL_ARG_CHARS = 500


@dataclass
class AgentRunContext:
    """单轮 agent 调用的动态上下文。"""

    persona_prompt: str
    core_memory: Optional[str]
    kb_context: Optional[str]
    tool_profile: Optional[str]
    tool_profiles: List[str]
    tool_summary: str


class LangChainAgentService:
    """最小 LangChain Agent 适配层。

    只负责最终回复生成与工具调用，不接管记忆和 RAG 编排。
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        rag_service: Optional[object] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max(2, int(os.getenv("ATRINEXUS_AGENT_MAX_ITERATIONS", "8")))
        self.workspace_root = str(Path(__file__).resolve().parents[3])
        search_cfg = config.network_search
        search_api_key = search_cfg.api_key if search_cfg.search_enabled and search_cfg.api_key else None
        self.tool_catalog = ToolCatalog(
            workspace_root=self.workspace_root,
            search_api_key=search_api_key,
            rag_service=rag_service,
        )
        self.prompt_manager = PromptManager(self.workspace_root)
        self.model_client = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self._agent_cache: Dict[Tuple[str, ...], Any] = {}
        self._agent_cache_lock = threading.Lock()

    def generate_reply(
        self,
        message: str,
        user_id: str,
        system_prompt: str,
        tool_profile: Optional[str] = None,
        previous_context: Optional[List[Dict[str, Any]]] = None,
        core_memory: Optional[str] = None,
        kb_context: Optional[str] = None,
    ) -> str:
        try:
            asyncio.get_running_loop()
            logger.error("generate_reply() 在事件循环中被直接调用，请改用 generate_reply_async()")
            return USER_VISIBLE_AGENT_ERROR
        except RuntimeError:
            return asyncio.run(
                self.generate_reply_async(
                    message=message,
                    user_id=user_id,
                    system_prompt=system_prompt,
                    tool_profile=tool_profile,
                    previous_context=previous_context,
                    core_memory=core_memory,
                    kb_context=kb_context,
                )
            )

    async def generate_reply_async(
        self,
        message: str,
        user_id: str,
        system_prompt: str,
        tool_profile: Optional[str] = None,
        previous_context: Optional[List[Dict[str, Any]]] = None,
        core_memory: Optional[str] = None,
        kb_context: Optional[str] = None,
    ) -> str:
        try:
            tool_bundle = self.tool_catalog.build_tool_bundle(
                user_id=user_id,
                message=message,
                allow_tools=not self._model_lacks_tool_support(),
                tool_profile=tool_profile,
            )
            if self._looks_like_tool_overview(message):
                return self._format_tool_overview(tool_bundle)
            agent = self._get_or_build_agent(
                tool_bundle.profiles,
                tool_bundle.summary,
                tool_bundle.tools,
            )
            result = await self._invoke_agent_async(
                agent,
                message=message,
                previous_context=previous_context,
                system_prompt=system_prompt,
                core_memory=core_memory,
                kb_context=kb_context,
                tool_profile=tool_profile,
                tool_profiles=tool_bundle.profiles,
                tool_summary=tool_bundle.summary,
            )
            self._record_token_usage(
                result=result,
                user_id=user_id,
                system_prompt=system_prompt,
                core_memory=core_memory,
                kb_context=kb_context,
                previous_context=previous_context,
                user_message=message,
            )
            return self._extract_text(result) or ""
        except Exception as e:
            logger.error(f"LangChain agent 调用失败: {e}", exc_info=True)
            return USER_VISIBLE_AGENT_ERROR

    async def _invoke_agent_async(
        self,
        agent: Any,
        *,
        message: str,
        previous_context: Optional[List[Dict[str, Any]]],
        system_prompt: str,
        core_memory: Optional[str],
        kb_context: Optional[str],
        tool_profile: Optional[str],
        tool_profiles: List[str],
        tool_summary: str,
    ) -> Any:
        return await agent.ainvoke(
                {
                    "messages": self._build_messages(
                        message=message,
                        previous_context=previous_context,
                    )
                },
                context=AgentRunContext(
                    persona_prompt=system_prompt,
                    core_memory=core_memory,
                    kb_context=kb_context,
                    tool_profile=tool_profile,
                    tool_profiles=tool_profiles,
                    tool_summary=tool_summary,
                ),
                config={"recursion_limit": self.max_iterations},
            )

    def _get_or_build_agent(
        self,
        tool_profiles: List[str],
        tool_summary: str,
        tools,
    ):
        cache_key = self._build_agent_cache_key(tool_profiles, tools)

        with self._agent_cache_lock:
            cached_agent = self._agent_cache.get(cache_key)
        if cached_agent is not None:
            return cached_agent

        logger.info(
            "LangChain tools selected: tool_count=%s, profiles=%s, tools=%s",
            len(tools),
            tool_profiles,
            [tool.name for tool in tools],
        )
        agent = create_agent(
            model=self.model_client,
            tools=tools,
            system_prompt=self._build_static_system_prompt(tool_profiles, tool_summary),
            middleware=[
                self._build_dynamic_prompt_middleware(),
                self._build_model_middleware(),
                self._build_tool_middleware(),
            ],
            context_schema=AgentRunContext,
        )
        with self._agent_cache_lock:
            self._agent_cache[cache_key] = agent
        return agent

    def _build_static_system_prompt(
        self,
        tool_profiles: List[str],
        tool_summary: str,
    ) -> str:
        return self.prompt_manager.build_agent_system_prompt(
            tool_profiles=tool_profiles,
            tool_summary=tool_summary,
        )

    def _build_messages(
        self,
        *,
        message: str,
        previous_context: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        for item in previous_context or []:
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})
        return messages

    def _build_agent_cache_key(self, tool_profiles: List[str], tools) -> Tuple[str, ...]:
        tool_names = tuple(tool.name for tool in tools)
        return tuple(tool_profiles) + ("|",) + tool_names

    def _build_dynamic_prompt_middleware(self):
        prompt_manager = self.prompt_manager

        @dynamic_prompt
        def runtime_prompt(request) -> str:
            context = request.runtime.context
            return prompt_manager.build_runtime_prompt(
                persona_prompt=context.persona_prompt,
                tool_profile=context.tool_profile,
                tool_profiles=context.tool_profiles,
                tool_summary=context.tool_summary,
                core_memory=context.core_memory,
                kb_context=context.kb_context,
            )

        return runtime_prompt

    def _build_tool_middleware(self):
        @wrap_tool_call
        async def managed_tool_call(request, handler):
            tool_name = request.tool_call.get("name", "<unknown>")
            tool_args = request.tool_call.get("args", {})
            logger.info(
                "Tool call start: name=%s args=%s",
                tool_name,
                self._summarize_tool_args(tool_args),
            )
            try:
                response = await handler(request)
            except Exception as exc:
                logger.warning("Tool call failed: name=%s error=%s", tool_name, exc)
                return ToolMessage(
                    content=f"工具 {tool_name} 执行失败：{exc}",
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )

            if not isinstance(response, ToolMessage):
                logger.info("Tool call end: name=%s result_type=%s", tool_name, type(response).__name__)
                return response

            content = self._extract_tool_message_text(response)
            logger.info(
                "Tool call end: name=%s status=%s content_chars=%s",
                tool_name,
                response.status,
                len(content),
            )
            if len(content) <= MAX_TOOL_RESULT_CHARS:
                return response

            return ToolMessage(
                content=self._truncate_tool_text(content),
                tool_call_id=response.tool_call_id,
                status=response.status,
                artifact=response.artifact,
                name=response.name,
                id=response.id,
            )

        return managed_tool_call

    def _build_model_middleware(self):
        @wrap_model_call
        async def managed_model_call(request, handler):
            model_name = getattr(request.model, "model_name", None) or getattr(request.model, "model", None) or self.model
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
                logger.warning("Model call failed: model=%s duration_ms=%.2f error=%s", model_name, duration * 1000, exc)
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

    def _summarize_tool_args(self, tool_args: Any) -> str:
        raw = str(tool_args)
        if len(raw) <= MAX_TOOL_ARG_CHARS:
            return raw
        return raw[:MAX_TOOL_ARG_CHARS] + "...[truncated]"

    def _extract_tool_message_text(self, message: ToolMessage) -> str:
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

    def _truncate_tool_text(self, text: str) -> str:
        if len(text) <= MAX_TOOL_RESULT_CHARS:
            return text
        return text[:MAX_TOOL_RESULT_CHARS] + "\n\n[工具输出过长，已截断]"

    def _looks_like_tool_overview(self, message: str) -> bool:
        normalized = (message or "").strip()
        return any(hint in normalized for hint in TOOL_OVERVIEW_HINTS)

    def _format_tool_overview(self, tool_bundle) -> str:
        tool_names = [tool.name for tool in tool_bundle.tools]
        profile_text = "、".join(tool_bundle.profiles) if tool_bundle.profiles else "无"
        lines = [
            "我刚检查了当前这条消息下启用的工具。",
            "",
            "当前工具组：",
            profile_text,
            "",
            "当前可用工具：",
        ]
        for name in tool_names:
            lines.append(f"- {name}")
        if tool_bundle.summary_lines:
            lines.append("")
            lines.append("这些工具当前分别能做：")
            lines.extend(tool_bundle.summary_lines)
        return "\n".join(lines)

    def _extract_text(self, result: Any) -> str:
        if isinstance(result, dict):
            messages = result.get("messages") or []
            for message in reversed(messages):
                if getattr(message, "type", "") != "ai":
                    continue
                content = getattr(message, "content", "")
                if isinstance(content, str):
                    return content.strip()
                if isinstance(content, list):
                    parts = [item.get("text", "") for item in content if isinstance(item, dict)]
                    return "\n".join(part for part in parts if part).strip()
        return str(result).strip()

    def _record_token_usage(
        self,
        *,
        result: Any,
        user_id: str,
        system_prompt: str,
        core_memory: Optional[str],
        kb_context: Optional[str],
        previous_context: Optional[List[Dict[str, Any]]],
        user_message: str,
    ) -> None:
        usage = self._collect_usage_metadata(result)
        if not usage:
            logger.debug("LangChain agent 未返回 token usage 元数据")
            return

        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        if prompt_tokens <= 0 and completion_tokens <= 0:
            return

        token_monitor.record(
            user_id=user_id,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            request_type="agent",
            system_prompt_tokens=self._estimate_tokens(system_prompt),
            core_memory_tokens=self._estimate_tokens(core_memory),
            kb_context_tokens=self._estimate_tokens(kb_context),
            chat_history_tokens=self._estimate_message_tokens(previous_context),
            user_message_tokens=self._estimate_tokens(user_message),
        )

    def _collect_usage_metadata(self, result: Any) -> Optional[Dict[str, int]]:
        if not isinstance(result, dict):
            return None

        total_prompt = 0
        total_completion = 0
        found_usage = False

        for message in result.get("messages") or []:
            if getattr(message, "type", "") != "ai":
                continue

            usage = self._normalize_usage_dict(message)
            if not usage:
                continue

            total_prompt += usage["prompt_tokens"]
            total_completion += usage["completion_tokens"]
            found_usage = True

        if not found_usage:
            return None

        return {
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
        }

    def _normalize_usage_dict(self, message: Any) -> Optional[Dict[str, int]]:
        usage_sources = [
            getattr(message, "usage_metadata", None),
            getattr(message, "response_metadata", None),
        ]

        for usage_source in usage_sources:
            usage = self._extract_usage_fields(usage_source)
            if usage:
                return usage
        return None

    def _extract_usage_fields(self, source: Any) -> Optional[Dict[str, int]]:
        if not isinstance(source, dict):
            return None

        if "prompt_tokens" in source or "completion_tokens" in source:
            return {
                "prompt_tokens": int(source.get("prompt_tokens", 0)),
                "completion_tokens": int(source.get("completion_tokens", 0)),
            }

        if "input_tokens" in source or "output_tokens" in source:
            return {
                "prompt_tokens": int(source.get("input_tokens", 0)),
                "completion_tokens": int(source.get("output_tokens", 0)),
            }

        token_usage = source.get("token_usage")
        if isinstance(token_usage, dict):
            return {
                "prompt_tokens": int(token_usage.get("prompt_tokens", 0)),
                "completion_tokens": int(token_usage.get("completion_tokens", 0)),
            }

        return None

    def _estimate_tokens(self, text: Optional[str]) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _estimate_message_tokens(self, messages: Optional[List[Dict[str, Any]]]) -> int:
        if not messages:
            return 0

        total_chars = 0
        for item in messages:
            total_chars += len(str(item.get("content", "")).strip())

        if total_chars <= 0:
            return 0
        return max(1, total_chars // 4)

    def _model_lacks_tool_support(self) -> bool:
        model_name = self.model.lower()
        return any(item in model_name for item in MODELS_WITHOUT_TOOL_SUPPORT)

    def apply_pending_change(self, change_id: str, user_id: str) -> str:
        return self.tool_catalog.runtime.apply_pending_change(change_id, owner_user_id=user_id)

    def discard_pending_change(self, change_id: str, user_id: str) -> str:
        return self.tool_catalog.runtime.discard_pending_change(change_id, owner_user_id=user_id)

    def confirm_pending_command(self, confirm_id: str, user_id: str) -> str:
        return self.tool_catalog.runtime.confirm_pending_command(confirm_id, owner_user_id=user_id)

    def discard_pending_command(self, confirm_id: str, user_id: str) -> str:
        return self.tool_catalog.runtime.discard_pending_command(confirm_id, owner_user_id=user_id)

    def get_latest_pending_change_id(self, user_id: str) -> Optional[str]:
        return self.tool_catalog.runtime.get_latest_pending_change_id(owner_user_id=user_id)

    def get_latest_pending_command_id(self, user_id: str) -> Optional[str]:
        return self.tool_catalog.runtime.get_latest_pending_command_id(owner_user_id=user_id)
