from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from data.config import config
from src.agent_runtime.agent_context import AgentRunContext
from src.agent_runtime.agent_tool_guard import AgentToolGuard
from src.agent_runtime.middleware import (
    build_dynamic_prompt_middleware,
    build_model_middleware,
)
from src.agent_runtime.tool_catalog import ToolCatalog
from src.agent_runtime.agent_usage import (
    MAX_HISTORY_MESSAGES,
    collect_usage_metadata,
    estimate_message_tokens,
    estimate_tokens,
    extract_text,
    truncate_message_content,
)
from src.prompting.prompt_manager import PromptManager
from src.platform_core.token_monitor import token_monitor
from src.agent_runtime.trajectory import record_turn as record_trajectory_turn

logger = logging.getLogger("wecom")

USER_VISIBLE_AGENT_ERROR = "抱歉，我暂时无法处理你的消息，请稍后再试。"
MODELS_WITHOUT_TOOL_SUPPORT = {"deepseek-reasoner", "deepseek-r1"}


def _extract_tool_events(result: Any) -> List[Dict[str, Any]]:
    """从 agent 返回里提取工具调用序列用于 trajectory。失败时返回空列表。"""
    events: List[Dict[str, Any]] = []
    if not isinstance(result, dict):
        return events
    messages = result.get("messages") or []
    pending_calls: Dict[str, Dict[str, Any]] = {}
    for message in messages:
        msg_type = getattr(message, "type", "")
        if msg_type == "ai":
            for call in getattr(message, "tool_calls", None) or []:
                call_id = call.get("id") or f"{call.get('name', '')}:{len(pending_calls)}"
                pending_calls[call_id] = {
                    "name": call.get("name", ""),
                    "args": call.get("args", {}),
                }
        elif msg_type == "tool":
            call_id = getattr(message, "tool_call_id", "") or ""
            base = pending_calls.pop(call_id, {"name": getattr(message, "name", ""), "args": {}})
            events.append({
                **base,
                "result": str(getattr(message, "content", "")),
                "status": getattr(message, "status", "ok"),
            })
    return events


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
        # OpenClaw 的默认策略不是用很小的总轮数硬截停，而是给 loop detection
        # 留足历史窗口。这里把默认 recursion_limit 提高到 12，避免简单文件任务在
        # 完成最后一次 read/search 后还没来得及收尾就被截断。
        self.max_iterations = max(2, int(os.getenv("ATRINEXUS_AGENT_MAX_ITERATIONS", "12")))
        self.workspace_root = str(Path(__file__).resolve().parents[3])
        search_cfg = config.network_search
        search_api_key = search_cfg.api_key if search_cfg.search_enabled and search_cfg.api_key else None
        self.tool_catalog = ToolCatalog(
            workspace_root=self.workspace_root,
            search_api_key=search_api_key,
            rag_service=rag_service,
        )
        self.tool_guard = AgentToolGuard(self.tool_catalog)
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
    ) -> str:
        try:
            tool_bundle = self.tool_catalog.build_tool_bundle(
                user_id=user_id,
                message=message,
                allow_tools=not self._model_lacks_tool_support(),
                tool_profile=tool_profile,
            )
            if self.tool_catalog.looks_like_tool_overview(message):
                return self.tool_catalog.format_tool_overview(tool_bundle)
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
                tool_profile=tool_profile,
                tool_profiles=tool_bundle.profiles,
                tool_summary=tool_bundle.summary,
            )
            self._record_token_usage(
                result=result,
                user_id=user_id,
                system_prompt=system_prompt,
                core_memory=core_memory,
                previous_context=previous_context,
                user_message=message,
            )
            reply_text = extract_text(result) or ""
            self._record_trajectory(
                user_id=user_id,
                user_message=message,
                assistant_reply=reply_text,
                system_prompt=system_prompt,
                result=result,
            )
            return reply_text
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
        tool_profile: Optional[str],
        tool_profiles: List[str],
        tool_summary: str,
    ) -> Any:
        loop_state_token = self.tool_guard.set_loop_state(self.tool_guard.create_loop_state())
        try:
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
                        tool_profile=tool_profile,
                        tool_profiles=tool_profiles,
                        tool_summary=tool_summary,
                    ),
                    config={"recursion_limit": self.max_iterations},
                )
        finally:
            self.tool_guard.reset_loop_state(loop_state_token)

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
            system_prompt=self.prompt_manager.build_agent_static_prompt(),
            middleware=[
                build_dynamic_prompt_middleware(self.prompt_manager),
                build_model_middleware(self.model),
                self.tool_guard.build_tool_middleware(),
            ],
            context_schema=AgentRunContext,
        )
        with self._agent_cache_lock:
            self._agent_cache[cache_key] = agent
        return agent

    def _build_messages(
        self,
        *,
        message: str,
        previous_context: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        history_items = list(previous_context or [])[-MAX_HISTORY_MESSAGES:]
        for item in history_items:
            role = str(item.get("role", "")).strip()
            content = truncate_message_content(str(item.get("content", "")).strip())
            if role and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})
        return messages

    def _build_agent_cache_key(self, tool_profiles: List[str], tools) -> Tuple[str, ...]:
        tool_names = tuple(tool.name for tool in tools)
        return tuple(tool_profiles) + ("|",) + tool_names

    def _record_token_usage(
        self,
        *,
        result: Any,
        user_id: str,
        system_prompt: str,
        core_memory: Optional[str],
        previous_context: Optional[List[Dict[str, Any]]],
        user_message: str,
    ) -> None:
        usage = collect_usage_metadata(result)
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
            system_prompt_tokens=estimate_tokens(system_prompt),
            core_memory_tokens=estimate_tokens(core_memory),
            chat_history_tokens=estimate_message_tokens(previous_context),
            user_message_tokens=estimate_tokens(user_message),
        )

    def _model_lacks_tool_support(self) -> bool:
        model_name = self.model.lower()
        return any(item in model_name for item in MODELS_WITHOUT_TOOL_SUPPORT)

    def _record_trajectory(
        self,
        *,
        user_id: str,
        user_message: str,
        assistant_reply: str,
        system_prompt: str,
        result: Any,
    ) -> None:
        """把本轮对话以 ShareGPT 形式追加到 trajectory 文件（未启用时直接返回）。"""
        try:
            tool_events = _extract_tool_events(result)
            record_trajectory_turn(
                user_id=user_id,
                user_message=user_message,
                assistant_reply=assistant_reply,
                model=self.model,
                system_prompt=system_prompt,
                tool_events=tool_events,
                completed=bool(assistant_reply),
            )
        except Exception as exc:
            logger.debug("trajectory 记录失败: %s", exc)

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
