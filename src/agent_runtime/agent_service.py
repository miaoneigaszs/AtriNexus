"""AtriNexus 自建 agent 服务。

替代 LangChain 的 `langchain.agents.create_agent`：用 PR11 的 OpenAICompatProvider
+ PR12 的 run_agent_loop 串起来。

对外接口与原 LangChainAgentService 保持一致：
- generate_reply / generate_reply_async — 主入口
- is_running / abort / queue_follow_up / drain_follow_up — run 控制
- apply_pending_change / discard_pending_change / confirm_pending_command /
  discard_pending_command / get_latest_pending_change_id /
  get_latest_pending_command_id — 转发给 WorkspaceRuntime
- tool_catalog / tool_guard / hooks / context_engine — 公开属性

行为差异（PR12 落地）：
- 不再依赖 langchain / langchain-openai / langchain-core
- 不再走 middleware 装配；hook 由 agent_loop 直接调用
- prompt caching 在 transform_context hook 中应用 dict 形式消息，对 Anthropic 路径
  会真正生效（OpenAI 兼容代理对未知字段一般透传或忽略，不影响主流路径）
- mid-loop context 压缩成为可能（每轮 invoke 前都会 preflight）
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from data.config import config
from src.agent_runtime.agent_loop import LoopResult, ToolEvent, run_agent_loop
from src.agent_runtime.agent_tool_guard import AgentToolGuard
from src.agent_runtime.context_engine import ContextEngine, DefaultCompressor
from src.agent_runtime.default_hooks import DefaultAgentHooks
from src.agent_runtime.hooks import AgentHooks
from src.agent_runtime.tool_catalog import ToolCatalog
from src.agent_runtime.user_runtime import UserRuntimeRegistry
from src.agent_runtime.user_runtime import user_runtime as default_user_runtime
from src.agent_runtime.trajectory import record_turn as record_trajectory_turn
from src.ai.providers.base import ProviderAdapter
from src.ai.providers.openai_compat import OpenAICompatProvider
from src.ai.types import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    UserMessage,
)
from src.platform_core.token_monitor import token_monitor


logger = logging.getLogger("wecom")

USER_VISIBLE_AGENT_ERROR = "抱歉，我暂时无法处理你的消息，请稍后再试。"
USER_VISIBLE_AGENT_CANCELLED = "已取消当前处理。"
MODELS_WITHOUT_TOOL_SUPPORT = {"deepseek-reasoner", "deepseek-r1"}

MAX_HISTORY_MESSAGES = 8
MAX_HISTORY_MESSAGE_CHARS = 800


class AgentService:
    """自建 agent 服务，去 LangChain 后的主入口。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        rag_service: Optional[object] = None,
        hooks: Optional[AgentHooks] = None,
        runtime_registry: Optional[UserRuntimeRegistry] = None,
        context_engine: Optional[ContextEngine] = None,
        provider: Optional[ProviderAdapter] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.runtime_registry = runtime_registry or default_user_runtime
        self.context_engine: ContextEngine = context_engine or DefaultCompressor(
            context_length=int(os.getenv("ATRINEXUS_AGENT_CONTEXT_LENGTH", "32000"))
        )
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
        self.hooks: AgentHooks = hooks or DefaultAgentHooks(self.tool_guard)
        self.provider: ProviderAdapter = provider or OpenAICompatProvider(
            api_key=api_key,
            base_url=base_url,
        )

    # ── 主入口 ──────────────────────────────────────────────────────────

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
        current_mode: Optional[str] = None,
    ) -> str:
        """跑一次 agent。外部调用方必须先确认 user 没有活跃 run（is_running 为 False）。"""
        async with self.runtime_registry.claim_run(user_id):
            try:
                tool_bundle = self.tool_catalog.build_tool_bundle(
                    user_id=user_id,
                    message=message,
                    allow_tools=not self._model_lacks_tool_support(),
                    tool_profile=tool_profile,
                )

                full_system_prompt = self._merge_system_prompt(
                    system_prompt,
                    core_memory,
                    tool_bundle,
                    tool_profile=tool_profile,
                    current_mode=current_mode,
                    user_id=user_id,
                )
                initial_messages = self._build_initial_messages(message, previous_context)

                loop_state_token = self.tool_guard.set_loop_state(self.tool_guard.create_loop_state())
                try:
                    result = await run_agent_loop(
                        provider=self.provider,
                        model=self.model,
                        system_prompt=full_system_prompt,
                        initial_messages=initial_messages,
                        tools=tool_bundle.tools,
                        hooks=self.hooks,
                        context_engine=self.context_engine,
                        max_iterations=self.max_iterations,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                finally:
                    self.tool_guard.reset_loop_state(loop_state_token)

                if result.cancelled:
                    return USER_VISIBLE_AGENT_CANCELLED

                self._record_token_usage(
                    result=result,
                    user_id=user_id,
                    system_prompt=full_system_prompt,
                    core_memory=core_memory,
                    previous_context=previous_context,
                    user_message=message,
                )
                self._record_trajectory(
                    user_id=user_id,
                    user_message=message,
                    assistant_reply=result.text,
                    system_prompt=full_system_prompt,
                    tool_events=result.tool_events,
                )
                return result.text or ""
            except asyncio.CancelledError:
                logger.info("Agent 运行被取消: user=%s", user_id)
                return USER_VISIBLE_AGENT_CANCELLED
            except Exception as exc:
                logger.error("Agent 调用失败: %s", exc, exc_info=True)
                return USER_VISIBLE_AGENT_ERROR

    # ── Run 控制对外接口 ───────────────────────────────────────────────

    async def is_running(self, user_id: str) -> bool:
        return await self.runtime_registry.is_running(user_id)

    async def abort(self, user_id: str) -> bool:
        return await self.runtime_registry.abort(user_id)

    async def queue_follow_up(self, user_id: str, message: str) -> int:
        return await self.runtime_registry.queue_follow_up(user_id, message)

    async def drain_follow_up(self, user_id: str) -> List[str]:
        return await self.runtime_registry.drain_follow_up(user_id)

    # ── pending change/command 转发（保留 message_handler 现有 API） ─

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

    # ── 内部 ──────────────────────────────────────────────────────────

    def _model_lacks_tool_support(self) -> bool:
        model_name = self.model.lower()
        return any(item in model_name for item in MODELS_WITHOUT_TOOL_SUPPORT)

    def _merge_system_prompt(
        self,
        persona_prompt: str,
        core_memory: Optional[str],
        tool_bundle,
        *,
        tool_profile: Optional[str] = None,
        current_mode: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """把静态壳 + 风格 + 核心记忆 + 当前工具组拼成最终 system prompt。

        替代旧版 dynamic_prompt middleware 的拼装逻辑——但只在调用时拼一次。
        把 `当前模式 / 工具档位 / 详细工具清单 / 当前待办` 渲染进 runtime 段，
        agent 被问到 "当前是什么模式"/"有哪些工具" 时直接从 system prompt 答，
        同时看得见自己的待办进度。
        """
        from src.prompting.prompt_manager import PromptManager
        from src.agent_runtime.todo_store import todo_store
        pm = PromptManager(self.workspace_root)
        static_prefix = pm.build_agent_static_prompt()
        todo_snapshot = todo_store.render(user_id) if user_id else ""
        runtime_prefix = pm.build_runtime_prompt(
            persona_prompt=persona_prompt or "",
            tool_profile=tool_profile,
            tool_profiles=tool_bundle.profiles,
            tool_summary=tool_bundle.detailed_summary,
            core_memory=core_memory,
            current_mode=current_mode,
            todo_snapshot=todo_snapshot,
        )
        parts = [static_prefix]
        if runtime_prefix:
            parts.append(runtime_prefix)
        return "\n\n".join(p for p in parts if p)

    def _build_initial_messages(
        self,
        message: str,
        previous_context: Optional[List[Dict[str, Any]]],
    ) -> List[Message]:
        msgs: List[Message] = []
        history_items = list(previous_context or [])[-MAX_HISTORY_MESSAGES:]
        for item in history_items:
            role = str(item.get("role", "")).strip()
            content = self._truncate(str(item.get("content", "")).strip())
            if not role or not content:
                continue
            if role == "user":
                msgs.append(UserMessage(content=content))
            elif role == "assistant":
                msgs.append(AssistantMessage(content=content))
            elif role == "tool":
                msgs.append(ToolResultMessage(
                    tool_call_id=str(item.get("tool_call_id", "")),
                    content=content,
                    name=item.get("name"),
                ))
            # 其他 role 跳过
        msgs.append(UserMessage(content=message))
        return msgs

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= MAX_HISTORY_MESSAGE_CHARS:
            return text
        return text[: MAX_HISTORY_MESSAGE_CHARS - 17].rstrip() + "\n[内容已截断]"

    def _record_token_usage(
        self,
        *,
        result: LoopResult,
        user_id: str,
        system_prompt: str,
        core_memory: Optional[str],
        previous_context: Optional[List[Dict[str, Any]]],
        user_message: str,
    ) -> None:
        usage = result.usage
        if usage.prompt_tokens <= 0 and usage.completion_tokens <= 0:
            return
        token_monitor.record(
            user_id=user_id,
            model=self.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            request_type="agent",
            system_prompt_tokens=_estimate(system_prompt),
            core_memory_tokens=_estimate(core_memory),
            chat_history_tokens=_estimate_messages(previous_context),
            user_message_tokens=_estimate(user_message),
        )

    def _record_trajectory(
        self,
        *,
        user_id: str,
        user_message: str,
        assistant_reply: str,
        system_prompt: str,
        tool_events: List[ToolEvent],
    ) -> None:
        try:
            record_trajectory_turn(
                user_id=user_id,
                user_message=user_message,
                assistant_reply=assistant_reply,
                model=self.model,
                system_prompt=system_prompt,
                tool_events=[
                    {
                        "name": ev.name,
                        "args": ev.args,
                        "result": ev.result,
                        "status": ev.status,
                    }
                    for ev in tool_events
                ],
                completed=bool(assistant_reply),
                fast_path_hit=False,
                intent="none",
            )
        except Exception as exc:
            logger.debug("trajectory 记录失败: %s", exc)


# ── token 估算工具（与 token_monitor 口径一致） ─────────────────────────


def _estimate(text: Optional[str]) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _estimate_messages(messages: Optional[List[Dict[str, Any]]]) -> int:
    if not messages:
        return 0
    total = 0
    for item in list(messages)[-MAX_HISTORY_MESSAGES:]:
        content = str(item.get("content", "")).strip()
        if len(content) > MAX_HISTORY_MESSAGE_CHARS:
            content = content[:MAX_HISTORY_MESSAGE_CHARS]
        total += len(content)
    if total <= 0:
        return 0
    return max(1, total // 4)


# ── 兼容旧名 ────────────────────────────────────────────────────────────


# 旧 import 路径仍能用：from src.agent_runtime.agent_service import LangChainAgentService
LangChainAgentService = AgentService


__all__ = ["AgentService", "LangChainAgentService"]
