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
from src.services.agent.tool_catalog import ToolCatalog
from src.services.token_monitor import token_monitor

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
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max(2, int(os.getenv("ATRINEXUS_AGENT_MAX_ITERATIONS", "8")))
        self.workspace_root = str(Path(__file__).resolve().parents[3])
        self.agent_prompt_path = Path(self.workspace_root) / "src" / "base" / "agent.md"
        search_cfg = config.network_search
        search_api_key = search_cfg.api_key if search_cfg.search_enabled and search_cfg.api_key else None
        self.tool_catalog = ToolCatalog(workspace_root=self.workspace_root, search_api_key=search_api_key)
        self.model_client = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self._agent_cache: Dict[Tuple[str, ...], Any] = {}
        self._agent_cache_lock = threading.Lock()
        self._agent_identity_prompt = self._load_agent_identity_prompt()

    def generate_reply(
        self,
        message: str,
        user_id: str,
        system_prompt: str,
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
        previous_context: Optional[List[Dict[str, Any]]] = None,
        core_memory: Optional[str] = None,
        kb_context: Optional[str] = None,
    ) -> str:
        try:
            tool_bundle = self.tool_catalog.build_tool_bundle(
                user_id=user_id,
                message=message,
                allow_tools=not self._model_lacks_tool_support(),
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
    ) -> Any:
        return await agent.ainvoke(
                {
                    "messages": self._build_messages(
                        message=message,
                        previous_context=previous_context,
                        system_prompt=system_prompt,
                        core_memory=core_memory,
                        kb_context=kb_context,
                    )
                },
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
        )
        with self._agent_cache_lock:
            self._agent_cache[cache_key] = agent
        return agent

    def _build_static_system_prompt(
        self,
        tool_profiles: List[str],
        tool_summary: str,
    ) -> str:
        prompt_parts: List[str] = []
        if self._agent_identity_prompt:
            prompt_parts.append(f"【Agent 认知】\n{self._agent_identity_prompt}")
        prompt_parts.append(f"【当前工具组】\n{', '.join(tool_profiles) if tool_profiles else 'none'}")
        prompt_parts.append(f"【可用工具摘要】\n{tool_summary}")
        prompt_parts.append(
            "【工具使用指导】\n"
            "你拥有按需启用的工具组，包括 workspace 读取、命令执行、联网搜索和文件修改预览。\n"
            "只要任务涉及文件、目录、代码、命令执行，就应优先调用工具，而不是空谈步骤。\n"
            "如果用户要求修改文件，先读取必要内容，再使用 preview_write_file 或 preview_edit_file 生成修改预览。\n"
            "Markdown、README、说明文档、白皮书、报告这类文档同样允许修改，不要因为它们是文档就只读不改。\n"
            "如果局部替换难以精确命中，就先 read_file，再直接用 preview_write_file 生成整份文件的新版本。\n"
            "安全命令会直接执行；含 shell 操作符、未知可执行文件或高风险命令会进入确认流程；文件修改预览也需要用户后续确认才能真正落盘。"
        )
        prompt_parts.append(
            "【工具选择示例】\n"
            "- 用户说“看看 src 目录” -> 调用 list_directory。\n"
            "- 用户说“读一下 README.md” -> 调用 read_file。\n"
            "- 用户说“把 config.json 里某项改成 xxx” -> 先 read_file，再 preview_edit_file。\n"
            "- 用户说“把 README 第一段改短一点” -> 先 read_file，再 preview_edit_file；如果难以精确替换，就用 preview_write_file。\n"
            "- 用户说“把这篇 Markdown 文档整体重写得更清楚” -> 先 read_file，再用 preview_write_file。\n"
            "- 用户说“执行 git status” -> 调用 run_command。\n"
            "- 用户问“今天星期几” -> 调用 get_current_time。"
        )
        prompt_parts.append(
            "【执行类回复风格】\n"
            "当你在执行命令、读取文件、搜索代码或生成修改预览时，必须使用直接、简洁、结果导向的表达。\n"
            "不要使用“主人”等称呼，不要撒娇，不要附加人格化感叹，不要添加与结果无关的修饰句。\n"
            "优先回答这几项：做了什么、结果是什么、是否成功；必要时再补一行关键细节。"
        )
        return "\n\n".join(prompt_parts)

    def _load_agent_identity_prompt(self) -> str:
        try:
            return self.agent_prompt_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            logger.warning("Agent 认知提示词不存在: %s", self.agent_prompt_path)
            return ""
        except Exception as exc:
            logger.warning("加载 Agent 认知提示词失败: %s", exc)
            return ""

    def _build_messages(
        self,
        *,
        message: str,
        previous_context: Optional[List[Dict[str, Any]]],
        system_prompt: str,
        core_memory: Optional[str],
        kb_context: Optional[str],
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        dynamic_system_prompt = self._build_dynamic_system_prompt(
            system_prompt=system_prompt,
            core_memory=core_memory,
            kb_context=kb_context,
        )
        if dynamic_system_prompt:
            messages.append({"role": "system", "content": dynamic_system_prompt})
        for item in previous_context or []:
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})
        return messages

    def _build_dynamic_system_prompt(
        self,
        *,
        system_prompt: str,
        core_memory: Optional[str],
        kb_context: Optional[str],
    ) -> str:
        prompt_parts: List[str] = []
        if system_prompt:
            prompt_parts.append(f"【角色设定】\n{system_prompt}")
        if core_memory:
            prompt_parts.append(f"【核心记忆】\n{core_memory}")
        if kb_context:
            prompt_parts.append(
                "【参考资料】\n"
                f"{kb_context}\n"
                "必须结合上述参考资料，并严格保持你的【角色设定】来回答用户的问题。"
                "如果参考资料无相关性，请忽略资料，自然回复即可。"
            )
        return "\n\n".join(prompt_parts)

    def _build_agent_cache_key(self, tool_profiles: List[str], tools) -> Tuple[str, ...]:
        tool_names = tuple(tool.name for tool in tools)
        return tuple(tool_profiles) + ("|",) + tool_names

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
