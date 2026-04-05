from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from data.config import config
from src.services.agent.tool_catalog import ToolCatalog

logger = logging.getLogger("wecom")

USER_VISIBLE_AGENT_ERROR = "抱歉，我暂时无法处理你的消息，请稍后再试。"
MODELS_WITHOUT_TOOL_SUPPORT = {"deepseek-reasoner", "deepseek-r1"}


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

    def get_response(
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
            agent = self._build_agent(
                message,
                system_prompt,
                core_memory,
                kb_context,
                tool_bundle.profiles,
                tool_bundle.summary,
                tool_bundle.tools,
            )
            result = agent.invoke(
                {"messages": self._build_messages(message, previous_context)},
                config={"recursion_limit": self.max_iterations},
            )
            return self._extract_text(result) or ""
        except Exception as e:
            logger.error(f"LangChain agent 调用失败: {e}", exc_info=True)
            return USER_VISIBLE_AGENT_ERROR

    def _build_agent(
        self,
        message: str,
        system_prompt: str,
        core_memory: Optional[str],
        kb_context: Optional[str],
        tool_profiles: List[str],
        tool_summary: str,
        tools,
    ):
        logger.info(
            "LangChain tools selected: message_len=%s, tool_count=%s, profiles=%s, tools=%s",
            len(message),
            len(tools),
            tool_profiles,
            [tool.name for tool in tools],
        )
        return create_agent(
            model=self.model_client,
            tools=tools,
            system_prompt=self._build_system_prompt(system_prompt, core_memory, kb_context, tool_profiles, tool_summary),
        )

    def _build_system_prompt(
        self,
        system_prompt: str,
        core_memory: Optional[str],
        kb_context: Optional[str],
        tool_profiles: List[str],
        tool_summary: str,
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
        prompt_parts.append(f"【当前工具组】\n{', '.join(tool_profiles) if tool_profiles else 'none'}")
        prompt_parts.append(f"【可用工具摘要】\n{tool_summary}")
        prompt_parts.append(
            "【工具使用指导】\n"
            "你拥有按需启用的工具组，包括 workspace 读取、命令执行、联网搜索和文件修改预览。\n"
            "只要任务涉及文件、目录、代码、命令执行，就应优先调用工具，而不是空谈步骤。\n"
            "如果用户要求修改文件，先读取必要内容，再使用 preview_write_file 或 preview_edit_file 生成修改预览。\n"
            "安全命令会直接执行；含 shell 操作符、未知可执行文件或高风险命令会进入确认流程；文件修改预览也需要用户后续确认才能真正落盘。"
        )
        prompt_parts.append(
            "【工具选择示例】\n"
            "- 用户说“看看 src 目录” -> 调用 list_directory。\n"
            "- 用户说“读一下 README.md” -> 调用 read_file。\n"
            "- 用户说“把 config.json 里某项改成 xxx” -> 先 read_file，再 preview_edit_file。\n"
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

    def _build_messages(
        self,
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
