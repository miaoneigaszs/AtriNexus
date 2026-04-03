from __future__ import annotations

import logging
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
        self.workspace_root = str(Path(__file__).resolve().parents[3])

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
            agent = self._build_agent(system_prompt, core_memory, kb_context)
            result = agent.invoke({"messages": self._build_messages(message, previous_context)})
            return self._extract_text(result) or ""
        except Exception as e:
            logger.error(f"LangChain agent 调用失败: {e}", exc_info=True)
            return USER_VISIBLE_AGENT_ERROR

    def _build_agent(
        self,
        system_prompt: str,
        core_memory: Optional[str],
        kb_context: Optional[str],
    ):
        model = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return create_agent(
            model=model,
            tools=self._build_tools(),
            system_prompt=self._build_system_prompt(system_prompt, core_memory, kb_context),
        )

    def _build_tools(self):
        if self._model_lacks_tool_support():
            return []

        search_cfg = config.network_search
        search_api_key = search_cfg.api_key if search_cfg.search_enabled and search_cfg.api_key else None
        catalog = ToolCatalog(workspace_root=self.workspace_root, search_api_key=search_api_key)
        return catalog.build_tools()

    def _build_system_prompt(
        self,
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
