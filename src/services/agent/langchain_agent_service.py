from __future__ import annotations

import asyncio
import contextvars
import difflib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, wrap_model_call, wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI

from data.config import config
from src.services.agent.agent_context import AgentRunContext
from src.services.agent.tool_catalog import ToolCatalog
from src.services.agent.agent_usage import (
    MAX_HISTORY_MESSAGES,
    collect_usage_metadata,
    estimate_message_tokens,
    estimate_tokens,
    extract_text,
    truncate_message_content,
)
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
MAX_TOOL_REPEAT_COUNT = 2
MAX_TOOL_HISTORY = 30
WORKSPACE_PATH_TOOL_KEYS = {"path", "source_path", "target_path"}
RUN_COMMAND_TOOL_NAME = "run_command"
READ_FILE_TOOL_NAME = "read_file"
LIST_DIRECTORY_TOOL_NAME = "list_directory"
SEARCH_FILES_TOOL_NAME = "search_files"
EXPLORATION_TOOL_NAMES = {LIST_DIRECTORY_TOOL_NAME, SEARCH_FILES_TOOL_NAME}
MAX_LIST_DIRECTORY_LINES = 40
MAX_SEARCH_RESULT_LINES = 20
MAX_READ_FILE_CHARS = 2500
TOOL_LOOP_STATE: contextvars.ContextVar[Dict[str, Any] | None] = contextvars.ContextVar(
    "tool_loop_state",
    default=None,
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
            return extract_text(result) or ""
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
        loop_state_token = TOOL_LOOP_STATE.set({"counts": {}, "recent": []})
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
                        kb_context=kb_context,
                        tool_profile=tool_profile,
                        tool_profiles=tool_profiles,
                        tool_summary=tool_summary,
                    ),
                    config={"recursion_limit": self.max_iterations},
                )
        finally:
            TOOL_LOOP_STATE.reset(loop_state_token)

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
            repaired_args, repair_message = self._repair_tool_args(tool_name, tool_args)
            if repaired_args != tool_args:
                request.tool_call["args"] = repaired_args
            tool_args = request.tool_call.get("args", {})

            validation_error = self._validate_tool_args(tool_name, tool_args)
            if validation_error:
                logger.warning(
                    "Tool call blocked by validation: name=%s args=%s error=%s",
                    tool_name,
                    self._summarize_tool_args(tool_args),
                    validation_error,
                )
                return ToolMessage(
                    content=validation_error,
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )

            loop_guard_message = self._check_tool_loop(tool_name, tool_args)
            if loop_guard_message:
                logger.warning("Tool call blocked by loop guard: name=%s args=%s", tool_name, self._summarize_tool_args(tool_args))
                return ToolMessage(
                    content=loop_guard_message,
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )

            logger.info(
                "Tool call start: name=%s args=%s%s",
                tool_name,
                self._summarize_tool_args(tool_args),
                f" repair={repair_message}" if repair_message else "",
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
            content = self._shape_tool_result(tool_name, tool_args, content)
            response = self._replace_tool_message_content(response, content)
            self._record_tool_outcome(tool_name, tool_args, response.status, content)
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

    def _repair_tool_args(self, tool_name: str, tool_args: Any) -> Tuple[Any, str]:
        if not isinstance(tool_args, dict):
            return tool_args, ""

        repaired = dict(tool_args)
        notes: List[str] = []

        if tool_name == RUN_COMMAND_TOOL_NAME:
            command = repaired.get("command")
            if isinstance(command, str):
                stripped = command.strip()
                if stripped != command:
                    repaired["command"] = stripped
                    notes.append("trim-command")
            return repaired, ", ".join(notes)

        if tool_name not in {
            "list_directory",
            "read_file",
            "preview_write_file",
            "preview_edit_file",
            "preview_append_file",
            "rename_path",
        }:
            return repaired, ""

        for key in WORKSPACE_PATH_TOOL_KEYS:
            value = repaired.get(key)
            if not isinstance(value, str):
                continue
            candidate, note = self._repair_workspace_path_argument(
                value,
                expect_dir=(tool_name == "list_directory" and key == "path"),
                allow_dir=(tool_name == "rename_path"),
            )
            if note:
                notes.append(f"{key}:{note}")
            if candidate != value:
                repaired[key] = candidate

        return repaired, ", ".join(notes)

    def _repair_workspace_path_argument(
        self,
        value: str,
        *,
        expect_dir: bool,
        allow_dir: bool,
    ) -> Tuple[str, str]:
        normalized = str(value).strip().strip("`'\"“”‘’")
        if not normalized:
            return value, ""
        if normalized.lower() == "readme":
            return "README.md", "readme"

        runtime = self.tool_catalog.runtime
        candidate, error = runtime._resolve_path_or_error(normalized)
        if not error and candidate and candidate.exists():
            if expect_dir and candidate.is_dir():
                return normalized, ""
            if not expect_dir and (candidate.is_file() or (allow_dir and candidate.is_dir())):
                return normalized, ""

        file_candidate = self._find_workspace_candidate(normalized, expect_dir=expect_dir, allow_dir=allow_dir)
        if file_candidate:
            return file_candidate, "path-repaired"
        return normalized, ""

    def _find_workspace_candidate(
        self,
        value: str,
        *,
        expect_dir: bool,
        allow_dir: bool,
    ) -> Optional[str]:
        runtime = self.tool_catalog.runtime
        query = value.replace("\\", "/").strip().strip("/")
        if not query:
            return None
        query_name = Path(query).name
        query_key = self._normalize_lookup_key(query_name)
        if not query_key:
            return None

        candidates: List[Tuple[float, str]] = []
        if not expect_dir:
            for file_path in runtime._iter_files(runtime.workspace_root):
                relative_path = runtime._to_relative(file_path)
                score = self._score_workspace_candidate(query_key, file_path.name, relative_path)
                if score >= 0.86:
                    candidates.append((score, relative_path))

        if expect_dir or allow_dir:
            for current_root, dirnames, _ in os.walk(runtime.workspace_root):
                dirnames[:] = [name for name in dirnames if name not in runtime.SKIP_DIRS]
                for dirname in dirnames:
                    relative_path = str(
                        Path(
                            os.path.relpath(
                                os.path.join(current_root, dirname),
                                runtime.workspace_root,
                            )
                        )
                    )
                    score = self._score_workspace_candidate(query_key, dirname, relative_path)
                    if score >= 0.86:
                        candidates.append((score, relative_path))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (-item[0], len(item[1])))
        best_score, best_path = candidates[0]
        if len(candidates) > 1 and abs(best_score - candidates[1][0]) < 0.03:
            return None
        return best_path

    def _score_workspace_candidate(self, query_key: str, name: str, relative_path: str) -> float:
        name_key = self._normalize_lookup_key(name)
        path_key = self._normalize_lookup_key(relative_path)
        score = max(
            difflib.SequenceMatcher(None, query_key, name_key).ratio(),
            difflib.SequenceMatcher(None, query_key, path_key).ratio(),
        )
        if query_key == name_key:
            return 1.0
        if name_key.startswith(query_key):
            return max(score, 0.93)
        if query_key in name_key:
            return max(score, 0.88)
        return score

    def _normalize_lookup_key(self, value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")

    def _check_tool_loop(self, tool_name: str, tool_args: Any) -> Optional[str]:
        loop_state = TOOL_LOOP_STATE.get()
        if loop_state is None:
            return None

        signature = self._build_tool_signature(tool_name, tool_args)
        counts = loop_state.setdefault("counts", {})
        recent = loop_state.setdefault("recent", [])
        counts[signature] = int(counts.get(signature, 0)) + 1
        recent_signatures = loop_state.setdefault("recent_signatures", [])
        recent_signatures.append(signature)
        if len(recent_signatures) > MAX_TOOL_HISTORY:
            del recent_signatures[0]

        if counts[signature] > MAX_TOOL_REPEAT_COUNT:
            return (
                f"工具 {tool_name} 已重复尝试同样的参数多次，请停止绕圈，"
                "改用别的工具、直接给出结论，或先向用户确认目标。"
            )

        if len(recent_signatures) >= 4 and recent_signatures[-1] == recent_signatures[-3] and recent_signatures[-2] == recent_signatures[-4]:
            return (
                f"工具链出现来回循环：{tool_name}。请不要继续重复试探，"
                "直接总结当前已知信息，或先向用户确认下一步。"
            )

        if self._is_no_progress_tool_call(tool_name, tool_args, recent):
            return (
                f"工具 {tool_name} 继续试探也不会有新进展。"
                "请直接根据现有结果回答，或向用户确认具体目标。"
            )

        return None

    def _record_tool_outcome(self, tool_name: str, tool_args: Any, status: str, content: str) -> None:
        loop_state = TOOL_LOOP_STATE.get()
        if loop_state is None:
            return

        recent = loop_state.setdefault("recent", [])
        recent.append(
            {
                "tool_name": tool_name,
                "signature": self._build_tool_signature(tool_name, tool_args),
                "status": status,
                "path": self._extract_primary_path(tool_args),
                "content_hash": hash(content),
            }
        )
        if len(recent) > MAX_TOOL_HISTORY:
            del recent[0]

    def _is_no_progress_tool_call(
        self,
        tool_name: str,
        tool_args: Any,
        recent: List[Dict[str, Any]],
    ) -> bool:
        if tool_name not in EXPLORATION_TOOL_NAMES or not recent:
            return False

        recent_successes = [
            item
            for item in reversed(recent)
            if item.get("status") != "error"
        ]
        if not recent_successes:
            return False

        if any(item.get("tool_name") == READ_FILE_TOOL_NAME for item in recent_successes[:3]):
            return True

        recent_explorations = [
            item for item in recent_successes[:4] if item.get("tool_name") in EXPLORATION_TOOL_NAMES
        ]
        if len(recent_explorations) < 3:
            return False

        current_path = self._extract_primary_path(tool_args)
        previous_paths = {
            str(item.get("path", "")).strip()
            for item in recent_explorations
            if str(item.get("path", "")).strip()
        }
        if current_path and current_path in previous_paths:
            return True
        return False

    def _validate_tool_args(self, tool_name: str, tool_args: Any) -> Optional[str]:
        if not isinstance(tool_args, dict):
            return None

        if tool_name == SEARCH_FILES_TOOL_NAME:
            query = str(tool_args.get("query", "")).strip()
            normalized = "".join(ch for ch in query.lower() if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
            if not query:
                return "search_files 缺少搜索关键词。"
            if len(normalized) < 2 and "." not in query and "/" not in query and "\\" not in query:
                return "search_files 的关键词太短，请给更具体一点的内容。"

        if tool_name in {LIST_DIRECTORY_TOOL_NAME, READ_FILE_TOOL_NAME, "preview_write_file", "preview_edit_file", "preview_append_file"}:
            path = str(tool_args.get("path", "")).strip()
            if not path:
                return f"{tool_name} 缺少路径参数。"

        if tool_name == "rename_path":
            source_path = str(tool_args.get("source_path", "")).strip()
            target_path = str(tool_args.get("target_path", "")).strip()
            if not source_path or not target_path:
                return "rename_path 需要同时提供 source_path 和 target_path。"
            if source_path == target_path:
                return "rename_path 的源路径和目标路径相同，不需要重复执行。"

        if tool_name == "preview_edit_file":
            find_text = str(tool_args.get("find_text", ""))
            replace_text = str(tool_args.get("replace_text", ""))
            if not find_text.strip():
                return "preview_edit_file 缺少待替换文本。"
            if find_text == replace_text:
                return "preview_edit_file 的替换前后文本相同，不需要执行。"

        return None

    def _extract_primary_path(self, tool_args: Any) -> str:
        if not isinstance(tool_args, dict):
            return ""
        for key in ("path", "source_path", "target_path"):
            value = str(tool_args.get(key, "")).strip()
            if value:
                return value
        return ""

    def _shape_tool_result(self, tool_name: str, tool_args: Any, content: str) -> str:
        if tool_name == LIST_DIRECTORY_TOOL_NAME:
            return self._shape_directory_result(content)
        if tool_name == SEARCH_FILES_TOOL_NAME:
            return self._shape_search_result(content)
        if tool_name == READ_FILE_TOOL_NAME:
            return self._shape_read_file_result(content, self._extract_primary_path(tool_args))
        return content

    def _shape_directory_result(self, content: str) -> str:
        lines = content.splitlines()
        if len(lines) <= MAX_LIST_DIRECTORY_LINES:
            return content
        kept = lines[:MAX_LIST_DIRECTORY_LINES]
        omitted = max(0, len(lines) - len(kept))
        kept.append(f"... 其余 {omitted} 项已省略")
        return "\n".join(kept)

    def _shape_search_result(self, content: str) -> str:
        lines = content.splitlines()
        if len(lines) <= MAX_SEARCH_RESULT_LINES:
            return content
        kept = lines[:MAX_SEARCH_RESULT_LINES]
        kept.append(f"[匹配结果较多，仅保留前 {MAX_SEARCH_RESULT_LINES} 条；如需继续，请缩小路径或关键词]")
        return "\n".join(kept)

    def _shape_read_file_result(self, content: str, path: str) -> str:
        if len(content) <= MAX_READ_FILE_CHARS:
            return content

        header, separator, body = content.partition("\n\n")
        if not separator:
            return self._truncate_tool_text(content)

        trimmed_body = body[: MAX_READ_FILE_CHARS - len(header) - 120].rstrip()
        hint_path = path or "文件"
        return (
            f"{header}\n\n{trimmed_body}\n\n"
            f"[文件内容较长，已只保留前半段。若要继续，请指定更小范围，例如“{hint_path}最后一行”或“{hint_path} 某段内容”。]"
        )

    def _replace_tool_message_content(self, message: ToolMessage, content: str) -> ToolMessage:
        if message.content == content:
            return message
        return ToolMessage(
            content=content,
            tool_call_id=message.tool_call_id,
            status=message.status,
            artifact=message.artifact,
            name=message.name,
            id=message.id,
        )

    def _build_tool_signature(self, tool_name: str, tool_args: Any) -> str:
        try:
            serialized = json.dumps(tool_args, sort_keys=True, ensure_ascii=False, default=str)
        except TypeError:
            serialized = str(tool_args)
        return f"{tool_name}:{serialized}"

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
        if tool_bundle.detailed_summary_lines:
            lines.append("")
            lines.append("这些工具当前分别能做：")
            lines.extend(tool_bundle.detailed_summary_lines)
        return "\n".join(lines)

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
            kb_context_tokens=estimate_tokens(kb_context),
            chat_history_tokens=estimate_message_tokens(previous_context),
            user_message_tokens=estimate_tokens(user_message),
        )

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
