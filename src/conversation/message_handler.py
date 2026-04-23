"""
WeCom 消息处理器（重构版）
负责消息处理流程的编排，将具体逻辑委托给各个处理器
"""

import logging
import os
from typing import List

from sqlalchemy.exc import IntegrityError

from src.ai.llm_service import LLMService
from src.agent_runtime.agent_service import AgentService
from src.agent_runtime.trajectory import record_fast_path_turn
from src.memory.memory_manager import MemoryManager
from src.knowledge.rag_service import SdkRAGService
from src.platform_core.session_service import SessionService
from src.platform_core.vector_store import QdrantVectorStore
from src.platform_core.database import ChatMessage
from src.platform_core.db_session import new_session
from src.platform_core.async_utils import run_sync
from src.ingress.client import WeComClient
from data.config import config

# 导入拆分后的处理器
from src.conversation.command_handler import CommandHandler
from src.conversation.image_handler import ImageHandler
from src.conversation.pending_confirmation_handler import PendingConfirmationHandler
from src.conversation.context_builder import ContextBuilder
from src.conversation.fast_path_config import (
    INTENT_PENDING_RESOLUTION,
    strip_agent_prefix,
)
from src.conversation.fast_path_router import FastPathRouter
from src.conversation.reply_cleaner import ReplyCleaner
from src.ingress.middleware.dedup_middleware import DedupMiddleware

logger = logging.getLogger('wecom')

# 当前有活跃 run 时，仅把规范化后的明确取消口令识别为 abort。
_ABORT_COMMANDS = {"取消", "停止", "别弄了", "不做了", "算了", "stop", "cancel", "abort"}
_ABORT_TRAILING_PUNCTUATION = "!?！？…。,，;；:：'\"`’”)]}】」』>"

# 长 confirm_reply 的展示阈值：超过即给用户精简版，原文仍进上下文供 agent 下一轮使用。
_COMPACT_CONFIRM_REPLY_MAX_LINES = 25
_COMPACT_CONFIRM_REPLY_MAX_CHARS = 1500
_COMPACT_CONFIRM_REPLY_PREVIEW_LINES = 8


def _compact_confirm_reply_for_user(confirm_reply: str) -> str:
    """长确认回复精简后给用户，原文仍保留在 ChatMessage / short_memory 里。

    短输出直接原样返回；超过行数或字符阈值时，拼一个只含元信息+前几行预览的
    紧凑版，并提示用户可以让 agent 总结/展开。
    """
    if not confirm_reply:
        return confirm_reply

    lines = confirm_reply.splitlines()
    total_chars = len(confirm_reply)
    if len(lines) <= _COMPACT_CONFIRM_REPLY_MAX_LINES and total_chars <= _COMPACT_CONFIRM_REPLY_MAX_CHARS:
        return confirm_reply

    header_lines: List[str] = []
    stdout_body: List[str] = []
    seen_stdout_marker = False
    for line in lines:
        if line.startswith("标准输出:"):
            seen_stdout_marker = True
            continue
        if seen_stdout_marker:
            stdout_body.append(line)
        elif line.startswith(("命令:", "退出码:", "执行模式:")):
            header_lines.append(line)

    if not header_lines and not stdout_body:
        head_preview = "\n".join(lines[:_COMPACT_CONFIRM_REPLY_PREVIEW_LINES])
        return (
            f"✓ 命令已执行\n{head_preview}\n\n"
            f"（完整输出共 {len(lines)} 行 / {total_chars} 字符，已存入上下文；"
            "需要我总结或展开请直接说）"
        )

    preview = "\n".join(stdout_body[:_COMPACT_CONFIRM_REPLY_PREVIEW_LINES])
    parts = ["✓ 命令已执行"]
    if header_lines:
        parts.append("\n".join(header_lines))
    if preview:
        parts.append(f"输出预览（前 {_COMPACT_CONFIRM_REPLY_PREVIEW_LINES} 行）:\n{preview}")
    parts.append(
        f"（完整输出共 {len(lines)} 行 / {total_chars} 字符，已存入上下文；"
        "需要我总结或展开请直接说）"
    )
    return "\n\n".join(parts)


def _normalize_abort_command(content: str) -> str:
    text = (content or "").strip().lower()
    if not text:
        return ""
    text = " ".join(text.split())
    return text.rstrip(_ABORT_TRAILING_PUNCTUATION).strip()


def _is_abort_command(content: str) -> bool:
    return _normalize_abort_command(content) in _ABORT_COMMANDS


class MessageHandler:
    """企业微信消息处理器 - 重构版"""

    def __init__(self, wecom_client: WeComClient):
        """初始化消息处理器"""
        self.client = wecom_client

        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.vector_backend = self._resolve_vector_backend()

        # ========== 初始化核心服务 ==========
        # LLM 服务
        self.llm_service = LLMService(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            model=config.llm.model,
            max_token=config.llm.max_tokens,
            temperature=config.llm.temperature,
            max_groups=config.behavior.context.max_groups,
            auto_model_switch=getattr(config.llm, 'auto_model_switch', False),
            fallback_models=getattr(config.llm, 'fallback_models', [])
        )

        self.vector_store = self._init_vector_store()

        # 其他服务
        self.memory = MemoryManager(llm_service=self.llm_service, vector_store=self.vector_store)
        self.rag = SdkRAGService()
        self.reply_service = AgentService(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            rag_service=self.rag,
        )
        self.session_service = SessionService()

        # ========== 初始化处理器（使用拆分后的组件）==========
        self.command_handler = CommandHandler(self.rag)
        self.image_handler = ImageHandler(self.client)
        self.context_builder = ContextBuilder(self.memory, self.session_service, self.root_dir)
        self.fast_path_router = FastPathRouter(
            self.reply_service.tool_catalog,
            self.session_service,
            self.llm_service,
        )
        self.pending_confirmation_handler = PendingConfirmationHandler(
            reply_service=self.reply_service,
            fast_path_router=self.fast_path_router,
            run_sync_func=run_sync,
        )

        logger.info(
            f"MessageHandler 初始化完成: vector_backend={self.vector_backend}, "
            f"rag_backend=sdk, agent_backend=self-built"
        )

    def _resolve_vector_backend(self) -> str:
        backend = os.getenv("VECTOR_BACKEND", "qdrant").strip().lower() or "qdrant"
        if backend != "qdrant":
            logger.warning(f"VECTOR_BACKEND={backend} 已不再作为主路径支持，改用 qdrant")
        return "qdrant"

    def _init_vector_store(self) -> QdrantVectorStore:
        qdrant_url = os.getenv("ATRINEXUS_QDRANT_URL", "").strip() or None
        qdrant_api_key = os.getenv("ATRINEXUS_QDRANT_API_KEY", "").strip() or None
        qdrant_path = os.getenv(
            "ATRINEXUS_QDRANT_PATH",
            os.path.join(self.root_dir, "data", "vectordb_qdrant"),
        )

        if qdrant_url:
            logger.info(f"初始化 Qdrant 向量存储: url={qdrant_url}")
            return QdrantVectorStore(url=qdrant_url, api_key=qdrant_api_key, embedding_function=None)

        os.makedirs(qdrant_path, exist_ok=True)
        logger.info(f"初始化 Qdrant 向量存储: path={qdrant_path}")
        return QdrantVectorStore(path=qdrant_path, embedding_function=None)

    # ========== 消息处理入口 ==========

    async def process_image_message(self, user_id: str, media_id: str, msg_id: str, pic_url: str = None):
        """处理图片消息"""
        logger.info(f"开始处理图片消息: user={user_id}, media_id={media_id}")

        # 去重检查
        if DedupMiddleware.is_duplicate_message(msg_id):
            logger.info(f"跳过重复图片消息: {msg_id}")
            return

        # 处理图片
        image_description = await run_sync(self.image_handler.process_image, user_id, media_id)
        if not image_description:
            await self.client.send_text_async(user_id, "抱歉，图片下载失败了，请重新发送试试 😊")
            return

        # 转换为文本消息处理
        content = f"[用户发来一张图片，图片内容：{image_description}]"
        await self.process_message(user_id=user_id, content=content, msg_id=msg_id)
        logger.info(f"图片消息处理完成: user={user_id}")

    async def process_message(self, user_id: str, content: str, msg_id: str):
        """
        异步处理消息的核心方法

        Args:
            user_id: 用户ID
            content: 消息内容
            msg_id: 消息ID
        """
        logger.info(f"开始处理消息: user={user_id}, content_len={len(content)}")

        # 1. 去重检查
        if DedupMiddleware.is_duplicate_message(msg_id):
            logger.info(f"跳过重复消息: {msg_id}")
            return

        content_trim = content.strip()

        # 2. 当前用户已有活跃 agent run：明确取消口令立即处理；其他消息进 follow-up 队列
        if await self.reply_service.is_running(user_id):
            if _is_abort_command(content_trim):
                aborted = await self.reply_service.abort(user_id)
                ack = "已请求取消，马上停下手里的活。" if aborted else "当前没有在处理的任务。"
                await self.client.send_text_async(user_id, ack)
                return
            queue_len = await self.reply_service.queue_follow_up(user_id, content_trim)
            logger.info(
                "用户 %s 处于 run 中，消息进入 follow-up 队列（第 %s 条）",
                user_id,
                queue_len,
            )
            return

        confirm_reply = await self._handle_pending_action_confirmation(user_id, content_trim)
        if confirm_reply is not None:
            state = self.session_service.get_session(user_id)
            avatar_name = state.avatar_name or "ATRI"
            await run_sync(self._save_chat_message, user_id, content_trim, confirm_reply, msg_id)
            try:
                await self.memory.after_reply_async(user_id, avatar_name, content_trim, confirm_reply)
            except Exception as exc:
                logger.error(f"confirm-reply 写入 short_memory 失败: {exc}")
            record_fast_path_turn(
                user_id=user_id,
                user_message=content_trim,
                assistant_reply=confirm_reply,
                intent=INTENT_PENDING_RESOLUTION,
            )
            await self.client.send_text_async(user_id, _compact_confirm_reply_for_user(confirm_reply))
            return

        # 3. `/agent` 前缀：用户显式绕过 FastPath 意图路由，进 agent loop。
        agent_payload, bypass_fast_path = strip_agent_prefix(content_trim)
        if bypass_fast_path:
            content = agent_payload
            content_trim = agent_payload

        # 4. 快路径：能力查询、读文件、列目录、重命名
        if not bypass_fast_path:
            outcome = await run_sync(self.fast_path_router.try_handle, user_id, content_trim)
            if outcome.reply is not None:
                record_fast_path_turn(
                    user_id=user_id,
                    user_message=content_trim,
                    assistant_reply=outcome.reply,
                    intent=outcome.intent,
                )
                await self.client.send_text_async(user_id, outcome.reply)
                return

        # 5. 检查是否是命令
        if self.command_handler.is_command(content_trim):
            reply = await run_sync(self.command_handler.handle_command, user_id, content_trim)
            if reply:
                await self.client.send_text_async(user_id, reply)
            return

        # 6. 正常消息处理流程
        await self._execute_kb_search(user_id, content, msg_id)

        # 7. 运行期间用户可能追发了其他消息，依次消耗 follow-up 队列
        await self._drain_follow_up_queue(user_id)

    async def _drain_follow_up_queue(self, user_id: str) -> None:
        follow_ups = await self.reply_service.drain_follow_up(user_id)
        for queued in follow_ups:
            logger.info("消化 follow-up 消息: user=%s, content_len=%s", user_id, len(queued))
            try:
                await self._execute_kb_search(user_id, queued, msg_id=f"followup-{user_id}")
            except Exception as exc:
                logger.error("follow-up 消息处理失败: %s", exc, exc_info=True)

    async def _handle_pending_action_confirmation(self, user_id: str, content: str):
        return await self.pending_confirmation_handler.handle(user_id, content)

    async def _execute_kb_search(self, user_id: str, content: str, msg_id: str,
                                  category_filter: str = None):
        """执行普通消息回复生成。KB 检索改为 agent 按需工具调用。"""

        # 1. 构建上下文
        ctx = await run_sync(
            self.context_builder.build_search_context,
            user_id, content
        )
        avatar_name = ctx["avatar_name"]
        current_mode = ctx["current_mode"]
        previous_context = ctx["previous_context"]

        # 2. 构建记忆上下文
        core_memory = self.context_builder.build_merged_memory_context(ctx["mem_ctx"])

        # 3. 构建系统提示词
        system_prompt = self.context_builder.build_system_prompt(avatar_name, current_mode)
        tool_profile = self.session_service.get_tool_profile(user_id)
        # 4. 调用 LLM 生成回复
        try:
            reply = await self.reply_service.generate_reply_async(
                message=content,
                user_id=user_id,
                system_prompt=system_prompt,
                tool_profile=tool_profile,
                previous_context=previous_context,
                core_memory=core_memory,
                current_mode=current_mode,
            )

            # 清理回复
            reply = ReplyCleaner.clean_reply(reply)

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            reply = "抱歉，我暂时无法处理您的消息，请稍后再试。"

        # 5. 保存对话记录
        await run_sync(self._save_chat_message, user_id, content, reply, msg_id)

        # 6. 发送回复
        if reply.strip():
            await self.client.send_text_async(user_id, reply)
        else:
            logger.warning(f"回复内容为空，发送默认回复")
            await self.client.send_text_async(user_id, "😊")

        logger.info(f"消息已发送给用户，准备进行后台记忆更新: user={user_id}, reply_len={len(reply)}")

        # 7. 更新记忆
        try:
            await self.memory.after_reply_async(user_id, avatar_name, content, reply)
        except Exception as e:
            logger.error(f"更新记忆失败: {e}")

        logger.info(f"消息全部处理完成: user={user_id}")

    def _save_chat_message(self, user_id: str, content: str, reply: str, msg_id: str):
        """保存聊天记录到数据库"""
        with new_session() as session:
            try:
                chat_msg = ChatMessage(
                    sender_id=user_id,
                    sender_name=user_id,
                    message=content,
                    reply=reply,
                    wecom_msg_id=msg_id
                )
                session.add(chat_msg)
                session.commit()
            except IntegrityError:
                session.rollback()
                logger.info(f"聊天记录已存在，跳过重复保存: msg_id={msg_id}")
            except Exception as e:
                logger.error(f"保存聊天记录失败: {e}")
                session.rollback()

