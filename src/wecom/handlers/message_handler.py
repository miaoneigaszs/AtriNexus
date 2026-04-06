"""
WeCom 消息处理器（重构版）
负责消息处理流程的编排，将具体逻辑委托给各个处理器
"""

import logging
import os
import re

from sqlalchemy.exc import IntegrityError

from src.services.ai.llm_service import LLMService
from src.services.agent.langchain_agent_service import LangChainAgentService
from src.services.agent.tool_profiles import merge_tool_profile, normalize_tool_profile
from src.services.memory_manager import MemoryManager
from src.services.rag_service import SdkRAGService
from src.services.session_service import SessionService
from src.services.vector_store import QdrantVectorStore
from src.services.database import Session, ChatMessage
from src.utils.async_utils import run_sync
from src.wecom.client import WeComClient
from data.config import config

# 导入拆分后的处理器
from src.wecom.handlers.command_handler import CommandHandler
from src.wecom.handlers.image_handler import ImageHandler
from src.wecom.processors.context_builder import ContextBuilder
from src.wecom.processors.fast_path_router import FastPathRouter
from src.wecom.processors.reply_cleaner import ReplyCleaner
from src.wecom.middleware.dedup_middleware import DedupMiddleware

logger = logging.getLogger('wecom')


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
        self.reply_service = LangChainAgentService(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            rag_service=self.rag,
        )
        self.session_service = SessionService(kb_session_timeout=5)

        # ========== 初始化处理器（使用拆分后的组件）==========
        self.command_handler = CommandHandler(self.rag)
        self.image_handler = ImageHandler(self.client)
        self.context_builder = ContextBuilder(self.memory, self.session_service, self.root_dir)
        self.fast_path_router = FastPathRouter(
            self.reply_service.tool_catalog,
            self.session_service,
            self.llm_service,
        )

        logger.info(
            f"MessageHandler 初始化完成: vector_backend={self.vector_backend}, "
            f"rag_backend=sdk, agent_backend=langchain"
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

        confirm_reply = await self._handle_pending_action_confirmation(user_id, content_trim)
        if confirm_reply is not None:
            await self.client.send_text_async(user_id, confirm_reply)
            return

        # 2. 快路径：能力查询、读文件、列目录、重命名
        fast_path_reply = await run_sync(self.fast_path_router.try_handle, user_id, content_trim)
        if fast_path_reply is not None:
            await self.client.send_text_async(user_id, fast_path_reply)
            return

        # 3. 检查是否是命令
        if self.command_handler.is_command(content_trim):
            reply = await run_sync(self.command_handler.handle_command, user_id, content_trim)
            if reply:
                await self.client.send_text_async(user_id, reply)
            return

        # 4. 正常消息处理流程
        await self._execute_kb_search(user_id, content, msg_id)

    async def _handle_pending_action_confirmation(self, user_id: str, content: str):
        if content in {"审批通过", "通过", "确认", "同意"}:
            latest_change_id = self.reply_service.get_latest_pending_change_id(user_id)
            if latest_change_id:
                return await run_sync(self.reply_service.apply_pending_change, latest_change_id, user_id)
            latest_command_id = self.reply_service.get_latest_pending_command_id(user_id)
            if latest_command_id:
                return await run_sync(self.reply_service.confirm_pending_command, latest_command_id, user_id)
            return "当前没有待审批的命令或修改。"

        confirm_match = re.fullmatch(r"确认执行\s+([A-Za-z0-9_-]+)", content)
        if confirm_match:
            return await run_sync(self.reply_service.confirm_pending_command, confirm_match.group(1), user_id)

        if content == "确认执行":
            latest_command_id = self.reply_service.get_latest_pending_command_id(user_id)
            if latest_command_id:
                return await run_sync(self.reply_service.confirm_pending_command, latest_command_id, user_id)
            return "当前没有待确认执行的命令。"

        discard_match = re.fullmatch(r"取消执行\s+([A-Za-z0-9_-]+)", content)
        if discard_match:
            return await run_sync(self.reply_service.discard_pending_command, discard_match.group(1), user_id)

        if content == "取消执行":
            latest_command_id = self.reply_service.get_latest_pending_command_id(user_id)
            if latest_command_id:
                return await run_sync(self.reply_service.discard_pending_command, latest_command_id, user_id)
            return "当前没有待取消的命令。"

        apply_change_match = re.fullmatch(r"确认修改\s+([A-Za-z0-9_-]+)", content)
        if apply_change_match:
            return await run_sync(self.reply_service.apply_pending_change, apply_change_match.group(1), user_id)

        if content == "确认修改":
            latest_change_id = self.reply_service.get_latest_pending_change_id(user_id)
            if latest_change_id:
                return await run_sync(self.reply_service.apply_pending_change, latest_change_id, user_id)
            return "当前没有待确认的修改。"

        discard_change_match = re.fullmatch(r"取消修改\s+([A-Za-z0-9_-]+)", content)
        if discard_change_match:
            return await run_sync(self.reply_service.discard_pending_change, discard_change_match.group(1), user_id)

        if content == "取消修改":
            latest_change_id = self.reply_service.get_latest_pending_change_id(user_id)
            if latest_change_id:
                return await run_sync(self.reply_service.discard_pending_change, latest_change_id, user_id)
            return "当前没有待取消的修改。"

        workspace_resolution_reply = await run_sync(
            self.fast_path_router.try_handle_pending_resolution,
            user_id,
            content,
        )
        if workspace_resolution_reply is not None:
            return workspace_resolution_reply

        return None

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
        inferred_profile = self.reply_service.tool_catalog.infer_tool_profile(content)
        current_profile = self.session_service.get_tool_profile(user_id)
        tool_profile = merge_tool_profile(current_profile, inferred_profile)
        if tool_profile != normalize_tool_profile(current_profile):
            self.session_service.set_tool_profile(user_id, tool_profile)
        # 4. 调用 LLM 生成回复
        try:
            reply = await self.reply_service.generate_reply_async(
                message=content,
                user_id=user_id,
                system_prompt=system_prompt,
                tool_profile=tool_profile,
                previous_context=previous_context,
                core_memory=core_memory,
                kb_context=None,
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
        with Session() as session:
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

