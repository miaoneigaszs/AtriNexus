"""
WeCom 消息处理器（重构版）
负责消息处理流程的编排，将具体逻辑委托给各个处理器
"""

import logging
import os

from src.services.ai.llm_service import LLMService
from src.services.agent import LangChainAgentService
from src.services.memory_manager import MemoryManager
from src.services.rag_engine import RAGEngine
from src.services.rag_service import LegacyRAGService, SdkRAGService
from src.services.session_service import SessionService
from src.services.intent_service import IntentService
from src.services.vector_store import QdrantVectorStore
from src.services.database import Session, ChatMessage
from src.utils.async_utils import run_sync
from src.wecom.client import WeComClient
from data.config import config

# 导入拆分后的处理器
from src.wecom.handlers.command_handler import CommandHandler
from src.wecom.handlers.image_handler import ImageHandler
from src.wecom.processors.context_builder import ContextBuilder
from src.wecom.processors.rag_processor import RAGProcessor
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
        self.rag_backend = self._resolve_rag_backend()
        self.agent_backend = self._resolve_agent_backend()

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
        self.rag = self._init_rag_service()
        self.reply_service = self._init_reply_service()
        self.session_service = SessionService(kb_session_timeout=5)
        self.intent_service = IntentService(rag_service=self.rag)

        # ========== 初始化处理器（使用拆分后的组件）==========
        self.command_handler = CommandHandler(self.rag)
        self.image_handler = ImageHandler(self.client)
        self.context_builder = ContextBuilder(self.memory, self.session_service, self.root_dir)
        self.rag_processor = RAGProcessor(self.rag, self.intent_service, self.session_service)

        logger.info(
            f"MessageHandler 初始化完成: vector_backend={self.vector_backend}, "
            f"rag_backend={self.rag_backend}, agent_backend={self.agent_backend}"
        )

    def _resolve_vector_backend(self) -> str:
        backend = os.getenv("VECTOR_BACKEND", "qdrant").strip().lower() or "qdrant"
        if backend != "qdrant":
            logger.warning(f"VECTOR_BACKEND={backend} 已不再作为主路径支持，改用 qdrant")
        return "qdrant"

    def _resolve_rag_backend(self) -> str:
        backend = os.getenv("RAG_BACKEND", "sdk").strip().lower() or "sdk"
        if backend not in {"sdk", "legacy"}:
            logger.warning(f"RAG_BACKEND={backend} 非法，改用 sdk")
            return "sdk"
        return backend

    def _resolve_agent_backend(self) -> str:
        backend = os.getenv("AGENT_BACKEND", "legacy").strip().lower() or "legacy"
        if backend not in {"legacy", "langchain"}:
            logger.warning(f"AGENT_BACKEND={backend} 非法，改用 legacy")
            return "legacy"
        return backend

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

    def _build_legacy_rag_service(self) -> LegacyRAGService:
        return LegacyRAGService(RAGEngine(vector_store=self.vector_store))

    def _init_rag_service(self):
        legacy_rag = self._build_legacy_rag_service()
        if self.rag_backend == "sdk":
            try:
                logger.info("初始化 SDK RAG 服务")
                return SdkRAGService(fallback_service=legacy_rag)
            except Exception as e:
                logger.error(f"SDK RAG 初始化失败，回退到 legacy: {e}")
                self.rag_backend = "legacy"
        return legacy_rag

    def _init_reply_service(self):
        if self.agent_backend == "langchain":
            try:
                logger.info("初始化 LangChain Agent 回复服务")
                return LangChainAgentService(
                    api_key=config.llm.api_key,
                    base_url=config.llm.base_url,
                    model=config.llm.model,
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_tokens,
                )
            except Exception as e:
                logger.error(f"LangChain Agent 初始化失败，回退到 legacy: {e}")
                self.agent_backend = "legacy"
        return self.llm_service

    # ========== 消息处理入口 ==========

    async def process_image_message(self, user_id: str, media_id: str, msg_id: str, pic_url: str = None):
        """处理图片消息"""
        logger.info(f"开始处理图片消息: user={user_id}, media_id={media_id}")

        # 去重检查
        if DedupMiddleware.is_duplicate_message(msg_id):
            logger.info(f"跳过重复图片消息: {msg_id}")
            return

        # 处理图片
        image_description = self.image_handler.process_image(user_id, media_id)
        if not image_description:
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

        # 2. 检查是否是命令
        if self.command_handler.is_command(content_trim):
            reply = self.command_handler.handle_command(user_id, content_trim)
            if reply:
                self.client.send_text(user_id, reply)
            return

        # 3. 正常消息处理流程
        await self._execute_kb_search(user_id, content, msg_id)

    async def _execute_kb_search(self, user_id: str, content: str, msg_id: str,
                                  category_filter: str = None):
        """执行知识库检索和回复生成（流程编排）"""

        # 1. 构建上下文
        ctx = await run_sync(
            self.context_builder.build_search_context,
            user_id, content
        )
        avatar_name = ctx["avatar_name"]
        current_mode = ctx["current_mode"]
        previous_context = ctx["previous_context"]

        # 2. 执行 RAG 检索
        kb_results, _, need_search = await run_sync(
            self.rag_processor.execute_rag_retrieval,
            user_id, content, previous_context, category_filter
        )

        # 3. 构建知识库上下文
        kb_context = ""
        if kb_results:
            logger.info(f"==>[RAG Trace] 已将 {len(kb_results)} 个知识切片注入 Prompt 上下文。")
            kb_context = self.context_builder.build_kb_context(kb_results)

        # 4. 构建记忆上下文
        core_memory = self.context_builder.build_merged_memory_context(ctx["mem_ctx"])

        # 5. 构建系统提示词
        system_prompt = self.context_builder.build_system_prompt(avatar_name, current_mode)
        # 6. 调用 LLM 生成回复
        try:
            reply = await run_sync(
                self.reply_service.get_response,
                message=content,
                user_id=user_id,
                system_prompt=system_prompt,
                previous_context=previous_context,
                core_memory=core_memory,
                kb_context=kb_context if kb_context else None
            )

            # 清理回复
            reply = ReplyCleaner.clean_reply(reply)

            # 处理知识库引用：提取使用的片段并清理标记
            used_indices = []
            if need_search and kb_results:
                reply, used_indices = self.context_builder.extract_and_clean_references(reply)
                # 只添加被使用的片段作为参考
                if used_indices:
                    references = self.context_builder.format_kb_references(kb_results, used_indices)
                    if references:
                        reply += references

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            reply = "抱歉，我暂时无法处理您的消息，请稍后再试。"

        # 7. 保存对话记录
        await run_sync(self._save_chat_message, user_id, content, reply, msg_id)

        # 8. 发送回复
        if reply.strip():
            await run_sync(self.client.send_text, user_id, reply)
        else:
            logger.warning(f"回复内容为空，发送默认回复")
            await run_sync(self.client.send_text, user_id, "😊")

        logger.info(f"消息已发送给用户，准备进行后台记忆更新: user={user_id}, reply_len={len(reply)}")

        # 9. 更新记忆
        try:
            await self.memory.after_reply_async(user_id, avatar_name, content, reply)
        except Exception as e:
            logger.error(f"更新记忆失败: {e}")

        logger.info(f"消息全部处理完成: user={user_id}")

    def _save_chat_message(self, user_id: str, content: str, reply: str, msg_id: str):
        """保存聊天记录到数据库"""
        session = Session()
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
        except Exception as e:
            logger.error(f"保存聊天记录失败: {e}")
            session.rollback()
        finally:
            session.close()

