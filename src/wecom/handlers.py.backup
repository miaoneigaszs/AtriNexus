"""
WeCom 消息处理器（简化版）
核心业务逻辑:
- 模式判断（工作/陪伴）
- 记忆加载与更新
- 知识库检索 / 网络搜索 / 闲聊路由
- 调用 LLM 生成回复
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import httpx

from src.services.database import Session, ChatMessage
from src.services.ai.llm_service import LLMService
from src.services.memory_manager import MemoryManager
from src.services.rag_engine import RAGEngine
from src.services.session_service import SessionService
from src.services.intent_service import IntentService
from src.utils.async_utils import run_sync
from src.services.ai.image_recognition_service import ImageRecognitionService
from src.wecom.client import WeComClient
from data.config import config

logger = logging.getLogger('wecom')


class MessageHandler:
    """企业微信消息处理器"""

    def __init__(self, wecom_client: WeComClient):
        """初始化消息处理器"""
        self.client = wecom_client
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # 初始化 LLM 服务
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

        # 初始化 ChromaDB
        import chromadb
        from chromadb.config import Settings
        vectordb_path = os.path.join(self.root_dir, 'data', 'vectordb')
        os.makedirs(vectordb_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=vectordb_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # 初始化核心服务
        self.memory = MemoryManager(llm_service=self.llm_service, chroma_client=self.chroma_client)
        self.rag = RAGEngine(chroma_client=self.chroma_client)
        self.session_service = SessionService(kb_session_timeout=5)
        self.intent_service = IntentService(rag_engine=self.rag)

        # 人设目录
        self.avatar_dir = config.behavior.context.avatar_dir

        logger.info("MessageHandler 初始化完成")

    # ---------- 消息去重 ----------

    def is_duplicate_message(self, msg_id: str) -> bool:
        """检查消息是否已处理过"""
        session = Session()
        try:
            exists = session.query(ChatMessage).filter_by(wecom_msg_id=msg_id).first()
            return exists is not None
        finally:
            session.close()

    # ---------- 核心处理流程 ----------

    def _load_avatar_prompt(self, avatar_name: str) -> str:
        """加载人设提示词（基础提示词 + 角色专属）"""
        prompt = ""
        # 1. 加载系统基础预设 (Base Prompt)
        base_prompt_path = os.path.join(self.root_dir, 'data', 'prompts', 'base.md')
        if os.path.exists(base_prompt_path):
            with open(base_prompt_path, 'r', encoding='utf-8') as f:
                prompt += f.read() + "\n\n"
        
        # 2. 加载角色专属人设
        avatar_path = os.path.join(self.root_dir, 'data', 'avatars', avatar_name, 'avatar.md')
        try:
            with open(avatar_path, 'r', encoding='utf-8') as f:
                prompt += f.read()
        except FileNotFoundError:
            logger.warning(f"人设文件不存在: {avatar_path}，仅使用基础预设")
            
        return prompt

    def _check_companion_trigger(self, content: str) -> bool:
        """检查是否触发陪伴模式"""
        triggers = config.companion_mode.triggers
        for trigger in triggers:
            if trigger.lower() in content.lower():
                return True
        return False

    def _recognize_image(self, image_data: bytes) -> str:
        """调用 VL 模型识别图片内容，委托给 ImageRecognitionService"""
        vl_config = config.media.image_recognition
        api_key = vl_config.api_key or config.llm.api_key
        base_url = vl_config.base_url or config.llm.base_url
        model = vl_config.model

        if not model:
            logger.warning("[Handler] 未配置图片识别模型，跳过图片识别")
            return "无法识别的图片"

        service = ImageRecognitionService(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=getattr(vl_config, 'temperature', 0.3)
        )
        description = service.describe(image_data)
        return description

    @staticmethod
    def _clean_reply(text: str) -> str:
        """清理回复中的格式问题"""
        # 移除 LaTeX 行内公式
        text = re.sub(r'\$([^$]+)\$', r'\1', text)
        # 移除 LaTeX 块公式
        text = re.sub(r'\$\$([^$]+)\$\$', r'\1', text)
        # 移除残留的 $ 符号
        text = text.replace('$', '')
        # 移除 markdown 加粗
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        # 移除 markdown 斜体
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        return text.strip()

    # ========== 增强的意图识别 (V2) ==========

    def _check_kb_intent_v2(self, user_id: str, content: str,
                            previous_context: str = "") -> Dict[str, Any]:
        """委托给 IntentService 处理"""
        return self.intent_service.recognize_intent(
            user_id, content, previous_context
        )

    def _handle_kb_session_response(self, user_id: str, content: str) -> Optional[Dict[str, Any]]:
        """
        处理用户在 KB 检索会话中的回复
        
        返回:
            None: 不在 KB 会话中
            {"action": "search_all", "original_query": str}: 用户选择搜索全部
            {"action": "search_category", "category": str, "original_query": str}: 用户选择了分类
            {"action": "clarify", "message": str}: 需要继续询问用户
        """
        kb_session = self.session_service.get_kb_search_session(user_id)
        if not kb_session:
            return None

        # 解析会话数据
        try:
            candidates = json.loads(kb_session.candidates)
        except:
            self.session_service.clear_kb_search_session(user_id)
            return None

        original_query = kb_session.original_query or content
        content_trim = content.strip()

        # 处理分类选择
        if kb_session.waiting_for == "category":
            # 检查是否选择全部
            if "全部" in content_trim or "所有" in content_trim:
                self.session_service.clear_kb_search_session(user_id)
                return {"action": "search_all", "original_query": original_query}

            # 尝试解析数字选择
            try:
                numbers = re.findall(r'\d+', content_trim)
                if numbers:
                    choice = int(numbers[0]) - 1
                    if 0 <= choice < len(candidates):
                        selected_category = candidates[choice]
                        self.session_service.clear_kb_search_session(user_id)
                        return {"action": "search_category", "category": selected_category, "original_query": original_query}
            except:
                pass

            # 尝试直接匹配分类名称
            for cat in candidates:
                if cat in content_trim:
                    self.session_service.clear_kb_search_session(user_id)
                    return {"action": "search_category", "category": cat, "original_query": original_query}

            # 无法识别，重新提示
            return {"action": "clarify", "message": f"未能理解您的选择，请回复数字 1-{len(candidates)} 或分类名称。\n\n"}

        # 未知状态，清除会话
        self.session_service.clear_kb_search_session(user_id)
        return None

    # ========== 增强的知识库检索 ==========

    def _prepare_search_context(self, user_id: str, content: str) -> Dict[str, Any]:
        """获取会话状态和记忆上下文"""
        state = self.session_service.get_session(user_id)
        avatar_name = state.avatar_name or 'ATRI'
        current_mode = state.mode or 'work'
        if self._check_companion_trigger(content):
            current_mode = 'companion'
            self.session_service.update_session_mode(user_id, 'companion')
            logger.info(f"用户 {user_id} 切换至陪伴模式")
        mem_ctx = self.memory.build_full_context(user_id, avatar_name, content)
        return {
            "state": state,
            "avatar_name": avatar_name,
            "current_mode": current_mode,
            "mem_ctx": mem_ctx,
            "previous_context": mem_ctx["previous_context"],
        }

    def _execute_rag_retrieval(
        self, user_id: str, content: str, previous_context: list,
        category_filter: Optional[str] = None
    ) -> Optional[Tuple[List[Dict], str, bool]]:
        """
        简单意图识别 + 检索
        
        返回:
            - None: 需要澄清，已发送消息
            - (kb_results, kb_context, need_search): 检索结果
        """
        # 如果指定了分类过滤，直接检索
        if category_filter:
            kb_results = self.rag.retrieve_knowledge(user_id, content, top_k=3, category_filter=category_filter)
            return (kb_results, self._build_kb_context(kb_results), True)

        # 简单意图识别
        intent_result = self.intent_service.recognize_intent(user_id, content, previous_context)
        intent = intent_result.get("intent", "TYPE_CHITCHAT")
        query = intent_result.get("query", content)
        
        logger.info(f"[意图识别] intent={intent}, confidence={intent_result.get('confidence', 0):.2f}")

        # 闲聊（包括需要网络搜索的内容，由 LLM 工具调用处理）
        if intent != "TYPE_KNOWLEDGE_BASE":
            return ([], "", False)

        # 知识库查询 - 需要澄清
        if intent_result.get("need_clarify") and intent_result.get("suggestions"):
            kb_list = self.rag.get_knowledge_list(user_id)
            msg = self.intent_service.generate_clarification_message(
                query, intent_result["suggestions"], kb_list
            )
            self.session_service.create_kb_search_session(
                user_id=user_id, original_query=query,
                waiting_for="category", candidates=intent_result["suggestions"]
            )
            self.client.send_text(user_id, msg)
            return None

        # 知识库检索
        cat_filter = intent_result.get("category") if intent_result.get("confidence", 0) >= 0.7 else None
        kb_results = self.rag.retrieve_knowledge(user_id, query, top_k=3, category_filter=cat_filter)
        return (kb_results, self._build_kb_context(kb_results), True)

    def _build_kb_context(self, kb_results: List[Dict], include_headers: bool = True) -> str:
        """构建知识库上下文提示词"""
        if not kb_results:
            return ""

        kb_context = "[从知识库中检索到的极度相关参考资料，你必须优先根据这些资料回答用户的最后一条消息]：\n"

        for i, res in enumerate(kb_results):
            meta = res['metadata']
            score = res.get('score', 0)

            kb_context += f"\n--- 参考切片 {i+1} ---\n"
            kb_context += f"来源：《{meta.get('file_name', '未知文件')}》"

            # 添加分类信息
            category = meta.get('category', '')
            if category:
                kb_context += f" ({category})"
            kb_context += "\n"

            # 添加标题结构
            if include_headers:
                titles = [meta.get(f"H{j}") for j in range(1, 4) if meta.get(f"H{j}")]
                if titles:
                    kb_context += f"章节：{' > '.join(titles)}\n"

            # 添加相关度
            if score > 0:
                kb_context += f"相关度：{score:.2f}\n"

            kb_context += f"内容：{res['content']}\n"

        return kb_context

    def _format_kb_references(self, kb_results: List[Dict]) -> str:
        """格式化参考来源标注"""
        references = []
        for res in kb_results:
            meta = res.get('metadata', {})
            file_name = meta.get('file_name', '')
            h1 = meta.get('H1', '')
            if file_name:
                ref = f"《{file_name}》"
                if h1:
                    ref += f"({h1})"
                if ref not in references:
                    references.append(ref)

        if references:
            return f"\n\n🔍 **参考依据**: " + ", ".join(references[:3])
        return ""

    # ========== 文档结构展示命令 ==========

    def _handle_kb_outline_command(self, user_id: str, file_name: str = None) -> str:
        """处理查看文档结构大纲命令"""
        outline = self.rag.get_document_outline(user_id, file_name)

        if not outline or not outline.get("documents"):
            return "📚 您的知识库中没有文档，请先上传文档。"

        reply_parts = ["📚 知识库文档结构：\n"]

        for doc_name, structure in outline["documents"].items():
            reply_parts.append(f"\n📄 《{doc_name}》【{structure['category']}】")

            h1_list = structure.get("H1", [])
            h2_list = structure.get("H2", [])

            # 显示 H1 章节（最多10个）
            if h1_list:
                reply_parts.append("  📖 一级章节：")
                for h1 in h1_list[:10]:
                    reply_parts.append(f"    ├─ {h1}")
                if len(h1_list) > 10:
                    reply_parts.append(f"    └─ ... 还有 {len(h1_list) - 10} 个")

            # 显示 H2 子章节（最多10个）
            if h2_list:
                reply_parts.append("  📑 二级章节：")
                for h2 in h2_list[:10]:
                    reply_parts.append(f"    ├─ {h2}")
                if len(h2_list) > 10:
                    reply_parts.append(f"    └─ ... 还有 {len(h2_list) - 10} 个")

        reply_parts.append("\n💡 提示：提问时指定章节可以获得更精准的答案，如：")
        reply_parts.append('   "查一下财务制度中第一章的报销标准"')

        return "\n".join(reply_parts)

    # ========== 主消息处理流程 ==========

    async def process_image_message(self, user_id: str, media_id: str, msg_id: str, pic_url: str = None):
        """处理图片消息"""
        logger.info(f"开始处理图片消息: user={user_id}, media_id={media_id}")

        if self.is_duplicate_message(msg_id):
            logger.info(f"跳过重复图片消息: {msg_id}")
            return

        image_data = self.client.download_media(media_id)
        if not image_data:
            self.client.send_text(user_id, "抱歉，图片下载失败了，请重新发送试试 😊")
            return

        image_description = self._recognize_image(image_data)
        logger.info(f"图片描述完成: {image_description[:50]}...")

        content = f"[用户发来一张图片，图片内容：{image_description}]"
        await self.process_message(user_id=user_id, content=content, msg_id=msg_id)
        logger.info(f"图片消息处理完成: user={user_id}")

    async def process_message(self, user_id: str, content: str, msg_id: str):
        """
        异步处理消息的核心方法
        支持多轮知识库检索交互
        """
        logger.info(f"开始处理消息: user={user_id}, content={content[:50]}")

        # 1. 去重检查
        if self.is_duplicate_message(msg_id):
            logger.info(f"跳过重复消息: {msg_id}")
            return

        content_trim = content.strip()

        # ===== 知识库生命周期管理命令 =====
        if content_trim == '知识库列表':
            kb_list = self.rag.get_knowledge_list(user_id)
            if not kb_list:
                self.client.send_text(user_id, "您的知识库当前是空的哦~")
                return
            reply = "📚 **您的专属知识库一览**:\n\n"
            for cat, files in kb_list.items():
                reply += f"📂 【{cat}】\n"
                for i, f in enumerate(files):
                    reply += f"  {i+1}. {f}\n"
            reply += "\n发送【删除知识库 文件名】可移除特定资料。"
            reply += "\n发送【知识库结构】查看文档章节大纲。"
            self.client.send_text(user_id, reply)
            return

        if content_trim == '知识库结构' or content_trim.startswith('知识库结构 '):
            file_name = content_trim.replace('知识库结构', '').strip() or None
            reply = self._handle_kb_outline_command(user_id, file_name)
            self.client.send_text(user_id, reply)
            return

        if content_trim.startswith('删除知识库'):
            file_name_del = content_trim.replace('删除知识库', '').strip()
            if not file_name_del:
                self.client.send_text(user_id, "请附加要删除的文件名，例如：删除知识库 考勤规定.pdf")
                return
            success = self.rag.delete_document(user_id, file_name_del)
            if success:
                self.client.send_text(user_id, f"🗑️ 已成功从知识库中删除资料《{file_name_del}》")
            else:
                self.client.send_text(user_id, f"❌ 删除失败，可能未找到《{file_name_del}》相关的块。")
            return

        # ===== 检查是否在 KB 检索会话中 =====
        kb_session_response = self._handle_kb_session_response(user_id, content)

        if kb_session_response:
            action = kb_session_response.get("action")
            if action == "clarify":
                # 需要继续询问用户
                self.client.send_text(user_id, kb_session_response["message"])
                return
            elif action == "search_category":
                # 用户选择了特定分类，使用原始查询检索
                await self._execute_kb_search(
                    user_id, kb_session_response["original_query"], msg_id,
                    category_filter=kb_session_response["category"]
                )
                return
            elif action == "search_all":
                # 用户选择搜索全部，使用原始查询检索
                await self._execute_kb_search(
                    user_id, kb_session_response["original_query"], msg_id
                )
                return

        # ===== 正常消息处理流程 =====
        await self._execute_kb_search(user_id, content, msg_id)

    async def _execute_kb_search(self, user_id: str, content: str, msg_id: str,
                                  category_filter: str = None):
        """执行知识库检索和回复生成（流程编排）"""
        state = await run_sync(self.session_service.get_session, user_id)
        avatar_name = state.avatar_name or 'ATRI'
        ctx = await self.memory.build_full_context_async(user_id, avatar_name, content)
        # 兼容旧逻辑的数据结构准备
        current_mode = state.mode or 'work'
        if self._check_companion_trigger(content):
            current_mode = 'companion'
            await run_sync(self.session_service.update_session_mode, user_id, 'companion')
            logger.info(f"用户 {user_id} 切换至陪伴模式")
        
        mem_ctx = ctx
        previous_context = ctx["previous_context"]

        rag_result = await run_sync(
            self._execute_rag_retrieval,
            user_id, content, previous_context, category_filter
        )
        if rag_result is None:
            return  # 已发起澄清并发送消息

        kb_results, kb_context, need_search = rag_result
        if kb_results:
            logger.info(f"==>[RAG Trace] 已将 {len(kb_results)} 个知识切片注入 Prompt 上下文。")

        # 合并核心记忆和中期记忆
        core_memory = mem_ctx["core_memory"]
        if mem_ctx["relevant_memories"]:
            relevant_text = "\n".join(f"- {m}" for m in mem_ctx["relevant_memories"])
            if core_memory:
                core_memory = f"{core_memory}\n\n【相关历史对话记忆】\n{relevant_text}"
            else:
                core_memory = f"【相关历史对话记忆】\n{relevant_text}"

        # 5. 加载人设
        system_prompt = self._load_avatar_prompt(avatar_name)
        if current_mode == 'work' and system_prompt:
            system_prompt += "\n\n【当前模式：工作模式】请以专业、简洁、高效的方式回应。"

        # 如果检索到了知识库，赋予模型忽略无关知识的能力
        if kb_context:
            kb_context += "\n\n【重要提示】如果以上知识库内容与用户本次的新问题/日常分享毫不相关，请**完全忽略**它们，直接以你的身份自然地回复用户即可。"

        # 6. 调用 LLM 生成回复
        try:
            reply = await run_sync(
                self.llm_service.get_response,
                message=content,
                user_id=user_id,
                system_prompt=system_prompt,
                previous_context=previous_context,
                core_memory=core_memory,
                kb_context=kb_context if kb_context else None
            )
            reply = self._clean_reply(reply)

            # 添加参考来源
            if need_search and kb_results:
                references = self._format_kb_references(kb_results)
                if references:
                    reply += references

        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            reply = "抱歉，我暂时无法处理您的消息，请稍后再试。"

        # 7. 保存对话记录 (异步执行)
        def save_db():
            db_session = Session()
            try:
                chat_msg = ChatMessage(
                    sender_id=user_id,
                    sender_name=user_id,
                    message=content,
                    reply=reply,
                    wecom_msg_id=msg_id
                )
                db_session.add(chat_msg)
                db_session.commit()
            except Exception as e:
                logger.error(f"保存聊天记录失败: {e}")
                db_session.rollback()
            finally:
                db_session.close()
                
        await run_sync(save_db)

        # 8. 发送回复
        if reply.strip():
            await run_sync(self.client.send_text, user_id, reply)
        else:
            logger.warning(f"回复内容为空，发送默认回复")
            await run_sync(self.client.send_text, user_id, "😊")

        logger.info(f"消息已发送给用户，准备进行后台记忆更新: user={user_id}, reply_len={len(reply)}")

        # 9. 更新记忆 (这一步可能会调用大模型进行摘要，耗时较长，所以使用真正的后台异步任务)
        try:
            await self.memory.after_reply_async(user_id, avatar_name, content, reply)
        except Exception as e:
            logger.error(f"更新记忆失败: {e}")
            
        logger.info(f"消息全部处理完成: user={user_id}")
