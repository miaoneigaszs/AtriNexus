"""
RAG 处理器
负责知识库检索和意图识别
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

from src.services.rag_service import RAGService
from src.services.intent_service import IntentService
from src.services.session_service import SessionService

logger = logging.getLogger('wecom')


class RAGProcessor:
    """RAG 检索处理器"""

    KB_EXPLICIT_HINTS = (
        "知识库",
        "文档",
        "资料",
        "手册",
        "readme",
        "README",
        "接口",
        "api",
        "架构",
        "部署",
        "配置",
        "规范",
        "制度",
        "流程",
        "项目定位",
        "项目介绍",
        "说明",
    )
    KB_QUERY_HINTS = (
        "查",
        "查一下",
        "查询",
        "看看",
        "是什么",
        "怎么",
        "哪些",
        "有没有",
        "介绍",
        "说明",
        "定位",
    )

    def __init__(self, rag_service: RAGService, intent_service: IntentService, session_service: SessionService):
        """
        初始化 RAG 处理器

        Args:
            rag_service: RAG 服务
            intent_service: 意图识别服务
            session_service: 会话服务
        """
        self.rag = rag_service
        self.intent_service = intent_service
        self.session_service = session_service

    def execute_rag_retrieval(
        self,
        user_id: str,
        content: str,
        previous_context: list,
        category_filter: Optional[str] = None
    ) -> Tuple[List[Dict], str, bool]:
        """
        执行 RAG 检索

        Args:
            user_id: 用户ID
            content: 消息内容
            previous_context: 历史上下文
            category_filter: 分类过滤（可选）

        Returns:
            Tuple: (检索结果, 知识库上下文, 是否需要搜索)
        """
        # 如果指定了分类过滤，直接检索
        if category_filter:
            retrieval = self.rag.retrieve(
                user_id,
                content,
                top_k=3,
                filter_conditions={"category": category_filter},
            )
            return (retrieval["results"], "", True)

        if not self._looks_like_kb_candidate(content):
            logger.info("[意图识别] 跳过识别(非知识库候选) -> 普通对话")
            return ([], "", False)

        # 意图识别
        intent_result = self.intent_service.recognize_intent(user_id, content, previous_context)
        intent = intent_result.get("intent", "TYPE_CHITCHAT")
        query = intent_result.get("query", content)

        logger.info(f"[意图识别] intent={intent}")

        # 闲聊（包括需要网络搜索的内容，由 LLM 工具调用处理）
        if intent != "TYPE_KNOWLEDGE_BASE":
            return ([], "", False)

        # 知识库检索（不指定分类，检索所有内容）
        retrieval = self.rag.retrieve(user_id, query, top_k=3)
        return (retrieval["results"], "", True)

    def _looks_like_kb_candidate(self, content: str) -> bool:
        normalized = (content or "").strip()
        if len(normalized) < 4:
            return False
        if normalized in {"同意", "确认", "通过", "审批通过", "取消"}:
            return False

        lowered = normalized.lower()
        action_hints = (
            "重命名",
            "改成",
            "改为",
            "追加",
            "末尾",
            "开头",
            "最后一行",
            "第一行",
            "目录",
            "执行",
            "运行",
            "git ",
            "docs/",
            "src/",
            "readme.md",
        )
        if any(hint in lowered for hint in action_hints):
            return False

        has_kb_hint = any(hint.lower() in lowered for hint in self.KB_EXPLICIT_HINTS)
        has_query_hint = any(hint in normalized for hint in self.KB_QUERY_HINTS)
        return has_kb_hint and has_query_hint

