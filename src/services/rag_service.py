"""
RAG 服务边界。

先提供一层最小抽象，把调用方从当前 RAGEngine 解耦出来，
后续可以在不改上层业务的前提下切换到新的 SDK 实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from src.services.rag_engine import RAGEngine


@runtime_checkable
class RAGService(Protocol):
    """面向调用方的最小 RAG 服务协议。"""

    def index_document(
        self,
        user_id: str,
        file_name: str,
        file_path: str,
        *,
        category: str = "默认分类",
    ) -> tuple[bool, str]:
        """索引文档到用户知识库。"""

    def retrieve(
        self,
        user_id: str,
        query: str,
        *,
        top_k: int = 3,
        filter_conditions: Optional[Dict[str, Any]] = None,
        skip_rerank: bool = False,
    ) -> Dict[str, Any]:
        """检索知识库，返回统一结果结构。"""

    def list_documents(self, user_id: str) -> Dict[str, List[str]]:
        """按分类返回文档列表。"""

    def get_document_outline(self, user_id: str, file_name: Optional[str] = None) -> Dict[str, Any]:
        """获取文档结构大纲。"""

    def delete_document(self, user_id: str, file_name: str) -> bool:
        """删除指定文档。"""

    def format_retrieval_results(self, results: List[Dict[str, Any]], include_score: bool = True) -> str:
        """格式化检索结果。"""


class BaseRAGService(ABC):
    """抽象基类，便于后续接入 SDK 实现。"""

    @abstractmethod
    def index_document(
        self,
        user_id: str,
        file_name: str,
        file_path: str,
        *,
        category: str = "默认分类",
    ) -> tuple[bool, str]:
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self,
        user_id: str,
        query: str,
        *,
        top_k: int = 3,
        filter_conditions: Optional[Dict[str, Any]] = None,
        skip_rerank: bool = False,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_documents(self, user_id: str) -> Dict[str, List[str]]:
        raise NotImplementedError

    @abstractmethod
    def get_document_outline(self, user_id: str, file_name: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def delete_document(self, user_id: str, file_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def format_retrieval_results(self, results: List[Dict[str, Any]], include_score: bool = True) -> str:
        raise NotImplementedError


class LegacyRAGService(BaseRAGService):
    """对现有 RAGEngine 的兼容适配层。"""

    def __init__(self, engine: RAGEngine):
        self.engine = engine

    def index_document(
        self,
        user_id: str,
        file_name: str,
        file_path: str,
        *,
        category: str = "默认分类",
    ) -> tuple[bool, str]:
        return self.engine.add_document(
            user_id=user_id,
            file_name=file_name,
            file_path=file_path,
            category=category,
        )

    def retrieve(
        self,
        user_id: str,
        query: str,
        *,
        top_k: int = 3,
        filter_conditions: Optional[Dict[str, Any]] = None,
        skip_rerank: bool = False,
    ) -> Dict[str, Any]:
        filters = dict(filter_conditions or {})
        results = self.engine.retrieve_knowledge(
            user_id,
            query,
            top_k=top_k,
            category_filter=filters.get("category"),
            h1_filter=filters.get("H1"),
            h2_filter=filters.get("H2"),
            skip_rerank=skip_rerank,
        )
        return {
            "user_id": user_id,
            "query": query,
            "count": len(results),
            "results": results,
            "formatted_context": self.format_retrieval_results(results, include_score=False),
        }

    def list_documents(self, user_id: str) -> Dict[str, List[str]]:
        return self.engine.get_knowledge_list(user_id)

    def get_document_outline(self, user_id: str, file_name: Optional[str] = None) -> Dict[str, Any]:
        return self.engine.get_document_outline(user_id, file_name)

    def delete_document(self, user_id: str, file_name: str) -> bool:
        return self.engine.delete_document(user_id, file_name)

    def format_retrieval_results(self, results: List[Dict[str, Any]], include_score: bool = True) -> str:
        return self.engine.format_search_results(results, include_score=include_score)
