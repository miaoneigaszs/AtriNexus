"""
RAG 服务边界。

先提供一层最小抽象，把调用方从当前 RAGEngine 解耦出来，
后续可以在不改上层业务的前提下切换到新的 SDK 实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
import sys
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


class SdkRAGService(BaseRAGService):
    """面向 `atrinexus-rag-sdk` 的适配层。"""

    def __init__(
        self,
        sdk_root: Optional[str] = None,
        sdk: Any = None,
        fallback_service: Optional[BaseRAGService] = None,
    ):
        self._sdk_root = Path(
            sdk_root
            or os.getenv("ATRINEXUS_RAG_SDK_ROOT", "")
            or str(Path(__file__).resolve().parents[3] / "rag")
        )
        self._sdk = sdk
        self._fallback_service = fallback_service

    def _ensure_sdk(self) -> Any:
        if self._sdk is not None:
            return self._sdk

        sdk_parent = str(self._sdk_root.parent)
        if sdk_parent not in sys.path:
            sys.path.insert(0, sdk_parent)

        try:
            from rag import KnowledgeSDK
            from rag.models import DeleteOptions, IndexOptions, RetrieveOptions
        except ImportError as exc:
            raise RuntimeError(
                f"无法导入 RAG SDK，请检查路径或安装状态: {self._sdk_root}"
            ) from exc

        self._sdk = KnowledgeSDK()
        self._sdk_index_options = IndexOptions
        self._sdk_retrieve_options = RetrieveOptions
        self._sdk_delete_options = DeleteOptions
        return self._sdk

    def index_document(
        self,
        user_id: str,
        file_name: str,
        file_path: str,
        *,
        category: str = "默认分类",
    ) -> tuple[bool, str]:
        sdk = self._ensure_sdk()
        options = self._sdk_index_options(
            namespace=user_id,
            extra_meta={"category": category, "file_name": file_name},
            force_reindex=False,
        )
        result = sdk.index(file_path, options=options)
        if result.status:
            return True, f"文档《{file_name}》已完成 SDK 入库"
        return False, f"文档《{file_name}》SDK 入库失败"

    def retrieve(
        self,
        user_id: str,
        query: str,
        *,
        top_k: int = 3,
        filter_conditions: Optional[Dict[str, Any]] = None,
        skip_rerank: bool = False,
    ) -> Dict[str, Any]:
        sdk = self._ensure_sdk()
        options = self._sdk_retrieve_options(
            namespace=user_id,
            top_k=top_k,
            filter_conditions=filter_conditions,
            skip_rerank=skip_rerank,
        )
        result = sdk.search(query, options=options)
        raw = dict(result.raw)
        raw.setdefault("user_id", user_id)
        return {
            "user_id": user_id,
            "query": result.query,
            "count": result.count,
            "results": raw.get("results", []),
            "formatted_context": result.formatted_context,
        }

    def list_documents(self, user_id: str) -> Dict[str, List[str]]:
        if self._fallback_service is not None:
            return self._fallback_service.list_documents(user_id)
        raise NotImplementedError("SDK 目前没有等价的文档列表接口，接入时需要补 namespace 元数据查询。")

    def get_document_outline(self, user_id: str, file_name: Optional[str] = None) -> Dict[str, Any]:
        if self._fallback_service is not None:
            return self._fallback_service.get_document_outline(user_id, file_name)
        raise NotImplementedError("SDK 目前没有等价的大纲接口，接入时需要基于检索元数据补齐。")

    def delete_document(self, user_id: str, file_name: str) -> bool:
        sdk = self._ensure_sdk()
        options = self._sdk_delete_options(namespace=user_id)
        result = sdk.delete(file_name, options=options)
        return result.deleted

    def format_retrieval_results(self, results: List[Dict[str, Any]], include_score: bool = True) -> str:
        if not results:
            return "未找到相关内容。"

        lines: List[str] = ["📚 找到以下内容：", ""]
        for item in results:
            source_file = item.get("source_file") or item.get("metadata", {}).get("file_name", "未知文件")
            heading = item.get("heading_str") or "正文"
            content = str(item.get("content", "")).replace("\n", " ")[:150]
            score = item.get("score", 0)
            lines.append(f"📄 《{source_file}》")
            if heading and heading != "正文":
                lines.append(f"   📌 {heading}")
            if include_score:
                lines.append(f"      • {content}... [相关度: {float(score):.2f}]")
            else:
                lines.append(f"      • {content}...")
            lines.append("")
        return "\n".join(lines)
