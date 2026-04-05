"""
RAG 服务边界。

当前默认实现是 SDK，主项目只保留当前主路径所需的 RAG 服务接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from data.config import config


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
    """抽象基类，约束主项目对 RAG 能力的最小依赖面。"""

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


class SdkRAGService(BaseRAGService):
    """面向 `atrinexus-rag-sdk` 的适配层。"""

    def __init__(
        self,
        sdk_root: Optional[str] = None,
        sdk: Any = None,
    ):
        raw_sdk_root = sdk_root or os.getenv("ATRINEXUS_RAG_SDK_ROOT", "")
        self._sdk_root = Path(raw_sdk_root) if raw_sdk_root else None
        self._sdk = sdk

    @staticmethod
    def _build_sdk_config(RAGConfig: Any) -> Any:
        base_config = RAGConfig.from_env()

        embed_cfg = getattr(config, "embedding", None)
        llm_cfg = getattr(config, "llm", None)

        embed_api_key = getattr(embed_cfg, "api_key", "") or getattr(llm_cfg, "api_key", "")
        embed_base_url = getattr(embed_cfg, "base_url", "") or "https://api.siliconflow.cn/v1"
        embed_model = getattr(embed_cfg, "model", "") or "BAAI/bge-m3"
        reranker_model = getattr(embed_cfg, "reranker_model", "") or "BAAI/bge-reranker-v2-m3"

        base_config.embedding.provider = "proxy"
        base_config.embedding.proxy_api_key = embed_api_key
        base_config.embedding.proxy_base_url = embed_base_url
        base_config.embedding.proxy_model = embed_model

        if embed_model == "BAAI/bge-m3":
            base_config.embedding.dimension = 1024

        base_config.reranker.api_key = embed_api_key
        base_config.reranker.base_url = embed_base_url
        base_config.reranker.model = reranker_model

        qdrant_url = (
            os.getenv("ATRINEXUS_SDK_QDRANT_URL", "").strip()
            or os.getenv("ATRINEXUS_QDRANT_URL", "").strip()
        )
        qdrant_api_key = (
            os.getenv("ATRINEXUS_SDK_QDRANT_API_KEY", "").strip()
            or os.getenv("ATRINEXUS_QDRANT_API_KEY", "").strip()
        )
        qdrant_path = os.getenv("ATRINEXUS_SDK_QDRANT_PATH", "").strip()
        if qdrant_url:
            base_config.qdrant.mode = "cloud"
            base_config.qdrant.url = qdrant_url
            base_config.qdrant.api_key = qdrant_api_key
        else:
            if not qdrant_path:
                root_dir = Path(__file__).resolve().parents[2]
                qdrant_path = str(root_dir / "data" / "rag_sdk_qdrant")
            base_config.qdrant.mode = "local"
            base_config.qdrant.path = qdrant_path

        base_config.chunk.rag_mode = os.getenv("ATRINEXUS_RAG_MODE", "basic").strip() or "basic"
        base_config.chunk.use_contextual_retrieval = (
            os.getenv("ATRINEXUS_USE_CONTEXTUAL_RETRIEVAL", "").strip().lower() in {"1", "true", "yes", "on"}
        )
        base_config.chunk.context_model = (
            os.getenv("ATRINEXUS_CONTEXT_MODEL", "").strip()
            or getattr(llm_cfg, "model", "")
        )
        return base_config

    def _ensure_sdk(self) -> Any:
        if self._sdk is not None:
            return self._sdk

        try:
            from rag import KnowledgeSDK
            from rag.config import RAGConfig
            from rag.models import DeleteOptions, DocumentSource, IndexOptions, RetrieveOptions
        except ImportError as exc:
            if not self._sdk_root:
                raise RuntimeError(
                    "无法导入 RAG SDK，请先通过 GitHub 依赖安装，或设置 ATRINEXUS_RAG_SDK_ROOT 指向本地源码目录。"
                ) from exc

            sdk_root = str(self._sdk_root)
            if sdk_root not in sys.path:
                sys.path.insert(0, sdk_root)

            try:
                from rag import KnowledgeSDK
                from rag.config import RAGConfig
                from rag.models import DeleteOptions, DocumentSource, IndexOptions, RetrieveOptions
            except ImportError as retry_exc:
                raise RuntimeError(
                    f"无法导入 RAG SDK，请检查安装状态或源码路径: {self._sdk_root}"
                ) from retry_exc

        base_config = self._build_sdk_config(RAGConfig)
        self._sdk = KnowledgeSDK(base_config=base_config)
        self._sdk_document_source = DocumentSource
        self._sdk_index_options = IndexOptions
        self._sdk_retrieve_options = RetrieveOptions
        self._sdk_delete_options = DeleteOptions
        return self._sdk

    def _get_engine(self, user_id: str) -> Any:
        sdk = self._ensure_sdk()
        return sdk._get_engine(namespace=user_id)

    def _collect_payloads(self, user_id: str, file_name: Optional[str] = None) -> List[Dict[str, Any]]:
        engine = self._get_engine(user_id)
        vector_store = engine.vector_store

        if not hasattr(vector_store, "_scroll_all"):
            raise RuntimeError("当前 SDK vector store 不支持 payload 扫描")

        points = vector_store._scroll_all(None, with_payload=True)
        payloads: List[Dict[str, Any]] = []
        for point in points:
            payload = dict(getattr(point, "payload", None) or {})
            source_file = str(payload.get("source_file", "")).strip()
            if not source_file:
                continue
            if file_name and source_file != file_name:
                continue
            payloads.append(payload)
        return payloads

    def index_document(
        self,
        user_id: str,
        file_name: str,
        file_path: str,
        *,
        category: str = "默认分类",
    ) -> tuple[bool, str]:
        sdk = self._ensure_sdk()
        source = self._sdk_document_source.from_bytes(
            Path(file_path).read_bytes(),
            source_name=file_name,
            upload_origin="atrinexus",
        )
        options = self._sdk_index_options(
            namespace=user_id,
            extra_meta={"category": category, "file_name": file_name},
            force_reindex=False,
        )
        result = sdk.index(source, options=options)
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
        documents_by_category: Dict[str, set[str]] = defaultdict(set)
        for payload in self._collect_payloads(user_id):
            category = str(payload.get("category", "默认分类") or "默认分类")
            documents_by_category[category].add(str(payload["source_file"]))
        return {
            category: sorted(file_names)
            for category, file_names in sorted(documents_by_category.items())
        }

    def get_document_outline(self, user_id: str, file_name: Optional[str] = None) -> Dict[str, Any]:
        docs_structure: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"H1": set(), "H2": set(), "H3": set(), "category": "默认分类"}
        )
        categories: set[str] = set()

        for payload in self._collect_payloads(user_id, file_name=file_name):
            source_file = str(payload["source_file"])
            category = str(payload.get("category", "默认分类") or "默认分类")
            docs_structure[source_file]["category"] = category
            categories.add(category)

            heading_path = payload.get("heading_path") or []
            if len(heading_path) > 0 and heading_path[0]:
                docs_structure[source_file]["H1"].add(str(heading_path[0]))
            if len(heading_path) > 1 and heading_path[1]:
                docs_structure[source_file]["H2"].add(str(heading_path[1]))
            if len(heading_path) > 2 and heading_path[2]:
                docs_structure[source_file]["H3"].add(str(heading_path[2]))

        documents = {
            source_file: {
                "H1": sorted(values["H1"]),
                "H2": sorted(values["H2"]),
                "H3": sorted(values["H3"]),
                "category": values["category"],
            }
            for source_file, values in sorted(docs_structure.items())
        }
        return {
            "documents": documents,
            "categories": sorted(categories),
        }

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
