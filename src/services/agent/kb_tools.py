from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("wecom")


def build_kb_list_assets_response(rag_service: Any, user_id: str) -> str:
    """返回当前用户知识库资产的轻量清单。"""
    try:
        documents_by_category = rag_service.list_documents(user_id)
    except Exception as exc:
        logger.warning("KB 资产列表获取失败: %s", exc)
        return f"当前无法读取知识库资产：{exc}"
    if not documents_by_category:
        return "当前知识库为空。"

    try:
        outline = rag_service.get_document_outline(user_id)
    except Exception as exc:
        logger.warning("KB 文档结构获取失败: %s", exc)
        outline = {}
    documents = outline.get("documents", {}) if isinstance(outline, dict) else {}

    lines: List[str] = ["知识库资产："]
    for category, file_names in documents_by_category.items():
        for file_name in file_names:
            structure = documents.get(file_name, {})
            headings = _build_heading_preview(structure)
            lines.append(f"- {file_name} | {category} | {headings}")
    return "\n".join(lines)


def build_kb_search_response(
    rag_service: Any,
    user_id: str,
    query: str,
    *,
    top_k: int = 3,
    doc_filter: str = "",
    category: str = "",
) -> str:
    """执行 KB 检索并返回适合模型消费的文本结果。"""
    filter_conditions: Dict[str, str] = {}
    if doc_filter.strip():
        filter_conditions["source_file"] = doc_filter.strip()
    if category.strip():
        filter_conditions["category"] = category.strip()

    try:
        retrieval = rag_service.retrieve(
            user_id,
            query,
            top_k=max(1, min(int(top_k), 8)),
            filter_conditions=filter_conditions or None,
        )
    except Exception as exc:
        logger.warning("KB 检索失败: %s", exc)
        return f"当前无法执行知识库检索：{exc}"
    results = retrieval.get("results", []) if isinstance(retrieval, dict) else []
    if not results:
        return "未在知识库中找到相关内容。"
    return rag_service.format_retrieval_results(results, include_score=False)


def _build_heading_preview(structure: Dict[str, Any]) -> str:
    headings: List[str] = []
    for key in ("H1", "H2"):
        values = structure.get(key) or []
        for value in values[:2]:
            if value:
                headings.append(str(value))
    if headings:
        return f"标题：{' / '.join(headings[:4])}"
    return "标题：正文内容"
