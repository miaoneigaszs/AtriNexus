"""
知识库检索增强引擎 (RAG Engine)
负责处理上传的 PDF、Word、TXT、MD 文件。
实现：文本提取、高级递归切块 (Chunking)、向量化入库、元数据过滤搜索。
"""

import os
import re
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict

import jieba
from rank_bm25 import BM25Okapi

from src.services.ai.embedding_service import EmbeddingService
from src.services.rag_legacy_document import parse_document, recursive_character_split, split_markdown
from src.services.vector_store import QdrantVectorStore, VectorCollection, VectorStore
from data.config import config

logger = logging.getLogger('wecom')


class RAGEngine:
    """知识库检索增强引擎"""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store
        if not self.vector_store:
            qdrant_url = os.getenv("ATRINEXUS_QDRANT_URL", "").strip() or None
            qdrant_api_key = os.getenv("ATRINEXUS_QDRANT_API_KEY", "").strip() or None
            qdrant_path = os.getenv(
                "ATRINEXUS_QDRANT_PATH",
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "data",
                    "vectordb_qdrant",
                ),
            )
            if qdrant_url:
                self.vector_store = QdrantVectorStore(url=qdrant_url, api_key=qdrant_api_key)
            else:
                os.makedirs(qdrant_path, exist_ok=True)
                self.vector_store = QdrantVectorStore(path=qdrant_path)
            
        # BM25 Sparse 检索需要的语料库及分词后索引（内存缓存，启动时自动从向量存储重建）
        # 结构: user_id -> {"corpus": [list of strings], "bm25": BM25Okapi object, "ids": [list of ids]}
        self._bm25_store = {}

        # 初始化共享 Embedding 服务（使用独立的 embedding 配置）
        self._embedding_service = EmbeddingService()
        # 优先使用 embedding_settings，回退到 llm_settings
        if hasattr(config, 'embedding') and config.embedding and config.embedding.api_key:
            embed_key = config.embedding.api_key
            embed_url = config.embedding.base_url or 'https://api.siliconflow.cn/v1'
            logger.info(f"使用 embedding_settings 配置: base_url={embed_url}")
        else:
            embed_key = config.llm.api_key
            embed_url = config.llm.base_url
            logger.warning("embedding_settings 未配置，回退使用 llm_settings（可能不支持 embedding API）")
        if embed_key:
            self._embedding_service.configure(api_key=embed_key, base_url=embed_url)

        if self._embedding_service.is_available():
            self.embedding_fn = self._embedding_service.embedding_function
            self.reranker = self._embedding_service.reranker
            if hasattr(self.vector_store, "set_embedding_function"):
                self.vector_store.set_embedding_function(self.embedding_fn)
        else:
            logger.warning("未配置 API Key，RAG 引擎不可用")
            self.embedding_fn = None
            self.reranker = None

        # 可配置参数（未来可从 config.json 读取）
        self.batch_size = getattr(config.behavior, 'rag_batch_size', 8)  # 批量写入大小，避免 413 错误
        self.chunk_size = getattr(config.behavior, 'rag_chunk_size', 800)  # 文档切块大小
        self.chunk_overlap = getattr(config.behavior, 'rag_chunk_overlap', 150)  # 切块重叠字符数

    def _calculate_batch_size(self, chunks: List[str]) -> int:
        """
        智能计算批量大小，基于文档块平均长度动态调整
        
        API 限制约 8K tokens，按平均字符数估算：
        - 大块 (>2000字符): 4 个/批
        - 中块 (1000-2000字符): 6 个/批
        - 小块 (<1000字符): 8 个/批
        """
        if not chunks:
            return self.batch_size
        
        total_chars = sum(len(c) for c in chunks)
        avg_chunk_size = total_chars / len(chunks)
        
        if avg_chunk_size > 2000:
            return 4
        elif avg_chunk_size > 1000:
            return 6
        return self.batch_size  # 默认 8

    def _build_collection_name(self, user_id: str) -> str:
        """构建稳定的用户知识库集合名。"""
        normalized_user_id = re.sub(r'[^a-zA-Z0-9_-]+', '_', user_id).strip('_').lower()
        if not normalized_user_id:
            normalized_user_id = hashlib.md5(user_id.encode('utf-8')).hexdigest()[:12]
        if len(normalized_user_id) > 48:
            normalized_user_id = normalized_user_id[:48]
        return f"kb_{normalized_user_id}"

    def _get_kb_collection(self, user_id: str) -> Optional[VectorCollection]:
        """获取专属知识库 Collection，按用户隔离"""
        if not self.embedding_fn or not self.vector_store:
            return None
        collection_name = self._build_collection_name(user_id)
        try:
            return self.vector_store.get_or_create_collection(
                name=collection_name,
                metadata={"description": "User Knowledge Base"}
            )
        except Exception as e:
            logger.error(f"获取知识库集合失败: user_id={user_id}, collection={collection_name}, error={e}")
            return None

    def _init_or_update_bm25(self, user_id: str, collection=None):
        """初始化或更新用户的 BM25 索引"""
        if collection is None:
            collection = self._get_kb_collection(user_id)
        if not collection:
            return
            
        try:
            # 取出所有文档
            all_data = collection.get(include=["documents"])
            docs = all_data.get("documents", [])
            ids = all_data.get("ids", [])
            
            if not docs:
                self._bm25_store[user_id] = None
                return
                
            tokenized_corpus = [list(jieba.cut(doc)) for doc in docs]
            bm25 = BM25Okapi(tokenized_corpus)
            self._bm25_store[user_id] = {
                "corpus": docs,
                "ids": ids,
                "bm25": bm25
            }
            logger.debug(f"BM25 索引初始化/更新完成: 用户 {user_id}, {len(docs)} 个 Chunk")
        except Exception as e:
            logger.error(f"初始化 BM25 失败: {e}")

    def add_document(self, user_id: str, file_name: str, file_path: str, category: str = "默认分类") -> tuple[bool, str]:
        """
        完整的一键入库闭环流：读取文件 -> 解析格式 -> MD5防重校验 -> 结构化拆分 -> 递归切块 -> 入库
        """
        collection = self._get_kb_collection(user_id)
        if not collection:
            return False, "向量库初始化失败"

        # 1. 提取文本
        text = parse_document(file_path)
        if not text.strip():
            return False, "解析出的文本为空，暂不支持纯图片或损坏的文件"

        # 2. 文档级 MD5 防污防重校验
        doc_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        try:
            exist_results = collection.get(where={"doc_hash": doc_hash}, limit=1)
            if exist_results and exist_results['ids']:
                logger.info(f"文档防重触发: {file_name} (hash={doc_hash}) 已存在，跳过入库。")
                try:
                    os.remove(file_path)
                except:
                    pass
                return True, "该文档内容已存在于您的知识库中，无需重复投喂！"
        except Exception as e:
            logger.warning(f"防重校验异常，继续入库: {e}")

        # 3. 结构化拆分 -> 基础层级感知
        structured_sections = split_markdown(text)

        # 4. 对每个结构块进行带 Overlap 的智能切片
        final_chunks = []
        for section in structured_sections:
            sub_chunks = recursive_character_split(section["content"], chunk_size=self.chunk_size, overlap=self.chunk_overlap)
            for sub in sub_chunks:
                final_chunks.append({
                    "content": sub,
                    "headers": section["headers"]
                })

        if not final_chunks:
            return False, "文件文本分割失败"

        # 5. 准备写入向量存储
        ids = []
        documents = []
        metadatas = []
        upload_time = datetime.now().isoformat()

        for i, item in enumerate(final_chunks):
            chunk_content = item["content"]
            headers_meta = item["headers"]

            ids.append(f"kb_{doc_hash}_chunk_{i}")
            documents.append(chunk_content)

            # 将 Header 原文也拼接到摘要中，保证语义丰富
            meta_dict = {
                "file_name": file_name,
                "category": category,
                "chunk_index": i,
                "upload_time": upload_time,
                "doc_hash": doc_hash
            }
            # 把 H1, H2 直接铺平放入 metadata 供后续精准检索
            meta_dict.update(headers_meta)
            metadatas.append(meta_dict)

        # 6. 执行写入（智能分批，避免 413 错误）
        try:
            total_chunks = len(ids)
            smart_batch_size = self._calculate_batch_size(documents)
            
            for i in range(0, total_chunks, smart_batch_size):
                batch_ids = ids[i:i + smart_batch_size]
                batch_docs = documents[i:i + smart_batch_size]
                batch_metas = metadatas[i:i + smart_batch_size]
                
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas
                )
                logger.debug(f"批次 {i//smart_batch_size + 1}/{(total_chunks-1)//smart_batch_size + 1} 写入完成")
            
            logger.info(f"文档入库成功: {file_name}, Hash={doc_hash}, 共切分 {total_chunks} 块, 批大小={smart_batch_size}")
            # 更新 BM25 索引
            self._init_or_update_bm25(user_id, collection)
            try:
                os.remove(file_path)
            except:
                pass
            return True, f"成功存入知识库，为 AI 构建了 {total_chunks} 个记忆神经元。"
        except Exception as e:
            logger.error(f"知识块写入数据库异常: {e}")
            return False, f"存入失败: {str(e)}"

    def retrieve_knowledge(self, user_id: str, query: str, top_k: int = 3,
                           category_filter: Optional[str] = None,
                           h1_filter: Optional[str] = None,
                           h2_filter: Optional[str] = None,
                           skip_rerank: bool = False) -> List[Dict]:
        """
        检索知识库（支持分类和标题层级过滤）

        流程：
        1. 向量相似度粗筛（召回 top_k * 5，默认查 15 条）
        2. Reranker 重排序（Cross-Encoder 精打分，提取最终 Top-K）

        Args:
            user_id: 用户ID
            query: 查询文本
            top_k: 返回结果数量
            category_filter: 分类过滤
            h1_filter: H1 标题过滤
            h2_filter: H2 标题过滤
            skip_rerank: 跳过 Reranker，直接返回向量相似度结果（用于快速判断）
        """
        collection = self._get_kb_collection(user_id)
        if not collection:
            return []

        if collection.count() == 0:
            return []

        # 构建复合过滤条件
        where_clause = {}
        if category_filter:
            where_clause["category"] = category_filter
        if h1_filter:
            where_clause["H1"] = h1_filter
        if h2_filter:
            where_clause["H2"] = h2_filter

        # 确定检索数量：快速模式只取 top_k，海选模式多取一些
        fetch_k = top_k if skip_rerank else min(max(top_k * 5, 15), collection.count())
        
        query_args = {
            "query_texts": [query],
            "n_results": fetch_k
        }
        if where_clause:
            query_args["where"] = where_clause

        try:
            fetch_k_dense = fetch_k
            fetch_k_sparse = fetch_k

            # === 并行双路召回 (Dense + Sparse) ===
            import concurrent.futures

            dense_results_raw = {}
            sparse_results_raw = {}
            
            # 使用 ThreadPoolExecutor 并行发起 Dense 和 Sparse(BM25) 查询
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                def run_dense():
                    return collection.query(**query_args)

                def run_sparse():
                    if user_id not in self._bm25_store:
                        # 如未初始化，先初始化
                        self._init_or_update_bm25(user_id, collection)
                    
                    bm25_data = self._bm25_store.get(user_id)
                    if not bm25_data:
                        return []
                        
                    tokenized_query = list(jieba.cut(query))
                    scores = bm25_data["bm25"].get_scores(tokenized_query)
                    
                    # 按照分数降序排列，取前 fetch_k_sparse 名
                    top_n_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:fetch_k_sparse]
                    
                    sparse_res = []
                    for idx in top_n_idx:
                        if scores[idx] > 0: # 过滤掉得分为0的
                            sparse_res.append((bm25_data["ids"][idx], scores[idx]))
                    return sparse_res

                future_dense = executor.submit(run_dense)
                future_sparse = executor.submit(run_sparse)
                
                # 获取结果
                try:
                    dense_results_raw = future_dense.result(timeout=10)
                except Exception as e:
                    logger.error(f"Dense 向量查询失败: {e}")
                    
                try:
                    sparse_results_raw = future_sparse.result(timeout=10)
                except Exception as e:
                    logger.error(f"Sparse BM25查询失败: {e}")

            # === 解析与融合 (RRF) ===
            rrf_k = 60 # RRF 算法的平滑常数
            rrf_scores = defaultdict(float)
            
            # 记录 id -> object (由于我们需要返回带 metadata 的结果，所以要把查到的 doc 和 meta 全部映射起来)
            doc_meta_map = {}

            # 处理 Dense 榜单，注入到 RRF
            if dense_results_raw and dense_results_raw.get('documents') and dense_results_raw['documents'][0]:
                docs = dense_results_raw['documents'][0]
                ids = dense_results_raw['ids'][0]
                metas = dense_results_raw['metadatas'][0] if dense_results_raw.get('metadatas') else [{}] * len(docs)
                distances = dense_results_raw['distances'][0] if dense_results_raw.get('distances') else [0] * len(docs)
                
                for rank, (doc_id, doc, meta, dist) in enumerate(zip(ids, docs, metas, distances)):
                    # Dense RRF Score
                    rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1)
                    
                    # 记录实体
                    if doc_id not in doc_meta_map:
                        doc_meta_map[doc_id] = {"content": doc, "metadata": meta, "dense_dist": dist}

            # 处理 Sparse 榜单，注入到 RRF
            if sparse_results_raw:
                # Sparse 是 [(id, score), ...]
                # 为了拿到 docs 和 metadata，如果 doc_id 不在 map 里，需要回查向量存储
                missing_ids = [doc_id for doc_id, _ in sparse_results_raw if doc_id not in doc_meta_map]
                if missing_ids:
                    missing_data = collection.get(ids=missing_ids, include=["documents", "metadatas"])
                    if missing_data and missing_data.get("ids"):
                        m_ids = missing_data["ids"]
                        m_docs = missing_data["documents"]
                        m_metas = missing_data["metadatas"]
                        for i in range(len(m_ids)):
                            doc_meta_map[m_ids[i]] = {"content": m_docs[i], "metadata": m_metas[i]}

                for rank, (doc_id, bm25_score) in enumerate(sparse_results_raw):
                    if doc_id in doc_meta_map:
                        # Sparse RRF Score
                        rrf_scores[doc_id] += 1.0 / (rrf_k + rank + 1)
            
            # 依据 RRF 得分排序，截取前 fetch_k 个用于后续重排
            sorted_rrf_ids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:fetch_k]
            
            # 组装为最终混合海选列表，同时进行[层级上下文增强]
            fused_docs = []
            fused_metas = []
            fused_original_distances = []
            
            for doc_id in sorted_rrf_ids:
                item = doc_meta_map[doc_id]
                original_text = item["content"]
                meta = item["metadata"]
                dist = item.get("dense_dist", 0.5) # BM25找回的默认给一个居中的欧式距离
                
                # --- 层级上下文增强 ---
                h1 = meta.get("H1", "")
                h2 = meta.get("H2", "")
                h3 = meta.get("H3", "")
                
                # 拼接目录树前缀，让大模型在打分(Rerank)和推理时感知结构
                path_parts = [p for p in [h1, h2, h3] if p]
                if path_parts:
                    path_str = " > ".join(path_parts)
                    enriched_text = f"[所属章节: {path_str}]\n{original_text}"
                else:
                    enriched_text = original_text
                    
                fused_docs.append(enriched_text)
                fused_metas.append(meta)
                fused_original_distances.append(dist)

            if not fused_docs:
                return []

            if skip_rerank:
                logger.debug(f"[快速混合检索] 查询='{query[:20]}...' 返回 {len(fused_docs)} 条，跳过重排")
            else:
                logger.info(f"RAG海选(BM25+Dense融合): '{query[:20]}...' 获取候选 {len(fused_docs)} 块。开始调用大模型二次重排。")

                # 2. 第二阶段：重排模型精确打分 (Reranking)
                if self.reranker and len(fused_docs) > 1:
                    rerank_results = self.reranker.rerank(query=query, texts=fused_docs, top_n=top_k)
                    if rerank_results:
                        final_ret = []
                        for rank_info in rerank_results:
                            idx = rank_info["index"]
                            score = rank_info["relevance_score"]
                            final_ret.append({
                                "content": fused_docs[idx],
                                "metadata": fused_metas[idx],
                                "score": score
                            })
                        logger.info(f"Reranker 重排完成，提取最终 {len(final_ret)} 个高分片段。")
                        return final_ret
                    else:
                        logger.warning("重排接口未返回数据，降级为默认RRF排序")

            # 降级或直接返回：使用融合分数排列
            ret = []
            for i in range(min(top_k, len(fused_docs))):
                ret.append({
                    "content": fused_docs[i],
                    "metadata": fused_metas[i],
                    "score": rrf_scores[sorted_rrf_ids[i]] * 100 # 将 RRF 放大便于展示
                })
            return ret

        except Exception as e:
            logger.error(f"检索知识库失败: {e}")
            return []

    # ---------- 新增：文档结构大纲功能 ----------

    def get_document_outline(self, user_id: str, file_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取文档的标题结构大纲

        Args:
            user_id: 用户ID
            file_name: 指定文件名，为 None 则返回所有文档的大纲

        Returns:
            Dict: {
                "documents": {
                    "file1.md": {
                        "H1": ["第一章 概述", "第二章 详细说明"],
                        "H2": ["1.1 背景", "1.2 目的"],
                        "category": "产品文档"
                    }
                },
                "categories": ["产品文档", "规章制度"]
            }
        """
        collection = self._get_kb_collection(user_id)
        if not collection or collection.count() == 0:
            return {"documents": {}, "categories": []}

        try:
            # 构建过滤条件
            where_clause = {}
            if file_name:
                where_clause["file_name"] = file_name

            # 获取元数据
            if where_clause:
                res = collection.get(where=where_clause, include=["metadatas"])
            else:
                res = collection.get(include=["metadatas"])

            metas = res.get("metadatas", [])

            # 按文档聚合标题结构
            docs_structure = defaultdict(lambda: {"H1": set(), "H2": set(), "H3": set(), "category": ""})
            categories = set()

            for meta in metas:
                fname = meta.get("file_name", "未知文件")
                categories.add(meta.get("category", "默认分类"))
                docs_structure[fname]["category"] = meta.get("category", "默认分类")

                # 收集各级标题
                for level in [1, 2, 3]:
                    key = f"H{level}"
                    if meta.get(key):
                        docs_structure[fname][key].add(meta[key])

            # 转换为列表并排序
            result = {"documents": {}, "categories": sorted(list(categories))}
            for fname, structure in docs_structure.items():
                result["documents"][fname] = {
                    "category": structure["category"],
                    "H1": sorted(list(structure["H1"])),
                    "H2": sorted(list(structure["H2"])),
                    "H3": sorted(list(structure["H3"]))
                }

            return result

        except Exception as e:
            logger.error(f"获取文档大纲失败: {e}")
            return {"documents": {}, "categories": []}

    def get_knowledge_list(self, user_id: str) -> Dict[str, List[str]]:
        """获取用户当前知识库里拥有的文档一览，按 category 分组"""
        collection = self._get_kb_collection(user_id)
        if not collection or collection.count() == 0:
            return {}

        try:
            res = collection.get(include=["metadatas"])
            metas = res.get("metadatas", [])

            kb_map = {}
            for m in metas:
                cat = m.get("category", "默认分类")
                fname = m.get("file_name", "未知文件")
                if cat not in kb_map:
                    kb_map[cat] = set()
                kb_map[cat].add(fname)

            return {k: list(v) for k, v in kb_map.items()}
        except Exception as e:
            logger.error(f"获取知识库列表失败: {e}")
            return {}

    def delete_document(self, user_id: str, file_name: str) -> bool:
        """根据文件名从知识库移除相关碎片"""
        collection = self._get_kb_collection(user_id)
        if not collection:
            return False

        try:
            collection.delete(where={"file_name": file_name})
            # 文档被删除后同步刷新 BM25 索引
            self._init_or_update_bm25(user_id, collection)
            return True
        except Exception as e:
            logger.error(f"删除文档碎片失败: {e}")
            return False

    # ---------- 新增：检索结果格式化 ----------

    def format_search_results(self, results: List[Dict], include_score: bool = True) -> str:
        """
        将检索结果格式化为易读的分类展示

        Args:
            results: 检索结果列表
            include_score: 是否包含相关度分数

        Returns:
            str: 格式化后的文本
        """
        if not results:
            return "未找到相关内容。"

        # 按文档和标题分组
        grouped = defaultdict(lambda: defaultdict(list))
        for item in results:
            meta = item.get("metadata", {})
            file_name = meta.get("file_name", "未知文件")
            h1 = meta.get("H1", "")
            h2 = meta.get("H2", "")

            header_path = " > ".join(filter(None, [h1, h2])) or "正文"
            grouped[file_name][header_path].append(item)

        # 格式化输出
        output_lines = ["📚 找到以下内容：\n"]

        for file_name, headers in grouped.items():
            output_lines.append(f"📄 《{file_name}》")
            for header_path, items in headers.items():
                if header_path != "正文":
                    output_lines.append(f"   📌 {header_path}")
                for item in items:
                    content = item.get("content", "")[:150].replace("\n", " ")
                    score = item.get("score", 0)
                    if include_score:
                        output_lines.append(f"      • {content}... [相关度: {score:.2f}]")
                    else:
                        output_lines.append(f"      • {content}...")
            output_lines.append("")

        return "\n".join(output_lines)
