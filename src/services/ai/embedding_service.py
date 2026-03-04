"""
共享 Embedding 服务模块
提供统一的 Embedding 和 Reranker 功能，供 MemoryManager 和 RAGEngine 使用
"""

import logging
from typing import List, Dict
import requests

logger = logging.getLogger('wecom')


class SiliconFlowEmbedding:
    """
    SiliconFlow Embedding API 封装
    兼容 ChromaDB 的 EmbeddingFunction 协议
    """

    def __init__(self, api_key: str, base_url: str = "https://api.siliconflow.cn/v1",
                 model: str = "BAAI/bge-m3"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model

    def name(self) -> str:
        """兼容最新版 ChromaDB 的 EmbeddingFunction 协议要求"""
        return f"siliconflow_embedding_{self.model.replace('/', '_')}"

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        批量获取文档的 embedding
        用于 LangChain 风格的接口
        """
        if not documents:
            return []

        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            batch_size = 8
            all_embeddings = []

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                truncated_batch = [doc[:8000] if len(doc) > 8000 else doc for doc in batch]
                
                payload = {
                    "model": self.model,
                    "input": truncated_batch,
                    "encoding_format": "float"
                }

                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()

                for item in result.get("data", []):
                    emb = item.get("embedding")
                    if emb is None:
                        continue
                    # 处理 numpy 类型
                    if hasattr(emb, "tolist"):
                        emb = emb.tolist()
                    elif isinstance(emb, (list, tuple)):
                        emb = [float(x) for x in emb]
                    else:
                        logger.warning(f"Embedding 异常格式: type={type(emb)}")
                        continue
                    
                    if len(emb) < 100:
                        logger.warning(f"Embedding 维度过低 ({len(emb)})，已跳过")
                        continue
                        
                    all_embeddings.append(emb)

            return all_embeddings

        except Exception as e:
            logger.error(f"Embedding API 调用失败: {e}")
            raise

    def embed_query(self, query: str = None, input: List[str] = None) -> List[List[float]]:
        """
        获取单个查询的 embedding
        用于 LangChain 风格的接口
        兼容 ChromaDB 的调用方式
        必须返回 List[List[float]]
        """
        texts = input if input is not None else ([query] if query else [])
        if not texts:
            raise ValueError("embed_query 需要提供 query 或 input 参数")
        return self.embed_documents(texts)

    def __call__(self, input) -> List[List[float]]:
        """
        ChromaDB EmbeddingFunction 接口
        使实例可直接作为函数调用
        input 可能是 List[str] 或单 str，统一转为列表处理
        必须返回 List[List[float]]
        """
        if isinstance(input, str):
            input = [input]
        if not input:
            return []
        return self.embed_documents(list(input))


class SiliconFlowReranker:
    """
    SiliconFlow Reranker 服务封装
    用于搜索结果的二次精准重排序
    """

    def __init__(self, api_key: str, base_url: str = "https://api.siliconflow.cn/v1",
                 model: str = "BAAI/bge-reranker-v2-m3"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model

    def rerank(self, query: str, texts: List[str], top_n: int = 3) -> List[Dict]:
        """
        对候选文本进行重排序

        Args:
            query: 查询文本
            texts: 候选文本列表
            top_n: 返回前 N 个结果

        Returns:
            List[Dict]: 重排序结果，每项包含 index 和 relevance_score
        """
        if not texts:
            return []

        url = f"{self.base_url}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "query": query,
            "documents": texts,  # SiliconFlow API 要求使用 documents 参数
            "return_documents": False,
            "top_n": top_n
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except Exception as e:
            logger.error(f"调用 Reranker API 异常: {e}")
            return []


class EmbeddingService:
    """
    Embedding 服务统一入口
    提供单例模式和工厂方法
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, api_key: str = None, base_url: str = None):
        if self._initialized:
            return

        self.api_key = api_key
        self.base_url = base_url or "https://api.siliconflow.cn/v1"
        self._embedding_fn = None
        self._reranker = None
        self._initialized = True

    def configure(self, api_key: str, base_url: str = None):
        """配置 API 密钥和基础 URL"""
        self.api_key = api_key
        if base_url:
            self.base_url = base_url.rstrip('/')
        # 重置实例，使新配置生效
        self._embedding_fn = None
        self._reranker = None

    @property
    def embedding_function(self) -> SiliconFlowEmbedding:
        """获取 Embedding 函数实例"""
        if self._embedding_fn is None:
            if not self.api_key:
                raise ValueError("EmbeddingService 未配置 API Key")
            self._embedding_fn = SiliconFlowEmbedding(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._embedding_fn

    @property
    def reranker(self) -> SiliconFlowReranker:
        """获取 Reranker 实例"""
        if self._reranker is None:
            if not self.api_key:
                raise ValueError("EmbeddingService 未配置 API Key")
            self._reranker = SiliconFlowReranker(
                api_key=self.api_key,
                base_url=self.base_url
            )
        return self._reranker

    def is_available(self) -> bool:
        """检查服务是否可用"""
        return self.api_key is not None and len(self.api_key) > 0
