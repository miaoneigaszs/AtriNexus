from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from src.platform_core.vector_store.base import VectorCollection, VectorStore


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        path: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_prefix: str = "",
        embedding_function: Any = None,
    ) -> None:
        if not path and not url:
            raise ValueError("path or url is required for QdrantVectorStore")

        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise ImportError("缺少 qdrant-client 依赖，请先安装后再使用 QdrantVectorStore") from exc

        if path:
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(url=url, api_key=api_key)

        self._embedding_function = embedding_function
        self._collection_prefix = collection_prefix

    def set_embedding_function(self, embedding_function: Any) -> None:
        self._embedding_function = embedding_function

    def get_or_create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VectorCollection:
        return _QdrantCollection(
            client=self._client,
            name=f"{self._collection_prefix}{name}",
            embedding_function=self._embedding_function,
            metadata=metadata or {},
        )

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(collection_name=f"{self._collection_prefix}{name}")


class _QdrantCollection:
    def __init__(
        self,
        *,
        client: Any,
        name: str,
        embedding_function: Any,
        metadata: Dict[str, Any],
    ) -> None:
        self._client = client
        self._name = name
        self._embedding_function = embedding_function
        self._metadata = metadata

    @property
    def name(self) -> str:
        return self._name

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if not documents:
            return
        embeddings = self._embed(documents)
        self._ensure_collection(len(embeddings[0]))

        from qdrant_client.models import PointStruct

        payloads = metadatas or [{} for _ in documents]
        points = []
        for point_id, vector, document, payload in zip(ids, embeddings, documents, payloads):
            point_payload = dict(payload or {})
            point_payload["document"] = document
            point_payload["_original_id"] = str(point_id)
            points.append(
                PointStruct(
                    id=self._to_point_id(point_id),
                    vector=vector,
                    payload=point_payload,
                )
            )
        self._client.upsert(collection_name=self._name, points=points)

    def query(
        self,
        query_texts: List[str],
        n_results: int,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not query_texts or not self._collection_exists():
            return _empty_query_result()

        query_vectors = self._embed(query_texts)
        query_filter = self._build_filter(where)
        all_ids: List[List[str]] = []
        all_documents: List[List[str]] = []
        all_metadatas: List[List[Dict[str, Any]]] = []
        all_distances: List[List[float]] = []

        for vector in query_vectors:
            results = self._client.query_points(
                collection_name=self._name,
                query=vector,
                limit=n_results,
                query_filter=query_filter,
                with_payload=True,
            )
            points = getattr(results, "points", results)

            ids: List[str] = []
            documents: List[str] = []
            metadatas: List[Dict[str, Any]] = []
            distances: List[float] = []

            for point in points:
                payload = dict(point.payload or {})
                document = str(payload.pop("document", ""))
                original_id = str(payload.pop("_original_id", point.id))
                score = float(getattr(point, "score", 0.0) or 0.0)
                ids.append(original_id)
                documents.append(document)
                metadatas.append(payload)
                distances.append(1.0 - score)

            all_ids.append(ids)
            all_documents.append(documents)
            all_metadatas.append(metadatas)
            all_distances.append(distances)

        return {
            "ids": all_ids,
            "documents": all_documents,
            "metadatas": all_metadatas,
            "distances": all_distances,
        }

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not self._collection_exists():
            return _empty_get_result()

        if ids:
            results = self._client.retrieve(
                collection_name=self._name,
                ids=[self._to_point_id(point_id) for point_id in ids],
                with_payload=True,
            )
        else:
            results, _ = self._client.scroll(
                collection_name=self._name,
                scroll_filter=self._build_filter(where),
                limit=limit or 10000,
                with_payload=True,
                with_vectors=False,
            )

        out_ids: List[str] = []
        out_documents: List[str] = []
        out_metadatas: List[Dict[str, Any]] = []

        for point in results:
            payload = dict(point.payload or {})
            out_ids.append(str(payload.pop("_original_id", point.id)))
            out_documents.append(str(payload.pop("document", "")))
            out_metadatas.append(payload)

        return {
            "ids": out_ids,
            "documents": out_documents,
            "metadatas": out_metadatas,
        }

    def count(self) -> int:
        if not self._collection_exists():
            return 0
        result = self._client.count(collection_name=self._name, exact=True)
        return int(getattr(result, "count", 0) or 0)

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._collection_exists():
            return

        if ids:
            self._client.delete(
                collection_name=self._name,
                points_selector=[self._to_point_id(point_id) for point_id in ids],
            )
            return

        query_filter = self._build_filter(where)
        if query_filter is None:
            raise ValueError("delete requires ids or where filter")
        self._client.delete(collection_name=self._name, points_selector=query_filter)

    def _embed(self, documents: List[str]) -> List[List[float]]:
        if self._embedding_function is None:
            raise ValueError("embedding_function is required for Qdrant collection operations")
        if hasattr(self._embedding_function, "embed_documents"):
            return self._embedding_function.embed_documents(documents)
        return self._embedding_function(documents)

    def _collection_exists(self) -> bool:
        collections = self._client.get_collections().collections
        return any(collection.name == self._name for collection in collections)

    def _to_point_id(self, value: str) -> Any:
        text = str(value)
        try:
            return int(text)
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self._name}:{text}"))

    def _ensure_collection(self, vector_size: int) -> None:
        if self._collection_exists():
            return

        from qdrant_client.models import Distance, VectorParams

        self._client.create_collection(
            collection_name=self._name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        if self._metadata:
            try:
                self._client.update_collection_aliases(change_aliases_operations=[])
            except Exception:
                pass

    def _build_filter(self, where: Optional[Dict[str, Any]]) -> Any:
        if not where:
            return None

        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        must = []
        for key, value in where.items():
            if isinstance(value, list):
                must.append(FieldCondition(key=key, match=MatchAny(any=value)))
            else:
                must.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=must)


def _empty_query_result() -> Dict[str, Any]:
    return {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }


def _empty_get_result() -> Dict[str, Any]:
    return {
        "ids": [],
        "documents": [],
        "metadatas": [],
    }
