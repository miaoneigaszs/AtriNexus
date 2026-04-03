from __future__ import annotations

from typing import Any, Dict, Optional

from src.services.vector_store.base import VectorCollection, VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        path: Optional[str] = None,
        client: Any = None,
        embedding_function: Any = None,
    ) -> None:
        if client is not None:
            self._client = client
        else:
            if not path:
                raise ValueError("path is required when client is not provided")
            import chromadb
            from chromadb.config import Settings

            self._client = chromadb.PersistentClient(
                path=path,
                settings=Settings(anonymized_telemetry=False),
            )
        self._embedding_function = embedding_function

    def get_or_create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VectorCollection:
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embedding_function,
            metadata=metadata,
        )

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(name)
