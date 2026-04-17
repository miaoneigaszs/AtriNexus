from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol


class VectorCollection(Protocol):
    @property
    def name(self) -> str: ...

    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None: ...

    def query(
        self,
        query_texts: List[str],
        n_results: int,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]: ...

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        include: Optional[List[str]] = None,
    ) -> Dict[str, Any]: ...

    def count(self) -> int: ...

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None: ...


class VectorStore(ABC):
    @abstractmethod
    def get_or_create_collection(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VectorCollection:
        raise NotImplementedError

    @abstractmethod
    def delete_collection(self, name: str) -> None:
        raise NotImplementedError
