from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.errors import ChromaError
from pydantic import BaseModel, Field

from src.ingestion.models import Document
from src.vectorstore.exceptions import StoreConnectionError


class SearchResult(BaseModel):
    content: str
    metadata: dict[str, Any]
    score: float = Field(ge=0.0, le=1.0)
    id: str


class VectorStore(ABC):
    @abstractmethod
    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> list[str]: ...

    @abstractmethod
    def get_ids(
        self,
        filters: dict[str, Any],
    ) -> list[str]: ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> int: ...

    @abstractmethod
    def count(self) -> int: ...


class ChromaVectorStore(VectorStore):
    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        client: ClientAPI | None = None,
    ) -> None:
        # EphemeralClient is injected in tests; disk is used in production.
        self._client = client or chromadb.PersistentClient(path=str(persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> list[str]:
        if len(documents) != len(embeddings):
            raise ValueError("Document and embedding counts must be equal.")

        if not documents:
            return []

        ids: list[str] = []
        metadatas: list[dict[str, str | int | float | bool]] = []

        for document in documents:
            metadata = document.metadata

            if metadata.chunk_index is None:
                # None kullanılırsa aynı sayfadaki chunk'lar aynı ID'yi alabilir.
                raise ValueError("chunk_index must be assigned before the documents is indexed.")

            ids.append(f"{metadata.source}::{metadata.page}::{metadata.chunk_index}")
            metadatas.append(metadata.model_dump(exclude_none=True))

        for start in range(0, len(documents), 1000):
            end = start + 1000

            # Upsert updates instead of re-inserting the same deterministic ID.
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=[document.content for document in documents[start:end]],
                metadatas=metadatas[start:end],
            )

        return ids

    def search(
        self,
        query_embedding: list[float],
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k must be positive.")

        try:
            record_count = self.count()
            if record_count == 0:
                return []

            query_arguments: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": min(k, record_count),
            }

            if filters is not None:
                query_arguments["where"] = filters

            response = self._collection.query(**query_arguments)
        except ChromaError as exc:
            raise StoreConnectionError(
                "Vector search failed.",
                details={"cause": str(exc)},
            ) from exc

        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        ids = response["ids"][0]

        results: list[SearchResult] = []

        for document, metadata, distance, record_id in zip(
            documents,
            metadatas,
            distances,
            ids,
            strict=True,
        ):
            # Converts cosine distance from 0–2 range to similarity from 1–0 range.
            score = 1.0 - (float(distance) / 2.0)
            score = max(0.0, min(1.0, score))

            results.append(
                SearchResult(
                    content=document,
                    metadata=metadata or {},
                    score=score,
                    id=record_id,
                )
            )

        return results

    def delete(self, ids: list[str]) -> int:
        if not ids:
            return 0

        count_before = self.count()
        self._collection.delete(ids=ids)
        return count_before - self.count()

    def get_ids(self, filters: dict[str, Any]) -> list[str]:
        if not filters:
            raise ValueError("At least one metadata filter is required for ID queries.")

        response = self._collection.get(where=filters, include=[])
        return list(response["ids"])

    def count(self) -> int:
        return self._collection.count()
