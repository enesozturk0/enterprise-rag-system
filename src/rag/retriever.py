"""Retrieve relevant document chunks for a user query."""

from __future__ import annotations

from typing import Any

import structlog

from src.rag.exceptions import RetrievalError
from src.vectorstore.embeddings import EmbeddingClient
from src.vectorstore.exceptions import BaseVectorStoreError, EmbeddingError
from src.vectorstore.store import SearchResult, VectorStore

logger = structlog.get_logger(__name__)


class DocumentRetriever:
    """Embed a query and retrieve its most relevant document chunks."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        default_top_k: int = 5,
    ) -> None:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be positive.")

        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            raise RetrievalError("Query cannot be empty.")

        if top_k is not None and top_k <= 0:
            raise RetrievalError("top_k must be positive.")

        effective_top_k = self._default_top_k if top_k is None else top_k

        try:
            query_embedding = self._embedding_client.embed_text(query)
        except EmbeddingError as exc:
            raise RetrievalError(
                "Failed to embed the query.",
                details={"query_length": len(query), "cause": str(exc)},
            ) from exc

        try:
            results = self._vector_store.search(
                query_embedding=query_embedding,
                k=effective_top_k,
                filters=filters,
            )
        except BaseVectorStoreError as exc:
            raise RetrievalError(
                "Failed to search the vector store.",
                details={"query_length": len(query), "cause": str(exc)},
            ) from exc

        logger.info(
            "retrieval_completed",
            query_length=len(query),
            top_k=effective_top_k,
            results_found=len(results),
        )

        return results
