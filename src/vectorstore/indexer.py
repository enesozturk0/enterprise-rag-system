from __future__ import annotations

import time

import structlog
from pydantic import BaseModel, Field

from src.ingestion.models import Document
from src.vectorstore.embeddings import EmbeddingClient
from src.vectorstore.store import VectorStore

logger = structlog.get_logger(__name__)


class IndexingResult(BaseModel):
    total_documents: int = Field(ge=0)
    indexed_documents: int = Field(ge=0)
    failed_documents: int = Field(ge=0)
    total_batches: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)


class DocumentIndexer:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        batch_size: int,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self._embedding_client = embedding_client
        self._vector_store = vector_store
        self._batch_size = batch_size

    def index(self, documents: list[Document]) -> IndexingResult:
        started_at = time.perf_counter()

        if not documents:
            return IndexingResult(
                total_documents=0,
                indexed_documents=0,
                failed_documents=0,
                total_batches=0,
                duration_seconds=time.perf_counter() - started_at,
            )

        indexed_documents = 0
        failed_documents = 0
        total_batches = 0

        for start in range(0, len(documents), self._batch_size):
            batch = documents[start : start + self._batch_size]
            total_batches += 1

            try:
                texts = [document.content for document in batch]
                embeddings = self._embedding_client.embed_batch(texts)
                stored_ids = self._vector_store.add_documents(batch, embeddings)

                indexed_documents += len(stored_ids)

                logger.debug(
                    "indexing_batch_completed",
                    batch_number=total_batches,
                    document_count=len(batch),
                )
            except Exception:
                failed_documents += len(batch)

                logger.exception(
                    "indexing_batch_failed",
                    batch_number=total_batches,
                    document_count=len(batch),
                )

        return IndexingResult(
            total_documents=len(documents),
            indexed_documents=indexed_documents,
            failed_documents=failed_documents,
            total_batches=total_batches,
            duration_seconds=time.perf_counter() - started_at,
        )

    def delete_by_source(self, source: str) -> int:
        if not source.strip():
            raise ValueError("source cannot be empty.")

        ids = self._vector_store.get_ids({"source": source})

        if not ids:
            return 0

        return self._vector_store.delete(ids)