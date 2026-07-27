from pathlib import Path
from uuid import uuid4

import chromadb

from src.ingestion.models import Document, DocumentMetadata
from src.vectorstore.exceptions import EmbeddingError
from src.vectorstore.indexer import DocumentIndexer
from src.vectorstore.store import ChromaVectorStore


class FakeEmbeddingClient:
    DIMENSION = 2

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FailFirstBatchEmbeddingClient(FakeEmbeddingClient):
    def __init__(self) -> None:
        self._call_count = 0

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._call_count += 1

        if self._call_count == 1:
            raise EmbeddingError("Test batch error")

        return super().embed_batch(texts)


def make_document(source: str, chunk_index: int) -> Document:
    return Document(
        content=f"{source} content {chunk_index}",
        metadata=DocumentMetadata(
            source=source,
            page=1,
            total_pages=1,
            chunk_index=chunk_index,
        ),
    )


def make_store() -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_dir=Path("unused-in-test"),
        collection_name=f"indexer_test_{uuid4().hex}",
        client=chromadb.EphemeralClient(),
    )


def test_index_and_delete_by_source() -> None:
    store = make_store()
    indexer = DocumentIndexer(FakeEmbeddingClient(), store, batch_size=2)
    documents = [
        make_document("rapor.pdf", 0),
        make_document("rapor.pdf", 1),
        make_document("notes.pdf", 0),
    ]

    result = indexer.index(documents)

    assert result.total_documents == 3
    assert result.indexed_documents == 3
    assert result.failed_documents == 0
    assert result.total_batches == 2
    assert store.count() == 3

    assert indexer.delete_by_source("rapor.pdf") == 2
    assert store.count() == 1


def test_failed_batch_does_not_stop_remaining_batches() -> None:
    store = make_store()
    indexer = DocumentIndexer(
        FailFirstBatchEmbeddingClient(),
        store,
        batch_size=2,
    )
    documents = [
        make_document("rapor.pdf", 0),
        make_document("rapor.pdf", 1),
        make_document("rapor.pdf", 2),
        make_document("rapor.pdf", 3),
    ]

    result = indexer.index(documents)

    assert result.total_documents == 4
    assert result.indexed_documents == 2
    assert result.failed_documents == 2
    assert result.total_batches == 2
    assert store.count() == 2


def test_index_empty_list_returns_zero_counts() -> None:
    store = make_store()
    indexer = DocumentIndexer(FakeEmbeddingClient(), store, batch_size=2)

    result = indexer.index([])

    assert result.total_documents == 0
    assert result.indexed_documents == 0
    assert result.failed_documents == 0
    assert result.total_batches == 0
    assert result.duration_seconds >= 0
    assert store.count() == 0


def test_index_is_idempotent() -> None:
    store = make_store()
    indexer = DocumentIndexer(FakeEmbeddingClient(), store, batch_size=2)
    documents = [
        make_document("rapor.pdf", 0),
        make_document("rapor.pdf", 1),
    ]

    first_result = indexer.index(documents)
    count_after_first_index = store.count()

    second_result = indexer.index(documents)

    assert first_result.indexed_documents == 2
    assert second_result.indexed_documents == 2
    assert count_after_first_index == 2
    assert store.count() == count_after_first_index
