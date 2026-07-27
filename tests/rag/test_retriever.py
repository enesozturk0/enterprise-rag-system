from typing import Any

import pytest

from src.ingestion.models import Document
from src.rag.exceptions import RetrievalError
from src.rag.retriever import DocumentRetriever
from src.vectorstore.exceptions import EmbeddingError, StoreConnectionError
from src.vectorstore.store import SearchResult, VectorStore


class FakeEmbeddingClient:
    DIMENSION = 2

    def __init__(
        self,
        error: EmbeddingError | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.error = error
        self.events = events
        self.received_text: str | None = None

    def embed_text(self, text: str) -> list[float]:
        if self.events is not None:
            self.events.append("embed")

        self.received_text = text

        if self.error is not None:
            raise self.error

        return [1.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeVectorStore(VectorStore):
    def __init__(
        self,
        results: list[SearchResult] | None = None,
        error: StoreConnectionError | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.results = results or []
        self.error = error
        self.received_k: int | None = None
        self.received_filters: dict[str, Any] | None = None
        self.events = events

    def search(
        self,
        query_embedding: list[float],
        k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if self.events is not None:
            self.events.append("search")

        self.received_k = k
        self.received_filters = filters

        if self.error is not None:
            raise self.error

        return self.results[:k]

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> list[str]:
        return []

    def get_ids(self, filters: dict[str, Any]) -> list[str]:
        return []

    def delete(self, ids: list[str]) -> int:
        return 0

    def count(self) -> int:
        return len(self.results)


def test_retrieve_embeds_query_and_passes_search_options() -> None:
    result = SearchResult(
        id="doc-1",
        content="Motor yağı kontrol bilgisi",
        metadata={"source": "manual.pdf", "page": 10},
        score=0.9,
    )
    embedding_client = FakeEmbeddingClient()
    vector_store = FakeVectorStore(results=[result])
    retriever = DocumentRetriever(
        embedding_client,
        vector_store,
        default_top_k=5,
    )

    results = retriever.retrieve(
        "Motor yağı nasıl kontrol edilir?",
        top_k=1,
        filters={"source": "manual.pdf"},
    )

    assert results == [result]
    assert embedding_client.received_text == "Motor yağı nasıl kontrol edilir?"
    assert vector_store.received_k == 1
    assert vector_store.received_filters == {"source": "manual.pdf"}


def test_retrieve_wraps_embedding_error() -> None:
    original_error = EmbeddingError("Ollama unavailable")
    retriever = DocumentRetriever(
        FakeEmbeddingClient(error=original_error),
        FakeVectorStore(),
    )

    with pytest.raises(RetrievalError) as captured:
        retriever.retrieve("test query")

    assert captured.value.__cause__ is original_error


def test_retrieve_wraps_vector_store_error() -> None:
    original_error = StoreConnectionError("Chroma unavailable")
    retriever = DocumentRetriever(
        FakeEmbeddingClient(),
        FakeVectorStore(error=original_error),
    )

    with pytest.raises(RetrievalError) as captured:
        retriever.retrieve("test query")

    assert captured.value.__cause__ is original_error


@pytest.mark.parametrize("top_k", [0, -1])
def test_retrieve_rejects_invalid_top_k(top_k: int) -> None:
    retriever = DocumentRetriever(FakeEmbeddingClient(), FakeVectorStore())

    with pytest.raises(RetrievalError, match="top_k must be positive"):
        retriever.retrieve("test query", top_k=top_k)


def test_retrieve_calls_embed_before_search() -> None:
    events: list[str] = []
    retriever = DocumentRetriever(
        FakeEmbeddingClient(events=events),
        FakeVectorStore(events=events),
    )

    retriever.retrieve("test query")

    assert events == ["embed", "search"]


def test_retrieve_uses_default_top_k_and_returns_empty_results() -> None:
    vector_store = FakeVectorStore()
    retriever = DocumentRetriever(
        FakeEmbeddingClient(),
        vector_store,
        default_top_k=7,
    )

    results = retriever.retrieve("test query")

    assert results == []
    assert vector_store.received_k == 7


@pytest.mark.parametrize("query", ["", "   "])
def test_retrieve_empty_query_raises_without_embedding(query: str) -> None:
    embedding_client = FakeEmbeddingClient()
    retriever = DocumentRetriever(
        embedding_client,
        FakeVectorStore(),
    )

    with pytest.raises(RetrievalError, match="Query cannot be empty"):
        retriever.retrieve(query)

    assert embedding_client.received_text is None
