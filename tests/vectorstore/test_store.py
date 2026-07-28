from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import chromadb
import pytest
from chromadb.errors import ChromaError

from src.ingestion.models import Document, DocumentMetadata
from src.vectorstore.exceptions import StoreConnectionError
from src.vectorstore.store import ChromaVectorStore


def make_store() -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_dir=Path("unused-in-test"),
        collection_name=f"store_test_{uuid4().hex}",
        client=chromadb.EphemeralClient(),
    )


def make_document(
    source: str,
    chunk_index: int | None,
    content: str,
) -> Document:
    return Document(
        content=content,
        metadata=DocumentMetadata(
            source=source,
            page=1,
            total_pages=1,
            chunk_index=chunk_index,
        ),
    )


def test_add_search_delete_and_idempotency() -> None:
    store = make_store()
    document = make_document(
        source="rapor.pdf", chunk_index=0, content="Vector databases perform similarity search."
    )

    embedding = [1.0, 0.0]

    ids = store.add_documents([document], [embedding])
    store.add_documents([document], [embedding])  # Same ID again

    assert ids == ["rapor.pdf::1::0"]
    assert store.count() == 1

    results = store.search([1.0, 0.0], k=1)

    assert results[0].id == ids[0]
    assert results[0].score == 1.0

    assert store.delete(ids) == 1
    assert store.count() == 0


def test_search_returns_top_k_in_similarity_order() -> None:
    store = make_store()
    documents = [
        make_document("rapor.pdf", 0, "Aynı yöndeki vektör"),
        make_document("rapor.pdf", 1, "Dik yöndeki vektör"),
        make_document("rapor.pdf", 2, "Zıt yöndeki vektör"),
    ]
    embeddings = [
        [1.0, 0.0],  # Query ile aynı
        [0.0, 1.0],  # Query'ye dik
        [-1.0, 0.0],  # Query'nin zıttı
    ]
    store.add_documents(documents, embeddings)

    results = store.search(query_embedding=[1.0, 0.0], k=2)

    assert len(results) == 2
    assert results[0].id == "rapor.pdf::1::0"
    assert results[0].score > results[1].score
    assert all(0.0 <= result.score <= 1.0 for result in results)


def test_search_filters_by_source_metadata() -> None:
    store = make_store()
    documents = [
        make_document("rapor.pdf", 0, "Rapor içeriği"),
        make_document("notlar.pdf", 0, "Notlar içeriği"),
    ]
    store.add_documents(
        documents,
        embeddings=[[1.0, 0.0], [1.0, 0.0]],
    )

    results = store.search(
        query_embedding=[1.0, 0.0],
        k=5,
        filters={"source": "rapor.pdf"},
    )

    assert len(results) == 1
    assert results[0].metadata["source"] == "rapor.pdf"


def test_search_empty_store_returns_empty_list() -> None:
    store = make_store()

    results = store.search(query_embedding=[1.0, 0.0], k=3)

    assert results == []


def test_add_documents_requires_matching_document_and_embedding_counts() -> None:
    store = make_store()
    document = make_document("rapor.pdf", 0, "Test content")

    with pytest.raises(
        ValueError,
        match="Document and embedding counts must be equal",
    ):
        store.add_documents([document], [])


def test_add_documents_accepts_empty_batch() -> None:
    store = make_store()

    assert store.add_documents([], []) == []
    assert store.count() == 0


def test_add_documents_requires_chunk_index() -> None:
    store = make_store()
    document = make_document("rapor.pdf", None, "Test content")

    with pytest.raises(
        ValueError,
        match="chunk_index must be assigned",
    ):
        store.add_documents([document], [[1.0, 0.0]])


@pytest.mark.parametrize("invalid_k", [0, -1])
def test_search_requires_positive_k(invalid_k: int) -> None:
    store = make_store()

    with pytest.raises(ValueError, match="k must be positive"):
        store.search([1.0, 0.0], k=invalid_k)


def test_search_maps_chroma_errors() -> None:
    collection = Mock()
    collection.count.return_value = 1
    collection.query.side_effect = ChromaError("connection lost")

    client = Mock()
    client.get_or_create_collection.return_value = collection

    store = ChromaVectorStore(
        persist_dir=Path("unused-in-test"),
        collection_name="failing_store",
        client=client,
    )

    with pytest.raises(
        StoreConnectionError,
        match="Vector search failed",
    ) as captured:
        store.search([1.0, 0.0], k=1)

    assert captured.value.details == {"cause": "connection lost"}


def test_delete_empty_id_list_is_noop() -> None:
    store = make_store()

    assert store.delete([]) == 0
    assert store.count() == 0


def test_get_ids_requires_metadata_filter() -> None:
    store = make_store()

    with pytest.raises(
        ValueError,
        match="At least one metadata filter is required",
    ):
        store.get_ids({})


def test_get_ids_returns_only_matching_document_ids() -> None:
    store = make_store()
    documents = [
        make_document("rapor.pdf", 0, "Rapor içeriği"),
        make_document("notlar.pdf", 0, "Notlar içeriği"),
    ]
    store.add_documents(
        documents,
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )

    ids = store.get_ids({"source": "rapor.pdf"})

    assert ids == ["rapor.pdf::1::0"]
