from pathlib import Path
from uuid import uuid4

import chromadb

from src.ingestion.models import Document, DocumentMetadata
from src.vectorstore.store import ChromaVectorStore


def make_store() -> ChromaVectorStore:
    return ChromaVectorStore(
        persist_dir=Path("unused-in-test"),
        collection_name=f"store_test_{uuid4().hex}",
        client=chromadb.EphemeralClient(),
    )


def make_document(
    source: str,
    chunk_index: int,
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
