from pathlib import Path

import chromadb

from src.ingestion.models import Document, DocumentMetadata
from src.vectorstore.store import ChromaVectorStore


def test_add_search_delete_and_idempotency() -> None:
    store = ChromaVectorStore(
        persist_dir=Path("unused-in-test"),
        collection_name="test_documents",
        client=chromadb.EphemeralClient(),
    )
    document = Document(
        content="Vector databases perform similarity search.",
        metadata=DocumentMetadata(
            source="rapor.pdf",
            page=1,
            total_pages=1,
            chunk_index=0,
        ),
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