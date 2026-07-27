from collections.abc import Iterator
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import (
    get_document_indexer,
    get_ingestion_pipeline,
    get_vector_store,
)
from src.api.routes.documents import router
from src.config.settings import Settings, get_settings
from src.ingestion.models import Document, DocumentMetadata, IngestionResult
from src.vectorstore.indexer import IndexingResult

_TEST_API_KEY = "test-only-api-key-123456789"


@pytest.fixture
def document_api() -> Iterator[tuple[TestClient, Mock, Mock, Mock]]:
    document = Document(
        content="Test chunk.",
        metadata=DocumentMetadata(
            source="rapor.pdf",
            page=1,
            total_pages=1,
            chunk_index=0,
        ),
    )

    pipeline = Mock()
    pipeline.run.return_value = IngestionResult(
        documents=[document],
        total_files=1,
        successful_files=1,
        failed_files=[],
        total_chunks=1,
        processing_time_seconds=0.1,
    )

    indexer = Mock()
    indexer.index.return_value = IndexingResult(
        total_documents=1,
        indexed_documents=1,
        failed_documents=0,
        total_batches=1,
        duration_seconds=0.2,
    )
    indexer.delete_by_source.side_effect = (
        lambda source: 2 if source == "rapor.pdf" else 0
    )

    store = Mock()
    store.count.return_value = 7

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        api_key=_TEST_API_KEY,
        api_max_upload_size_mb=1,
    )
    app.dependency_overrides[get_ingestion_pipeline] = lambda: pipeline
    app.dependency_overrides[get_document_indexer] = lambda: indexer
    app.dependency_overrides[get_vector_store] = lambda: store

    with TestClient(
        app,
        headers={"X-API-Key": _TEST_API_KEY},
    ) as client:
        yield client, pipeline, indexer, store


def test_upload_pdf_returns_indexing_summary(
    document_api: tuple[TestClient, Mock, Mock, Mock],
) -> None:
    client, pipeline, indexer, _ = document_api

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("rapor.pdf", b"%PDF-1.4\nfake", "application/pdf")},
    )

    assert response.status_code == 201
    assert response.json()["total_chunks"] == 1
    assert response.json()["indexed_chunks"] == 1
    assert response.json()["failed_chunks"] == 0
    pipeline.run.assert_called_once()
    indexer.index.assert_called_once()


@pytest.mark.parametrize(
    ("filename", "content", "expected_status"),
    [
        ("notlar.txt", b"%PDF-1.4\nfake", 422),
        ("sahte.pdf", b"not a pdf", 422),
        ("buyuk.pdf", b"%PDF-" + b"x" * (1024 * 1024), 413),
    ],
    ids=[
        "wrong_extension",
        "invalid-pdf-signature",
        "oversized-pdf",
    ],
)
def test_upload_rejects_invalid_files(
    document_api: tuple[TestClient, Mock, Mock, Mock],
    filename: str,
    content: bytes,
    expected_status: int,
) -> None:
    client, _, _, _ = document_api

    response = client.post(
        "/api/v1/documents/upload",
        files={"file": (filename, content, "application/octet-stream")},
    )

    assert response.status_code == expected_status


def test_delete_and_stats_endpoints(
    document_api: tuple[TestClient, Mock, Mock, Mock],
) -> None:
    client, _, _, _ = document_api

    delete_response = client.request(
        "DELETE",
        "/api/v1/documents",
        json={"source": "rapor.pdf"},
    )
    stats_response = client.get("/api/v1/documents/stats")

    assert delete_response.status_code == 200
    assert delete_response.json()["chunks_deleted"] == 2
    assert stats_response.status_code == 200
    assert stats_response.json() == {"total_chunks": 7}


def test_delete_missing_source_returns_404(
    document_api: tuple[TestClient, Mock, Mock, Mock],
) -> None:
    client, _, _, _ = document_api

    response = client.request(
        "DELETE",
        "/api/v1/documents",
        json={"source": "olmayan.pdf"},
    )

    assert response.status_code == 404

def test_documents_reject_invalid_api_key(
    document_api: tuple[TestClient, Mock, Mock, Mock],
) -> None:
    client, _, _, _ = document_api

    response = client.get(
        "/api/v1/documents/stats",
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 401