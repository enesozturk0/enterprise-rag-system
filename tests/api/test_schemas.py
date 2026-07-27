import pytest
from pydantic import ValidationError

from src.api.schemas import (
    DeleteRequest,
    DocumentStats,
    QueryRequest,
    QueryResponse,
    SourceDetail,
    UploadResponse,
)
from src.api.schemas import __all__ as schema_exports


def test_query_request_normalizes_and_validates_input() -> None:
    request = QueryRequest(
        question="  Motor yağı nasıl kontrol edilir?  ",
        top_k=5,
    )

    assert request.question == "Motor yağı nasıl kontrol edilir?"
    assert request.top_k == 5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question", ""),
        ("question", "   "),
        ("question", "x" * 2001),
        ("top_k", 0),
        ("top_k", 21),
        ("source_filter", "../rapor.pdf"),
        ("source_filter", r"C:\docs\rapor.pdf"),
    ],
)
def test_query_request_rejects_invalid_values(field: str, value: object) -> None:
    payload: dict[str, object] = {"question": "Geçerli soru"}
    payload[field] = value

    with pytest.raises(ValidationError):
        QueryRequest.model_validate(payload)


def test_query_response_matches_public_contract() -> None:
    response = QueryResponse(
        answer="Test cevabı.",
        sources=[
            SourceDetail(
                source="test.pdf",
                page=1,
                score=0.95,
                content_preview="İlgili içerik.",
            )
        ],
        model="test-model",
        duration_seconds=0.1,
    )

    assert response.sources[0].source == "test.pdf"
    assert response.sources[0].score == 0.95


def test_upload_response_requires_balanced_chunk_counts() -> None:
    response = UploadResponse(
        file_name="rapor.pdf",
        total_chunks=5,
        indexed_chunks=4,
        failed_chunks=1,
        successful_files=1,
        failed_files=0,
        processing_time_seconds=0.5,
    )

    assert response.indexed_chunks == 4
    assert response.failed_chunks == 1

    with pytest.raises(ValidationError):
        UploadResponse(
            file_name="rapor.pdf",
            total_chunks=5,
            indexed_chunks=3,
            failed_chunks=1,
            successful_files=1,
            failed_files=0,
            processing_time_seconds=0.5,
        )


@pytest.mark.parametrize("source", ["", "   ", "../rapor.pdf", r"C:\docs\rapor.pdf"])
def test_delete_request_rejects_invalid_source(source: str) -> None:
    with pytest.raises(ValidationError):
        DeleteRequest(source=source)


def test_document_stats_rejects_negative_chunk_count() -> None:
    with pytest.raises(ValidationError):
        DocumentStats(total_chunks=-1)


def test_schema_package_exposes_expected_public_api() -> None:
    assert schema_exports == [
        "DeleteRequest",
        "DeleteResponse",
        "DocumentStats",
        "QueryRequest",
        "QueryResponse",
        "SourceDetail",
        "UploadResponse",
    ]
