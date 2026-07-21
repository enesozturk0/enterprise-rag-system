import pytest
from pydantic import ValidationError

from src.ingestion.models import DocumentMetadata, Document


def test_valid_metadata():
    metadata = DocumentMetadata(
        source="report.pdf",
        page=2,
        total_pages=10,
        chunk_index=0,
    )

    assert metadata.page == 2
    assert metadata.source == "report.pdf"


def test_page_must_be_positive():
    with pytest.raises(ValidationError):
        DocumentMetadata(
            source="report.pdf",
            page=0,
            total_pages=10,
        )


def test_page_cannot_exceed_total_pages():
    with pytest.raises(ValidationError):
        DocumentMetadata(
            source="report.pdf",
            page=15,
            total_pages=10,
        )


def test_source_must_not_be_path():
    with pytest.raises(ValidationError):
        DocumentMetadata(
            source="folder/report.pdf",
            page=1,
            total_pages=10,
        )


def test_chunk_index_cannot_be_negative():
    with pytest.raises(ValidationError):
        DocumentMetadata(
            source="report.pdf",
            page=1,
            total_pages=10,
            chunk_index=-1,
        )


def test_document_content_cannot_be_empty():
    with pytest.raises(ValidationError):
        Document(
            content="",
            metadata=DocumentMetadata(
                source="report.pdf",
                page=1,
                total_pages=10,
            ),
        )