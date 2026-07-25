import pytest

from src.vectorstore.exceptions import (
    BaseVectorStoreError,
    DocumentNotFoundError,
    DuplicateDocumentError,
    EmbeddingError,
    StoreConnectionError,
)


@pytest.mark.parametrize(
    "exception_type",
    [
        EmbeddingError,
        StoreConnectionError,
        DocumentNotFoundError,
        DuplicateDocumentError,
    ],
)
def test_vectorstore_exceptions_share_base_and_context(exception_type) -> None:
    exception = exception_type(
        "vectorstore process failed",
        source_file="rapor.pdf",
        details={"operation": "test"},
    )

    assert isinstance(exception, BaseVectorStoreError)
    assert str(exception) == "vectorstore process failed"
    assert exception.source_file == "rapor.pdf"
    assert exception.details == {"operation": "test"}