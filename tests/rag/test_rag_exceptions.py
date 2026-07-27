import pytest

from src.rag.exceptions import (
    BaseRAGError,
    ChainError,
    GenerationError,
    PromptBuildError,
    RetrievalError,
)


@pytest.mark.parametrize(
    "exception_type",
    [
        RetrievalError,
        PromptBuildError,
        GenerationError,
        ChainError,
    ],
)
def test_rag_exceptions_share_base_and_context(exception_type) -> None:
    exception = exception_type(
        "RAG işlemi başarısız",
        source_file="rapor.pdf",
        details={"operation": "test"},
    )

    assert isinstance(exception, BaseRAGError)
    assert str(exception) == "RAG işlemi başarısız"
    assert exception.source_file == "rapor.pdf"
    assert exception.details == {"operation": "test"}


def test_optional_context_defaults_to_none() -> None:
    exception = RetrievalError("Retrieval başarısız")

    assert exception.source_file is None
    assert exception.details is None
