import pytest

from src.ingestion.exceptions import (
    BaseIngestionError,
    PdfReadError,
    PageExtractionError,
    PreprocessingError,
    ChunkingError,
)


@pytest.mark.parametrize(
    "exc_cls",
    [PdfReadError, PageExtractionError, PreprocessingError, ChunkingError],
)
def test_all_exceptions_inherit_from_base(exc_cls):
    assert issubclass(exc_cls, BaseIngestionError)


def test_message_retrievable_via_str():
    exc = PdfReadError("dosya bozuk")
    assert str(exc) == "dosya bozuk"


def test_no_separate_message_attribute():
    exc = PdfReadError("dosya bozuk")
    assert not hasattr(exc, "message")


def test_optional_context_fields_default_to_none():
    exc = ChunkingError("bölme hatası")
    assert exc.source_file is None
    assert exc.details is None


def test_optional_context_fields_are_keyword_only():
    exc = PageExtractionError(
        "sayfa 3 okunamadı",
        source_file="rapor.pdf",
        details={"page": 3},
    )
    assert exc.source_file == "rapor.pdf"
    assert exc.details == {"page": 3}


def test_loader_reexports_still_work():
    # loader.py artık exceptions.py'den import ediyor olmalı,
    # kendi sınıflarını tanımlamıyor olmalı
    from src.ingestion.loader import PdfReadError as LoaderPdfReadError
    from src.ingestion.exceptions import PdfReadError as CentralPdfReadError

    assert LoaderPdfReadError is CentralPdfReadError
