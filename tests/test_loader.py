from pathlib import Path

import pytest

from src.ingestion.loader import PdfReadError, load_pdf, load_pdfs_iter
from src.ingestion.models import Document


def test_load_pdf_raises_on_missing_file():
    with pytest.raises(PdfReadError):
        load_pdf(Path("data/raw/does_not_exist.pdf"))


def test_load_pdf_returns_documents_with_correct_page_numbers():
    # Update this path to match a sample PDF in your data/raw directory.
    docs = load_pdf(Path("data/raw/test_loader_sample.pdf"))

    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].metadata.page == 1  # Verify 1-indexed page numbering.


def test_load_pdfs_iter_skips_broken_files(tmp_path):
    # Create an invalid PDF file (actually just plain text).
    broken = tmp_path / "broken.pdf"
    broken.write_text("this is not a pdf")

    docs = list(load_pdfs_iter(tmp_path))

    # The invalid PDF should be skipped without raising an exception.
    assert docs == []
