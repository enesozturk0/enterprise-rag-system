from pathlib import Path

import fitz
import pytest

from src.ingestion.loader import PdfReadError, load_pdf, load_pdfs_iter
from src.ingestion.models import Document


def test_load_pdf_raises_on_missing_file():
    with pytest.raises(PdfReadError):
        load_pdf(Path("data/raw/does_not_exist.pdf"))


def test_load_pdf_returns_documents_with_correct_page_numbers(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "sample.pdf"

    # Gerçek fakat test süresince yaşayan tek sayfalık PDF oluşturur.
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Deterministic test document.")
        pdf.save(pdf_path)

    docs = load_pdf(pdf_path)

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].content.strip() == "Deterministic test document."
    assert docs[0].metadata.source == "sample.pdf"
    assert docs[0].metadata.page == 1
    assert docs[0].metadata.total_pages == 1


def test_load_pdfs_iter_skips_broken_files(tmp_path):
    # Create an invalid PDF file (actually just plain text).
    broken = tmp_path / "broken.pdf"
    broken.write_text("this is not a pdf")

    docs = list(load_pdfs_iter(tmp_path))

    # The invalid PDF should be skipped without raising an exception.
    assert docs == []
