"""PDF document loader using PyMuPDF(fitz) for direct control over extraction."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import fitz
import structlog

from src.ingestion.exceptions import BaseIngestionError, PageExtractionError, PdfReadError
from src.ingestion.models import Document, DocumentMetadata

logger = structlog.get_logger(__name__)


def load_pdf(file_path: Path, skip_empty_pages: bool = True) -> list[Document]:
    if not file_path.exists():
        raise PdfReadError(f"PDF not found at: {file_path}")

    try:
        doc = fitz.open(str(file_path))
    except Exception as exc:
        raise PdfReadError(f"Cannot open PDF: {file_path}") from exc

    total_pages = doc.page_count
    documents: list[Document] = []

    try:
        for page_index in range(total_pages):
            try:
                page = doc.load_page(page_index)
                text = page.get_text()
            except Exception as exc:
                raise PageExtractionError(
                    f"Page {page_index + 1} extraction failed in {file_path.name}"
                ) from exc

            page_number = page_index + 1
            if not text.strip():
                if skip_empty_pages:
                    logger.warning("Empty page skipped", file=file_path.name, page=page_number)
                    continue
                text = "[EMPTY PAGE]"

            metadata = DocumentMetadata(
                source=file_path.name,
                page=page_number,
                total_pages=total_pages,
            )
            documents.append(Document(content=text, metadata=metadata))
    finally:
        doc.close()

    logger.info(
        "pdf_loaded",
        file=file_path.name,
        total_pages=total_pages,
        pages_extracted=len(documents),
    )

    return documents


def load_pdfs_iter(
    directory: Path,
    glob_pattern: str = "*.pdf",
    skip_empty_pages: bool = True,
) -> Generator[Document]:

    pdf_files = sorted(directory.glob(glob_pattern))
    logger.info(
        "pdf_directory_scan",
        directory=str(directory),
        files_found=len(pdf_files),
    )
    for pdf_file in pdf_files:
        try:
            documents = load_pdf(pdf_file, skip_empty_pages=skip_empty_pages)
        except BaseIngestionError as exc:
            logger.error(
                "pdf_load_failed_skipping",
                file=pdf_file.name,
                error=str(exc),
            )
            continue
        yield from documents
