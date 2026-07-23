"""Ingestion pipeline orchestrator.

Single entry-point that ties together the loader, preprocessor and
chunker modules.  The pipeline never crashes on a single bad file —
it logs the failure, records it in the result, and moves on.

Usage:
    pipeline = IngestionPipeline(settings)
    result = pipeline.run(Path("data/raw"))
    print(result.total_chunks, result.failed_files)
"""

from __future__ import annotations

import time
from pathlib import Path

import structlog

from src.config.settings import Settings
from src.ingestion.chunker import TextChunker
from src.ingestion.exceptions import BaseIngestionError
from src.ingestion.loader import load_pdf
from src.ingestion.models import Document, FailedFile, IngestionResult
from src.ingestion.preprocessor import TextPreprocessor

logger = structlog.get_logger(__name__)


class IngestionPipeline:
    """Orchestrates: PDF → load → preprocess → chunk → IngestionResult.

    Dependencies (loader, preprocessor, chunker) are injected through
    the constructor.  In production, you pass real instances built from
    ``Settings``; in tests you pass mocks/stubs.
    """

    # ── constructor ──────────────────────────────────────────────

    def __init__(
        self,
        settings: Settings,
        *,
        preprocessor: TextPreprocessor | None = None,
        chunker: TextChunker | None = None,
    ) -> None:
        self._settings = settings

        # Dependency injection:  accept overrides for testing,
        # fall back to real implementations when nothing is passed.
        self._preprocessor = preprocessor or TextPreprocessor()
        self._chunker = chunker or TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    # ── public API ───────────────────────────────────────────────

    def run(self, source: Path) -> IngestionResult:
        """Process *source* (single PDF **or** directory of PDFs).

        Returns an ``IngestionResult`` that always contains:
        * every successfully chunked document,
        * metadata about how many files succeeded / failed,
        * wall-clock duration for the entire pipeline.
        """
        start = time.perf_counter()

        if source.is_file():
            raw_docs, failed, file_count, ok_count = self._process_single_file(source)
        elif source.is_dir():
            raw_docs, failed, file_count, ok_count = self._process_directory(source)
        else:
            raise FileNotFoundError(f"Source path does not exist: {source}")

        # ── preprocess ───────────────────────────────────────────
        cleaned_docs = self._preprocess_documents(raw_docs)

        # ── chunk ────────────────────────────────────────────────
        chunked_docs = self._chunker.chunk(cleaned_docs)

        elapsed = time.perf_counter() - start

        result = IngestionResult(
            documents=chunked_docs,
            total_files=file_count,
            successful_files=ok_count,
            failed_files=failed,
            total_chunks=len(chunked_docs),
            processing_time_seconds=round(elapsed, 3),
        )

        logger.info(
            "ingestion_complete",
            total_files=result.total_files,
            successful=result.successful_files,
            failed=len(result.failed_files),
            total_chunks=result.total_chunks,
            duration_s=result.processing_time_seconds,
        )
        return result

    # ── private helpers ──────────────────────────────────────────

    def _process_single_file(
        self, file_path: Path
    ) -> tuple[list[Document], list[FailedFile], int, int]:
        """Load one PDF.  Returns (docs, failures, total_count, ok_count)."""
        try:
            docs = load_pdf(file_path)
            return docs, [], 1, 1
        except BaseIngestionError as exc:
            logger.error(
                "file_ingestion_failed",
                file=file_path.name,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            failed = FailedFile(
                file_name=file_path.name,
                error_type=type(exc).__name__,
                error_detail=str(exc),
            )
            return [], [failed], 1, 0

    def _process_directory(
        self, dir_path: Path
    ) -> tuple[list[Document], list[FailedFile], int, int]:
        """Load every PDF in *dir_path*.  Bad files are logged & skipped."""
        pdf_files = sorted(dir_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning("no_pdfs_found", directory=str(dir_path))
            return [], [], 0, 0

        all_docs: list[Document] = []
        failed_files: list[FailedFile] = []

        for pdf_file in pdf_files:
            try:
                docs = load_pdf(pdf_file)
                all_docs.extend(docs)
            except BaseIngestionError as exc:
                logger.error(
                    "file_ingestion_failed",
                    file=pdf_file.name,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                failed_files.append(
                    FailedFile(
                        file_name=pdf_file.name,
                        error_type=type(exc).__name__,
                        error_detail=str(exc),
                    )
                )

        ok_count = len(pdf_files) - len(failed_files)
        return all_docs, failed_files, len(pdf_files), ok_count

    def _preprocess_documents(self, documents: list[Document]) -> list[Document]:
        """Run header/footer detection, then clean each page.

        Header/footer detection is a *corpus-level* operation: it needs
        all pages of a given source PDF to figure out which lines
        repeat.  So we group pages by source, detect once per source,
        then create a preprocessor that knows those repeated lines.
        """
        pages_by_source: dict[str, list[Document]] = {}
        for doc in documents:
            pages_by_source.setdefault(doc.metadata.source, []).append(doc)

        cleaned: list[Document] = []

        for source, pages in pages_by_source.items():
            # Detect repeated headers/footers across all pages of this PDF
            page_texts = [p.content for p in pages]
            repeated = TextPreprocessor.detect_repeated_lines(page_texts)

            # Build a source-specific preprocessor that knows these headers
            source_preprocessor = TextPreprocessor(known_headers_footers=repeated)

            for page_doc in pages:
                clean_text = source_preprocessor.clean(page_doc.content)

                # Edge case: cleaning may leave a page completely empty
                if not clean_text.strip():
                    logger.debug(
                        "page_empty_after_cleaning",
                        source=source,
                        page=page_doc.metadata.page,
                    )
                    continue

                cleaned.append(
                    Document(
                        content=clean_text,
                        metadata=page_doc.metadata,
                    )
                )

        return cleaned
