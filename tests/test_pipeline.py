"""Tests for IngestionPipeline — the orchestrator module.

Strategy:
- Unit tests use real tiny PDFs created on the fly with ``fpdf2``.
- The pipeline is constructed with real dependencies (loader,
  preprocessor, chunker) so we're testing actual integration,
  not mocking everything away.
- A corrupt file is simply a non-PDF with a ``.pdf`` extension.
"""

from __future__ import annotations  # noqa: I001

from pathlib import Path

import pytest

from src.config.settings import Settings
from src.ingestion.models import IngestionResult
from src.ingestion.pipeline import IngestionPipeline


# ── helpers ──────────────────────────────────────────────────────


def _create_pdf(path: Path, pages: list[str]) -> Path:
    """Create a minimal valid PDF at *path* with given page texts.

    Uses fpdf2 which is a lightweight PDF generation library.
    Each string in *pages* becomes one page.
    """
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    for text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(w=0, h=10, text=text)
    pdf.output(str(path))
    return path


def _create_corrupt_file(path: Path) -> Path:
    """Write random bytes with a .pdf extension — guaranteed unparseable."""
    path.write_bytes(b"THIS IS NOT A PDF AT ALL \x00\xff\xfe")
    return path


@pytest.fixture()
def test_settings(tmp_path: Path) -> Settings:
    """Settings override: point chroma to tmp so nothing persists."""
    return Settings(chroma_persist_dir=tmp_path / "chroma")


@pytest.fixture()
def pipeline(test_settings: Settings) -> IngestionPipeline:
    return IngestionPipeline(test_settings)


# ── single file tests ────────────────────────────────────────────


class TestSingleFile:
    """Pipeline.run() with a single PDF path."""

    def test_valid_pdf_produces_chunks(self, pipeline: IngestionPipeline, tmp_path: Path) -> None:
        pdf = _create_pdf(tmp_path / "report.pdf", ["Page one content.", "Page two content."])

        result = pipeline.run(pdf)

        assert isinstance(result, IngestionResult)
        assert result.successful_files == 1
        assert result.total_files == 1
        assert result.total_chunks > 0
        assert len(result.documents) == result.total_chunks
        assert result.failed_files == []
        assert result.processing_time_seconds > 0

    def test_corrupt_file_records_failure(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        bad = _create_corrupt_file(tmp_path / "broken.pdf")

        result = pipeline.run(bad)

        assert result.successful_files == 0
        assert result.total_files == 1
        assert len(result.failed_files) == 1
        assert result.failed_files[0].file_name == "broken.pdf"
        assert result.failed_files[0].error_type == "PdfReadError"
        assert result.total_chunks == 0
        assert result.documents == []

    def test_nonexistent_path_raises(self, pipeline: IngestionPipeline) -> None:
        with pytest.raises(FileNotFoundError):
            pipeline.run(Path("/tmp/does_not_exist_at_all.pdf"))


# ── directory tests ──────────────────────────────────────────────


class TestDirectory:
    """Pipeline.run() with a directory of PDFs."""

    def test_mixed_directory_partial_success(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        """1 valid + 1 corrupt → valid is processed, corrupt is in failed_files."""
        _create_pdf(tmp_path / "good.pdf", ["Good content here."])
        _create_corrupt_file(tmp_path / "bad.pdf")

        result = pipeline.run(tmp_path)

        assert result.total_files == 2
        assert result.successful_files == 1
        assert len(result.failed_files) == 1
        assert result.failed_files[0].file_name == "bad.pdf"
        assert result.total_chunks > 0

    def test_empty_directory_no_error(self, pipeline: IngestionPipeline, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = pipeline.run(empty_dir)

        assert result.total_files == 0
        assert result.successful_files == 0
        assert result.total_chunks == 0
        assert result.documents == []
        assert result.failed_files == []

    def test_all_valid_directory(self, pipeline: IngestionPipeline, tmp_path: Path) -> None:
        _create_pdf(tmp_path / "a.pdf", ["Alpha content."])
        _create_pdf(tmp_path / "b.pdf", ["Bravo content.", "Page two."])

        result = pipeline.run(tmp_path)

        assert result.total_files == 2
        assert result.successful_files == 2
        assert result.failed_files == []
        assert result.total_chunks > 0


# ── metadata & content integrity ─────────────────────────────────


class TestIntegrity:
    """Verify that metadata flows correctly through the pipeline."""

    def test_chunk_metadata_has_source_and_page(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        _create_pdf(tmp_path / "doc.pdf", ["Some text on page one."])

        result = pipeline.run(tmp_path / "doc.pdf")

        for doc in result.documents:
            assert doc.metadata.source == "doc.pdf"
            assert doc.metadata.page >= 1
            assert doc.metadata.chunk_index is not None

    def test_chunk_indices_are_sequential(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        _create_pdf(
            tmp_path / "multi.pdf",
            ["First page with enough text to generate content."] * 3,
        )

        result = pipeline.run(tmp_path / "multi.pdf")

        indices = [d.metadata.chunk_index for d in result.documents]
        assert indices == list(range(len(indices)))

    def test_result_counts_are_consistent(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        _create_pdf(tmp_path / "x.pdf", ["Hello world."])

        result = pipeline.run(tmp_path / "x.pdf")

        assert result.total_chunks == len(result.documents)
        assert result.total_files == result.successful_files + len(result.failed_files)
