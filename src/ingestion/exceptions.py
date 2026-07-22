"""Centralized exception hierarchy for the ingestion pipeline.

All ingestion modules (loader, preprocessor, chunker, pipeline) raise
and catch from this hierarchy instead of generic `Exception`, so that
callers (pipeline.py, the API layer) can handle failures by type in
one place.
"""

from __future__ import annotations

from typing import Any


class BaseIngestionError(Exception):
    """Root of the ingestion exception hierarchy.

    No separate `message` attribute is defined -- the message is
    handed to `Exception.__init__` and retrieved the standard way
    (`str(exc)`). `source_file` and `details` are optional structured
    context for logging and API-layer error mapping.
    """

    def __init__(
        self,
        message: str = "",
        *,
        source_file: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.source_file = source_file
        self.details = details


class PdfReadError(BaseIngestionError):
    """Raised when a PDF file cannot be opened or read."""


class PageExtractionError(BaseIngestionError):
    """Raised when text extraction fails for a specific page."""


class PreprocessingError(BaseIngestionError):
    """Raised when a text-cleaning rule fails unexpectedly."""


class ChunkingError(BaseIngestionError):
    """Raised when splitting cleaned text into chunks fails unexpectedly."""
