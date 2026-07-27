"""Centralized exception hierarchy for the RAG engine."""

from __future__ import annotations

from typing import Any


class BaseRAGError(Exception):
    """Root of the RAG exception hierarchy."""

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


class RetrievalError(BaseRAGError):
    """Raised when document retrieval fails."""


class PromptBuildError(BaseRAGError):
    """Raised when prompt construction fails."""


class GenerationError(BaseRAGError):
    """Raised when the LLM cannot generate a valid response."""


class ChainError(BaseRAGError):
    """Raised when RAG orchestration fails."""
