"""Centralized exception hierarchy for the vectorstore module."""

from __future__ import annotations

from typing import Any


class BaseVectorStoreError(Exception):
    """Root of the vectorstore exception hierarchy."""

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


class EmbeddingError(BaseVectorStoreError):
    """Raised when an embedding fails."""


class StoreConnectionError(BaseVectorStoreError):
    """Raised when a vectorstore connection fails."""


class DocumentNotFoundError(BaseVectorStoreError):
    """Raised when a document ID does not exist in the vectorstore."""


class DuplicateDocumentError(BaseVectorStoreError):
    """Raised when a document ID already exists in the vectorstore."""
