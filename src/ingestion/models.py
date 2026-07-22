from __future__ import annotations

from typing import Self

from pydantic import BaseModel, Field, field_validator, model_validator

"""Shared data models for the ingestion pipeline."""


class DocumentMetadata(BaseModel):
    """Metadata attached to every Document flowing through the pipeline."""

    source: str = Field(..., description="PDF file name (not full path)")
    page: int = Field(..., ge=1, description="1-indexed page number, shown to users")
    total_pages: int = Field(..., ge=1, description="Total pages in the source PDF")
    chunk_index: int | None = Field(
        default=None, ge=0, description="Chunk order within the document; set after chunking"
    )

    @model_validator(mode="after")
    def page_not_exceed_total(self) -> Self:
        if self.page > self.total_pages:
            raise ValueError(f"page {self.page} cannot exceed total_pages {self.total_pages}")
        return self

    @field_validator("source")
    @classmethod
    def source_must_be_filename_only(cls, v: str) -> str:
        if "/" in v or "\\" in v:
            raise ValueError("source must be a filename, not path")
        return v


class Document(BaseModel):
    """A unit of text flowing through the ingestion pipeline."""

    content: str = Field(..., min_length=1, description="Raw or chunked text content")
    metadata: DocumentMetadata

    def to_langchain_document(self):
        """Adapter: convert to LangChain's Document type at the boundary"""
        from langchain_core.documents import Document as LCDocument

        return LCDocument(page_content=self.content, metadata=self.metadata.model_dump())
