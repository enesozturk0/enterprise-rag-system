"""Request and response schemas for document endpoints."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UploadResponse(BaseModel):
    """Response after a successful PDF upload and indexing operation."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    file_name: str = Field(min_length=1, max_length=255)
    total_chunks: int = Field(ge=0)
    indexed_chunks: int = Field(ge=0)
    failed_chunks: int = Field(ge=0)
    successful_files: int = Field(ge=0)
    failed_files: int = Field(ge=0)
    processing_time_seconds: float = Field(ge=0.0)
    message: str = Field(
        default="Document uploaded and indexed successfully.",
        min_length=1,
    )

    @model_validator(mode="after")
    def chunk_counts_must_balance(self) -> Self:
        # chunk counts must balance
        if self.indexed_chunks + self.failed_chunks != self.total_chunks:
            raise ValueError("indexed_chunks + failed_chunks must equal total_chunks")
        return self


class DeleteRequest(BaseModel):
    """Request body for deleting a document from the index."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    source: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[^/\\]+$",  # only file name
        description="Filename to remove from the index.",
        examples=["rapor.pdf"],
    )


class DeleteResponse(BaseModel):
    """Response after deleting a document's chunks."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, max_length=255)
    chunks_deleted: int = Field(ge=1)
    message: str = Field(min_length=1)


class DocumentStats(BaseModel):
    """Current document index statistics."""

    model_config = ConfigDict(extra="forbid")

    total_chunks: int = Field(ge=0)
