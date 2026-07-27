"""Request and response schemas for the query endpoint."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    """Request the body of POST /api/v1/query."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",  # Rejects JSON fields not defined in the contract.
    )

    question: str = Field(
        min_length=1,
        max_length=2000,
        description="The user's question to answer from indexed documents.",
        examples=["Motor yağı nasıl kontrol edilir?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional number of document chunks to retrieve.",
    )
    source_filter: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        pattern=r"^[^/\\]+$",
        description="Optional source filename filter.",
        examples=["kullanim-kilavuzu.pdf"],
    )


class SourceDetail(BaseModel):
    """One document chunk used while generating the answer."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    page: int = Field(ge=1)
    score: float = Field(ge=0.0, le=1.0)
    content_preview: str = Field(max_length=200)


class QueryResponse(BaseModel):
    """Response body of POST /api/v1/query."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    sources: list[SourceDetail]
    model: str = Field(min_length=1)
    duration_seconds: float = Field(gt=0.0)
