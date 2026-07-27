"""RAG query endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from src.api.dependencies import get_rag_chain
from src.api.middleware.auth import verify_api_key
from src.api.schemas import QueryRequest, QueryResponse, SourceDetail
from src.rag.chain import RAGChain

router = APIRouter(
    prefix="/api/v1",
    tags=["Query"],
    dependencies=[Depends(verify_api_key)],
)


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
)
def query_documents(
    body: QueryRequest,
    chain: Annotated[RAGChain, Depends(get_rag_chain)],
) -> QueryResponse:
    """Answer a question using the indexed documents."""

    filters = (
        {"source": body.source_filter}
        if body.source_filter is not None
        else None
    )

    rag_response = chain.query(
        question=body.question,
        top_k=body.top_k,
        filters=filters,
    )

    return QueryResponse(
        answer=rag_response.answer,
        sources=[
            SourceDetail(
                source=source.source,
                page=source.page,
                score=source.score,
                content_preview=source.content_preview,
            )
            for source in rag_response.sources
        ],
        model=rag_response.model,
        duration_seconds=rag_response.duration_seconds,
    )