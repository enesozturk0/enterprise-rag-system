"""Map domain exceptions to safe HTTP responses."""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from src.ingestion.exceptions import BaseIngestionError
from src.rag.exceptions import BaseRAGError, GenerationError, RetrievalError
from src.vectorstore.exceptions import BaseVectorStoreError

logger = structlog.get_logger(__name__)


def _error_response(
    *,
    status_code: int,
    detail: str,
    error_type: str,
) -> JSONResponse:
    """Build the common public error response."""

    return JSONResponse(
        status_code=status_code,
        content={
            "detail": detail,
            "error_type": error_type,
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register domain and fallback exception handlers."""

    @app.exception_handler(RetrievalError)
    async def retrieval_error_handler(
        request: Request,
        exc: RetrievalError,
    ) -> JSONResponse:
        logger.warning(
            "retrieval_error",
            path=request.url.path,
            error=str(exc),
            details=exc.details,
        )
        return _error_response(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Document retrieval service is unavailable.",
            error_type="RetrievalError",
        )

    @app.exception_handler(GenerationError)
    async def generation_error_handler(
        request: Request,
        exc: GenerationError,
    ) -> JSONResponse:
        logger.warning(
            "generation_error",
            path=request.url.path,
            error=str(exc),
            details=exc.details,
        )
        return _error_response(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Answer generation service is unavailable.",
            error_type="GenerationError",
        )

    @app.exception_handler(BaseRAGError)
    async def rag_error_handler(
        request: Request,
        exc: BaseRAGError,
    ) -> JSONResponse:
        logger.error(
            "rag_error",
            path=request.url.path,
            error_type=type(exc).__name__,
            error=str(exc),
            details=exc.details,
        )
        return _error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The RAG request could not be completed.",
            error_type=type(exc).__name__,
        )

    @app.exception_handler(BaseIngestionError)
    async def ingestion_error_handler(
        request: Request,
        exc: BaseIngestionError,
    ) -> JSONResponse:
        logger.warning(
            "ingestion_error",
            path=request.url.path,
            error_type=type(exc).__name__,
            error=str(exc),
            details=exc.details,
        )
        return _error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The document could not be processed.",
            error_type=type(exc).__name__,
        )

    @app.exception_handler(BaseVectorStoreError)
    async def vector_store_error_handler(
        request: Request,
        exc: BaseVectorStoreError,
    ) -> JSONResponse:
        logger.error(
            "vector_store_error",
            path=request.url.path,
            error_type=type(exc).__name__,
            error=str(exc),
            details=exc.details,
        )
        return _error_response(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The vector store is unavailable.",
            error_type=type(exc).__name__,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.exception(
            "unexpected_api_error",
            path=request.url.path,
            error_type=type(exc).__name__,
        )
        return _error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error.",
            error_type="InternalServerError",
        )