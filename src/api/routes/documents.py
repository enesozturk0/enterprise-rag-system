"""Document upload, deletion and statistics endpoints."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from src.api.dependencies import (
    get_document_indexer,
    get_ingestion_pipeline,
    get_vector_store,
)
from src.api.middleware.auth import verify_api_key
from src.api.schemas import (
    DeleteRequest,
    DeleteResponse,
    DocumentStats,
    UploadResponse,
)
from src.config.settings import Settings, get_settings
from src.ingestion.pipeline import IngestionPipeline
from src.vectorstore.indexer import DocumentIndexer
from src.vectorstore.store import VectorStore

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"],
    dependencies=[Depends(verify_api_key)],
)

_COPY_CHUNK_SIZE = 1024 * 1024
_PDF_SIGNATURE = b"%PDF-"
_INVALID_FILENAME_CHARS = frozenset('<>:"|?*')


def _validate_upload_filename(filename: str | None) -> str:
    """Return a safe PDF filename or reject the upload."""

    if not filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="A PDF filename is required.",
        )

    normalized = filename.replace("\\", "/")
    safe_name = Path(normalized).name

    contains_invalid_character = any(
        character in _INVALID_FILENAME_CHARS or ord(character) < 32
        for character in safe_name
    )

    if (
        safe_name != normalized
        or not safe_name
        or len(safe_name) > 255
        or contains_invalid_character
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The uploaded filename is invalid.",
        )

    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Only PDF files are accepted.",
        )

    return safe_name


def _save_upload(
    file: UploadFile,
    destination: Path,
    max_size_bytes: int,
) -> None:
    """Copy an upload to disk while enforcing size and PDF-signature checks."""

    total_size = 0
    signature = bytearray()

    with destination.open("wb") as output:
        while chunk := file.file.read(_COPY_CHUNK_SIZE):
            total_size += len(chunk)

            if total_size > max_size_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="The uploaded PDF exceeds the configured size limit.",
                )

            if len(signature) < len(_PDF_SIGNATURE):
                required = len(_PDF_SIGNATURE) - len(signature)
                signature.extend(chunk[:required])

            output.write(chunk)

    if bytes(signature) != _PDF_SIGNATURE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The uploaded file does not contain a valid PDF signature.",
        )


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    file: Annotated[UploadFile, File(description="PDF document to index.")],
    settings: Annotated[Settings, Depends(get_settings)],
    pipeline: Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)],
    indexer: Annotated[DocumentIndexer, Depends(get_document_indexer)],
) -> UploadResponse:
    """Upload, process and index one PDF document."""

    safe_name = _validate_upload_filename(file.filename)
    max_size_bytes = settings.api_max_upload_size_mb * 1024 * 1024

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory) / safe_name
        _save_upload(file, temporary_path, max_size_bytes)
        ingestion_result = pipeline.run(temporary_path)

    if not ingestion_result.documents:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="No usable text could be extracted from the PDF.",
        )

    indexing_result = indexer.index(ingestion_result.documents)

    if indexing_result.indexed_documents == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The document could not be indexed.",
        )

    has_failed_chunks = indexing_result.failed_documents > 0
    message = (
        "Document indexed with some failed chunks."
        if has_failed_chunks
        else "Document uploaded and indexed successfully."
    )

    logger.info(
        "document_uploaded_and_indexed",
        file_name=safe_name,
        total_chunks=indexing_result.total_documents,
        indexed_chunks=indexing_result.indexed_documents,
        failed_chunks=indexing_result.failed_documents,
    )

    return UploadResponse(
        file_name=safe_name,
        total_chunks=indexing_result.total_documents,
        indexed_chunks=indexing_result.indexed_documents,
        failed_chunks=indexing_result.failed_documents,
        successful_files=ingestion_result.successful_files,
        failed_files=len(ingestion_result.failed_files),
        processing_time_seconds=round(
            ingestion_result.processing_time_seconds
            + indexing_result.duration_seconds,
            3,
        ),
        message=message,
    )


@router.delete(
    "",
    response_model=DeleteResponse,
    status_code=status.HTTP_200_OK,
)
def delete_document(
    body: DeleteRequest,
    indexer: Annotated[DocumentIndexer, Depends(get_document_indexer)],
) -> DeleteResponse:
    """Delete all indexed chunks belonging to one source."""

    deleted_count = indexer.delete_by_source(body.source)

    if deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No chunks found for source '{body.source}'.",
        )

    return DeleteResponse(
        source=body.source,
        chunks_deleted=deleted_count,
        message=f"Deleted {deleted_count} chunks from '{body.source}'.",
    )


@router.get(
    "/stats",
    response_model=DocumentStats,
    status_code=status.HTTP_200_OK,
)
def document_stats(
    store: Annotated[VectorStore, Depends(get_vector_store)],
) -> DocumentStats:
    """Return the current number of indexed chunks."""

    return DocumentStats(total_chunks=store.count())