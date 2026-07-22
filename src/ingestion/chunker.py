"""Chunk cleaned page text into embedding-sized pieces using LangChain's
recursive splitter, while keeping our own Document model at the boundary."""

from __future__ import annotations

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import get_settings
from src.ingestion.models import Document

logger = structlog.get_logger(__name__)


class TextChunker:
    """Splits cleaned Document instances into smaller, embedding-sized chunks."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        settings = get_settings()
        self._chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self._chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Group pages by source file, then chunk each source independently."""
        pages_by_source: dict[str, list[Document]] = {}
        for doc in documents:
            pages_by_source.setdefault(doc.metadata.source, []).append(doc)

        all_chunks: list[Document] = []
        for source, pages in pages_by_source.items():
            source_chunks = self._chunk_single_source(pages)
            all_chunks.extend(source_chunks)
            logger.info(
                "source chunked",
                source=source,
                input_pages=len(pages),
                output_chunks=len(source_chunks),
            )

        return all_chunks

    def _chunk_single_source(self, pages: list[Document]) -> list[Document]:
        """Chunk each page independently"""
        chunks: list[Document] = []
        global_chunk_index = 0

        for page_doc in pages:
            pieces = self._splitter.split_text(page_doc.content)
            for piece in pieces:
                new_metadata = page_doc.metadata.model_copy(
                    update={"chunk_index": global_chunk_index}
                )
                chunks.append(Document(content=piece, metadata=new_metadata))
                global_chunk_index += 1

        return chunks
