"""Dependency-injection factories for FastAPI routes."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends

from src.config.settings import Settings, get_settings
from src.ingestion.pipeline import IngestionPipeline
from src.rag.chain import RAGChain
from src.rag.generator import LLMGenerator
from src.rag.prompt import RAGPromptBuilder
from src.rag.retriever import DocumentRetriever
from src.vectorstore.embeddings import EmbeddingClient, OllamaEmbeddingClient
from src.vectorstore.indexer import DocumentIndexer
from src.vectorstore.store import ChromaVectorStore, VectorStore


@lru_cache(maxsize=1)
def _build_embedding_client(
    base_url: str,
    model_name: str,
    batch_size: int,
) -> OllamaEmbeddingClient:
    """Build or reuse the Ollama embedding client."""

    return OllamaEmbeddingClient(
        base_url=base_url,
        model_name=model_name,
        batch_size=batch_size,
    )


@lru_cache(maxsize=1)
def _build_vector_store(
    persist_dir: Path,
    collection_name: str,
) -> ChromaVectorStore:
    """Build or reuse the persistent Chroma vector store."""

    return ChromaVectorStore(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )


@lru_cache(maxsize=1)
def _build_llm_generator(
    base_url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> LLMGenerator:
    """Build or reuse the Ollama generation client."""

    return LLMGenerator(
        base_url=base_url,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def get_embedding_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmbeddingClient:
    """Provide the shared embedding client."""

    return _build_embedding_client(
        settings.ollama_base_url,
        settings.embedding_model_name,
        settings.embedding_batch_size,
    )


def get_vector_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> VectorStore:
    """Provide the shared vector store."""

    return _build_vector_store(
        settings.chroma_persist_dir,
        settings.chroma_collection_name,
    )


def get_llm_generator(
    settings: Annotated[Settings, Depends(get_settings)],
) -> LLMGenerator:
    """Provide the shared LLM generator."""

    return _build_llm_generator(
        settings.ollama_base_url,
        settings.llm_model_name,
        settings.llm_temperature,
        settings.llm_max_tokens,
    )


def get_rag_chain(
    settings: Annotated[Settings, Depends(get_settings)],
    embedding_client: Annotated[EmbeddingClient, Depends(get_embedding_client)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    generator: Annotated[LLMGenerator, Depends(get_llm_generator)],
) -> RAGChain:
    """Provide a fully wired RAG chain."""

    retriever = DocumentRetriever(
        embedding_client=embedding_client,
        vector_store=vector_store,
        default_top_k=settings.retrieval_top_k,
    )

    return RAGChain(
        retriever=retriever,
        prompt_builder=RAGPromptBuilder(),
        generator=generator,
        model_name=settings.llm_model_name,
    )


def get_ingestion_pipeline(
    settings: Annotated[Settings, Depends(get_settings)],
) -> IngestionPipeline:
    """Provide the document ingestion pipeline."""

    return IngestionPipeline(settings)


def get_document_indexer(
    settings: Annotated[Settings, Depends(get_settings)],
    embedding_client: Annotated[EmbeddingClient, Depends(get_embedding_client)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
) -> DocumentIndexer:
    """Provide the document indexer."""

    return DocumentIndexer(
        embedding_client=embedding_client,
        vector_store=vector_store,
        batch_size=settings.embedding_batch_size,
    )
