"""Orchestrate the complete RAG pipeline."""

from __future__ import annotations

import time
from typing import Any

import structlog
from pydantic import BaseModel, Field

from src.rag.exceptions import ChainError
from src.rag.generator import LLMGenerator
from src.rag.prompt import RAGPromptBuilder
from src.rag.retriever import DocumentRetriever
from src.vectorstore.store import SearchResult

logger = structlog.get_logger(__name__)


class SourceInfo(BaseModel):
    """A document chunk that contributed to the generated answer."""

    source: str = Field(min_length=1)
    page: int = Field(ge=1)
    score: float = Field(ge=0.0, le=1.0)
    content_preview: str


class RAGResponse(BaseModel):
    """Complete response returned by the RAG pipeline."""

    answer: str = Field(min_length=1)
    sources: list[SourceInfo]
    retrieval_scores: list[float]
    model: str = Field(min_length=1)
    duration_seconds: float = Field(gt=0.0)


class RAGChain:
    """Wire retriever, prompt builder, and generator into one operation."""

    def __init__(
        self,
        retriever: DocumentRetriever,
        prompt_builder: RAGPromptBuilder,
        generator: LLMGenerator,
        model_name: str,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")

        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._generator = generator
        self._model_name = model_name

    def query(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RAGResponse:
        if not question.strip():
            raise ChainError("Question cannot be empty.")

        started_at = time.perf_counter()

        search_results = self._retriever.retrieve(
            query=question,
            top_k=top_k,
            filters=filters,
        )

        sources = self._build_sources(search_results)

        prompt = self._prompt_builder.build(
            query=question,
            context_results=search_results,
        )
        answer = self._generator.generate(
            system_prompt=prompt["system"],
            user_prompt=prompt["user"],
        )

        duration = time.perf_counter() - started_at

        logger.info(
            "rag_query_completed",
            question_length=len(question),
            sources_used=len(sources),
            answer_length=len(answer),
            duration_seconds=round(duration, 3),
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieval_scores=[result.score for result in search_results],
            model=self._model_name,
            duration_seconds=duration,
        )

    @staticmethod
    def _build_sources(results: list[SearchResult]) -> list[SourceInfo]:
        sources: list[SourceInfo] = []

        for result in results:
            source = result.metadata.get("source")
            page = result.metadata.get("page")

            if not isinstance(source, str) or not source.strip():
                raise ChainError(
                    "Search result has invalid source metadata.",
                    details={
                        "result_id": result.id,
                        "source": source,
                    },
                )

            if not isinstance(page, int) or isinstance(page, bool) or page < 1:
                raise ChainError(
                    "Search result has invalid page metadata.",
                    details={
                        "result_id": result.id,
                        "page": page,
                    },
                )

            sources.append(
                SourceInfo(
                    source=source.strip(),
                    page=page,
                    score=result.score,
                    content_preview=result.content[:200],
                )
            )

        return sources
