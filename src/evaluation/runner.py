"""Run deterministic RAG evaluation against a golden dataset."""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Protocol

import structlog
import yaml
from pydantic import ValidationError

from src.evaluation.exceptions import DatasetLoadError, ResultWriteError
from src.evaluation.metrics import (
    hit_rate,
    mean_reciprocal_rank,
    source_page_accuracy,
)
from src.evaluation.models import (
    EvalQuestion,
    EvalResult,
    EvalSummary,
    EvaluationDataset,
    RetrievedSource,
)
from src.rag.chain import RAGResponse
from src.rag.exceptions import BaseRAGError

logger = structlog.get_logger(__name__)


class RAGQueryChain(Protocol):
    """Minimal RAG chain interface required by evaluation."""

    def query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> RAGResponse:
        """Return a generated answer and its retrieved sources."""

        ...


class EvaluationRunner:
    """Load golden questions, run RAG queries and aggregate metrics."""

    def __init__(
        self,
        chain: RAGQueryChain,
        *,
        top_k: int = 5,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive.")

        self._chain = chain
        self._top_k = top_k

    @staticmethod
    def load_dataset(dataset_path: Path) -> EvaluationDataset:
        """Load and validate a YAML golden dataset."""

        try:
            raw_dataset = yaml.safe_load(
                dataset_path.read_text(encoding="utf-8")
            )
            return EvaluationDataset.model_validate(raw_dataset)
        except (OSError, yaml.YAMLError, ValidationError) as exc:
            raise DatasetLoadError(
                "Evaluation dataset could not be loaded.",
                details={
                    "path": str(dataset_path),
                    "cause_type": type(exc).__name__,
                },
            ) from exc

    def evaluate(
        self,
        questions: Sequence[EvalQuestion],
    ) -> EvalSummary:
        """Evaluate all questions without stopping on expected RAG errors."""

        results = [
            self._evaluate_question(question)
            for question in questions
        ]
        return self._build_summary(results)

    def _evaluate_question(
        self,
        question: EvalQuestion,
    ) -> EvalResult:
        started_at = time.perf_counter()

        try:
            response = self._chain.query(
                question.question,
                top_k=self._top_k,
            )
        except BaseRAGError as exc:
            duration = time.perf_counter() - started_at
            error_type = type(exc).__name__

            logger.warning(
                "evaluation_question_failed",
                question_id=question.id,
                error_type=error_type,
            )
            return EvalResult(
                question_id=question.id,
                question=question.question,
                ground_truth=question.ground_truth,
                error_type=error_type,
                duration_seconds=duration,
            )

        duration = time.perf_counter() - started_at
        retrieved_sources = [
            RetrievedSource(
                source=source.source,
                page=source.page,
                score=source.score,
                content_preview=source.content_preview,
            )
            for source in response.sources
        ]

        result = EvalResult(
            question_id=question.id,
            question=question.question,
            ground_truth=question.ground_truth,
            generated_answer=response.answer,
            model=response.model,
            retrieved_sources=retrieved_sources,
            duration_seconds=duration,
            hit_rate=hit_rate(
                retrieved_sources,
                question.expected_sources,
            ),
            reciprocal_rank=mean_reciprocal_rank(
                retrieved_sources,
                question.expected_sources,
            ),
            source_page_accuracy=source_page_accuracy(
                retrieved_sources,
                question.expected_sources,
            ),
        )

        logger.info(
            "evaluation_question_completed",
            question_id=question.id,
            succeeded=result.succeeded,
            sources_found=len(retrieved_sources),
            duration_seconds=round(duration, 3),
        )
        return result

    def _build_summary(
        self,
        results: list[EvalResult],
    ) -> EvalSummary:
        successful_questions = sum(
            result.succeeded for result in results
        )

        return EvalSummary(
            total_questions=len(results),
            successful_questions=successful_questions,
            failed_questions=len(results) - successful_questions,
            top_k=self._top_k,
            avg_hit_rate=self._average_available(
                result.hit_rate for result in results
            ),
            avg_reciprocal_rank=self._average_available(
                result.reciprocal_rank for result in results
            ),
            avg_source_page_accuracy=self._average_available(
                result.source_page_accuracy for result in results
            ),
            avg_duration_seconds=(
                sum(result.duration_seconds for result in results)
                / len(results)
                if results
                else 0.0
            ),
            avg_faithfulness=self._average_available(
                result.faithfulness for result in results
            ),
            avg_answer_relevancy=self._average_available(
                result.answer_relevancy for result in results
            ),
            results=results,
        )

    @staticmethod
    def _average_available(
        values: Iterable[float | None],
    ) -> float | None:
        available = [value for value in values if value is not None]

        if not available:
            return None

        return sum(available) / len(available)

    @staticmethod
    def save_results(
        summary: EvalSummary,
        output_path: Path,
    ) -> None:
        """Atomically save evaluation results as UTF-8 JSON."""

        temporary_path: Path | None = None

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(
                    summary.model_dump(mode="json"),
                    temporary_file,
                    ensure_ascii=False,
                    indent=2,
                )
                temporary_file.write("\n")

            temporary_path.replace(output_path)
        except OSError as exc:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "evaluation_temporary_file_cleanup_failed",
                        path=str(temporary_path),
                    )

            raise ResultWriteError(
                "Evaluation results could not be saved.",
                details={
                    "path": str(output_path),
                    "cause_type": type(exc).__name__,
                },
            ) from exc

        logger.info(
            "evaluation_results_saved",
            path=str(output_path),
            total_questions=summary.total_questions,
        )