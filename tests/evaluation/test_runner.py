import json
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from src.evaluation.exceptions import DatasetLoadError
from src.evaluation.models import (
    NOT_IN_CONTEXT,
    EvalQuestion,
    EvalSummary,
    ExpectedSource,
)
from src.evaluation.runner import EvaluationRunner
from src.rag.chain import RAGChain, RAGResponse, SourceInfo
from src.rag.exceptions import GenerationError


def _positive_question() -> EvalQuestion:
    return EvalQuestion(
        id="positive",
        question="Motor yağı nasıl kontrol edilir?",
        ground_truth="Yağ seviyesi düz zeminde kontrol edilir.",
        expected_sources=[ExpectedSource(source="manual.pdf", pages=[263])],
    )


def _negative_question() -> EvalQuestion:
    return EvalQuestion(
        id="negative",
        question="Belgede olmayan soru",
        ground_truth=NOT_IN_CONTEXT,
    )


def _response(
    *,
    source: str = "manual.pdf",
    page: int = 263,
    answer: str = "Test answer",
) -> RAGResponse:
    return RAGResponse(
        answer=answer,
        sources=[
            SourceInfo(
                source=source,
                page=page,
                score=0.9,
                content_preview="Test context",
            )
        ],
        retrieval_scores=[0.9],
        model="test-model",
        duration_seconds=0.1,
    )


def test_load_dataset_validates_yaml(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.yaml"
    dataset_path.write_text(
        """
questions:
  - id: q1
    question: Test question
    ground_truth: Test answer
    expected_sources:
      - source: manual.pdf
        pages: [1]
""".strip(),
        encoding="utf-8",
    )

    dataset = EvaluationRunner.load_dataset(dataset_path)

    assert len(dataset.questions) == 1
    assert dataset.questions[0].id == "q1"


@pytest.mark.parametrize(
    "dataset_name",
    ["missing.yaml", "invalid.yaml"],
)
def test_load_dataset_maps_file_and_validation_errors(
    tmp_path: Path,
    dataset_name: str,
) -> None:
    dataset_path = tmp_path / dataset_name

    if dataset_name == "invalid.yaml":
        dataset_path.write_text(
            "questions: [invalid",
            encoding="utf-8",
        )

    with pytest.raises(DatasetLoadError) as captured:
        EvaluationRunner.load_dataset(dataset_path)

    assert captured.value.details is not None
    assert captured.value.details["path"] == str(dataset_path)


def test_evaluate_aggregates_only_applicable_metrics() -> None:
    chain = Mock(spec=RAGChain)
    chain.query.side_effect = [
        _response(),
        _response(
            source="irrelevant.pdf",
            page=1,
            answer="Bu bilgi bağlamda bulunmuyor.",
        ),
    ]
    runner = EvaluationRunner(chain, top_k=3)

    summary = runner.evaluate([_positive_question(), _negative_question()])

    assert summary.total_questions == 2
    assert summary.successful_questions == 2
    assert summary.failed_questions == 0
    assert summary.top_k == 3
    assert summary.avg_hit_rate == 1.0
    assert summary.avg_reciprocal_rank == 1.0
    assert summary.avg_source_page_accuracy == 1.0
    assert summary.avg_faithfulness is None
    assert summary.results[1].hit_rate is None

    assert chain.query.call_args_list == [
        call("Motor yağı nasıl kontrol edilir?", top_k=3),
        call("Belgede olmayan soru", top_k=3),
    ]


def test_expected_rag_error_is_recorded_without_stopping() -> None:
    chain = Mock(spec=RAGChain)
    chain.query.side_effect = GenerationError("Sensitive detail.")
    runner = EvaluationRunner(chain)

    summary = runner.evaluate([_positive_question()])

    assert summary.successful_questions == 0
    assert summary.failed_questions == 1
    assert summary.avg_hit_rate is None
    assert summary.results[0].error_type == "GenerationError"
    assert summary.results[0].generated_answer is None


def test_empty_question_list_produces_valid_summary() -> None:
    runner = EvaluationRunner(Mock(spec=RAGChain), top_k=4)

    summary = runner.evaluate([])

    assert summary == EvalSummary(
        total_questions=0,
        successful_questions=0,
        failed_questions=0,
        top_k=4,
        avg_duration_seconds=0.0,
        results=[],
    )


def test_save_results_creates_valid_json_atomically(
    tmp_path: Path,
) -> None:
    runner = EvaluationRunner(Mock(spec=RAGChain), top_k=5)
    summary = runner.evaluate([])
    output_path = tmp_path / "nested" / "results.json"

    runner.save_results(summary, output_path)

    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved["total_questions"] == 0
    assert saved["top_k"] == 5
    assert saved["results"] == []
    assert not list(output_path.parent.glob("*.tmp"))


def test_runner_requires_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        EvaluationRunner(Mock(spec=RAGChain), top_k=0)
