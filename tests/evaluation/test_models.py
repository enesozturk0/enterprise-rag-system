import pytest
from pydantic import ValidationError

from src.evaluation.models import (
    NOT_IN_CONTEXT,
    EvalQuestion,
    EvalResult,
    EvalSummary,
    EvaluationDataset,
    ExpectedSource,
    RetrievedSource,
)


def _positive_question(question_id: str = "q1") -> EvalQuestion:
    return EvalQuestion(
        id=question_id,
        question="Motor yağı nasıl kontrol edilir?",
        ground_truth="Yağ seviyesi düz zeminde kontrol edilir.",
        expected_sources=[
            ExpectedSource(source="manual.pdf", pages=[262, 263])
        ],
    )


def _successful_result() -> EvalResult:
    return EvalResult(
        question_id="q1",
        question="Motor yağı nasıl kontrol edilir?",
        ground_truth="Yağ seviyesi düz zeminde kontrol edilir.",
        generated_answer="Yağ seviyesi düz zeminde kontrol edilir.",
        model="llama3.1:8b",
        retrieved_sources=[
            RetrievedSource(
                source="manual.pdf",
                page=263,
                score=0.84,
                content_preview="Motor yağının kontrol edilmesi...",
            )
        ],
        duration_seconds=1.2,
        hit_rate=1.0,
        reciprocal_rank=1.0,
        source_page_accuracy=0.5,
    )


def test_positive_and_negative_questions_are_validated() -> None:
    positive = _positive_question()
    negative = EvalQuestion(
        id="q-negative",
        question="Belgede bulunmayan bilgi nedir?",
        ground_truth=NOT_IN_CONTEXT,
    )

    assert positive.expects_no_answer is False
    assert negative.expects_no_answer is True

    with pytest.raises(ValidationError):
        EvalQuestion(
            id="invalid-positive",
            question="Kaynağı olmayan pozitif soru",
            ground_truth="Beklenen cevap",
        )

    with pytest.raises(ValidationError):
        EvalQuestion(
            id="invalid-negative",
            question="Negatif soru",
            ground_truth=NOT_IN_CONTEXT,
            expected_sources=[
                ExpectedSource(source="manual.pdf", pages=[1])
            ],
        )


def test_expected_source_rejects_invalid_pages() -> None:
    with pytest.raises(ValidationError):
        ExpectedSource(source="manual.pdf", pages=[1, 1])

    with pytest.raises(ValidationError):
        ExpectedSource(source="manual.pdf", pages=[0])


def test_dataset_requires_unique_question_ids() -> None:
    with pytest.raises(ValidationError):
        EvaluationDataset(
            questions=[
                _positive_question("duplicate"),
                _positive_question("duplicate"),
            ]
        )


def test_successful_result_exposes_derived_fields() -> None:
    result = _successful_result()

    assert result.succeeded is True
    assert result.retrieval_scores == [0.84]
    assert result.faithfulness is None


def test_result_rejects_mixed_success_and_failure_state() -> None:
    with pytest.raises(ValidationError):
        EvalResult(
            question_id="q1",
            question="Test question",
            ground_truth="Test answer",
            generated_answer="Partial answer",
            model="test-model",
            error_type="GenerationError",
            duration_seconds=0.1,
            hit_rate=0.0,
            reciprocal_rank=0.0,
            source_page_accuracy=0.0,
        )


def test_summary_counts_must_match_result_statuses() -> None:
    result = _successful_result()

    summary = EvalSummary(
        total_questions=1,
        successful_questions=1,
        failed_questions=0,
        top_k=5,
        avg_hit_rate=1.0,
        avg_reciprocal_rank=1.0,
        avg_source_page_accuracy=0.5,
        avg_duration_seconds=1.2,
        results=[result],
    )

    assert summary.total_questions == 1
    assert summary.avg_faithfulness is None

    with pytest.raises(ValidationError):
        EvalSummary(
            total_questions=1,
            successful_questions=0,
            failed_questions=1,
            top_k=5,
            avg_hit_rate=1.0,
            avg_reciprocal_rank=1.0,
            avg_source_page_accuracy=0.5,
            avg_duration_seconds=1.2,
            results=[result],
        )

def test_negative_result_must_not_contain_retrieval_metrics() -> None:
    result = EvalResult(
        question_id="q-negative",
        question="Belgede bulunmayan soru",
        ground_truth=NOT_IN_CONTEXT,
        generated_answer="Bu bilgi verilen bağlamda bulunmuyor.",
        model="llama3.1:8b",
        retrieved_sources=[],
        duration_seconds=0.5,
    )

    assert result.hit_rate is None
    assert result.reciprocal_rank is None
    assert result.source_page_accuracy is None

    with pytest.raises(ValidationError):
        EvalResult(
            question_id="q-negative",
            question="Belgede bulunmayan soru",
            ground_truth=NOT_IN_CONTEXT,
            generated_answer="Bu bilgi verilen bağlamda bulunmuyor.",
            model="llama3.1:8b",
            duration_seconds=0.5,
            hit_rate=1.0,
        )