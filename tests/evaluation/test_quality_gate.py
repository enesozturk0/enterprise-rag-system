import pytest

from src.evaluation.models import EvalSummary
from src.evaluation.quality_gate import (
    QualityThresholds,
    check_quality_gate,
)


def _summary(
    *,
    hit_rate: float | None,
    reciprocal_rank: float | None,
    source_page_accuracy: float | None,
) -> EvalSummary:
    return EvalSummary(
        total_questions=0,
        successful_questions=0,
        failed_questions=0,
        top_k=5,
        avg_hit_rate=hit_rate,
        avg_reciprocal_rank=reciprocal_rank,
        avg_source_page_accuracy=source_page_accuracy,
        avg_duration_seconds=0.0,
        results=[],
    )


def test_quality_gate_accepts_metrics_at_thresholds() -> None:
    summary = _summary(
        hit_rate=0.80,
        reciprocal_rank=0.60,
        source_page_accuracy=0.50,
    )

    assert check_quality_gate(summary) == ()


def test_quality_gate_reports_low_and_missing_metrics() -> None:
    summary = _summary(
        hit_rate=0.79,
        reciprocal_rank=None,
        source_page_accuracy=0.50,
    )

    violations = check_quality_gate(summary)

    assert [item.metric for item in violations] == [
        "avg_hit_rate",
        "avg_reciprocal_rank",
    ]
    assert violations[0].actual == 0.79
    assert violations[0].minimum == 0.80
    assert violations[1].actual is None


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("min_hit_rate", -0.01),
        ("min_reciprocal_rank", 1.01),
        ("min_source_page_accuracy", -1.0),
    ],
)
def test_thresholds_must_be_between_zero_and_one(
    field_name: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        QualityThresholds(**{field_name: value})