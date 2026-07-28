"""Deterministic quality thresholds for evaluation summaries."""

from __future__ import annotations

from dataclasses import dataclass

from src.evaluation.models import EvalSummary


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    """Minimum acceptable retrieval quality."""

    min_hit_rate: float = 0.80
    min_reciprocal_rank: float = 0.60
    min_source_page_accuracy: float = 0.50

    def __post_init__(self) -> None:
        values = {
            "min_hit_rate": self.min_hit_rate,
            "min_reciprocal_rank": self.min_reciprocal_rank,
            "min_source_page_accuracy": (self.min_source_page_accuracy),
        }

        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class QualityViolation:
    """One metric that could not satisfy its minimum threshold."""

    metric: str
    actual: float | None
    minimum: float


DEFAULT_QUALITY_THRESHOLDS = QualityThresholds()


def check_quality_gate(
    summary: EvalSummary,
    thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS,
) -> tuple[QualityViolation, ...]:
    """Return every missing or below-threshold retrieval metric."""
    checks = (
        (
            "avg_hit_rate",
            summary.avg_hit_rate,
            thresholds.min_hit_rate,
        ),
        (
            "avg_reciprocal_rank",
            summary.avg_reciprocal_rank,
            thresholds.min_reciprocal_rank,
        ),
        (
            "avg_source_page_accuracy",
            summary.avg_source_page_accuracy,
            thresholds.min_source_page_accuracy,
        ),
    )

    return tuple(
        QualityViolation(
            metric=metric,
            actual=actual,
            minimum=minimum,
        )
        for metric, actual, minimum in checks
        if actual is None or actual < minimum
    )
