"""Public evaluation contracts."""

from src.evaluation.exceptions import (
    BaseEvaluationError,
    DatasetLoadError,
    EvaluationPrerequisiteError,
    ResultWriteError,
)
from src.evaluation.metrics import (
    hit_rate,
    mean_reciprocal_rank,
    source_page_accuracy,
)
from src.evaluation.models import (
    NOT_IN_CONTEXT,
    EvalQuestion,
    EvalResult,
    EvalSummary,
    EvaluationDataset,
    ExpectedSource,
    RetrievedSource,
)
from src.evaluation.quality_gate import (
    DEFAULT_QUALITY_THRESHOLDS,
    QualityThresholds,
    QualityViolation,
    check_quality_gate,
)
from src.evaluation.runner import EvaluationRunner

__all__ = [
    "NOT_IN_CONTEXT",
    "BaseEvaluationError",
    "DatasetLoadError",
    "EvalQuestion",
    "EvalResult",
    "EvalSummary",
    "EvaluationRunner",
    "EvaluationDataset",
    "EvaluationPrerequisiteError",
    "ExpectedSource",
    "ResultWriteError",
    "RetrievedSource",
    "hit_rate",
    "mean_reciprocal_rank",
    "source_page_accuracy",
    "DEFAULT_QUALITY_THRESHOLDS",
    "QualityThresholds",
    "QualityViolation",
    "check_quality_gate",
]
