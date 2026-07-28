"""Exception hierarchy for evaluation infrastructure."""

from __future__ import annotations

from typing import Any


class BaseEvaluationError(Exception):
    """Root error for evaluation infrastructure failures."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.details = details


class DatasetLoadError(BaseEvaluationError):
    """Raised when a golden dataset cannot be loaded or validated."""


class ResultWriteError(BaseEvaluationError):
    """Raised when evaluation results cannot be saved."""

class EvaluationPrerequisiteError(BaseEvaluationError):
    """The required conditions were not met before running the evaluation."""