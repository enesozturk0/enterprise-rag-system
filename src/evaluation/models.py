"""Validated data contracts for RAG evaluation."""

from __future__ import annotations

from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

NOT_IN_CONTEXT = "__NOT_IN_CONTEXT__"


class ExpectedSource(BaseModel):
    """One document and its pages expected to support an answer."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        frozen=True,
    )

    source: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[^/\\]+$",
    )
    pages: list[int] = Field(min_length=1)

    @field_validator("pages")
    @classmethod
    def pages_must_be_positive_and_unique(
        cls,
        pages: list[int],
    ) -> list[int]:
        if any(isinstance(page, bool) or page < 1 for page in pages):
            raise ValueError("Expected page numbers must be positive integers.")

        if len(pages) != len(set(pages)):
            raise ValueError("Expected page numbers must be unique.")

        return pages


class EvalQuestion(BaseModel):
    """One golden-dataset question with its expected evidence."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        frozen=True,
    )

    id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    question: str = Field(min_length=1, max_length=2000)
    ground_truth: str = Field(min_length=1)
    expected_sources: list[ExpectedSource] = Field(default_factory=list)

    @property
    def expects_no_answer(self) -> bool:
        """Return whether the answer must be absent from the context."""

        return self.ground_truth == NOT_IN_CONTEXT

    @model_validator(mode="after")
    def source_expectations_must_match_question_type(self) -> Self:
        if self.expects_no_answer and self.expected_sources:
            raise ValueError(
                "Negative questions cannot define expected sources."
            )

        if not self.expects_no_answer and not self.expected_sources:
            raise ValueError(
                "Positive questions must define expected sources."
            )

        return self


class EvaluationDataset(BaseModel):
    """Validated collection of golden evaluation questions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    questions: list[EvalQuestion] = Field(default_factory=list)

    @field_validator("questions")
    @classmethod
    def question_ids_must_be_unique(
        cls,
        questions: list[EvalQuestion],
    ) -> list[EvalQuestion]:
        ids = [question.id for question in questions]

        if len(ids) != len(set(ids)):
            raise ValueError("Evaluation question IDs must be unique.")

        return questions


class RetrievedSource(BaseModel):
    """One source returned by the RAG chain during evaluation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        frozen=True,
    )

    source: str = Field(min_length=1, max_length=255)
    page: int = Field(ge=1)
    score: float = Field(ge=0.0, le=1.0)
    content_preview: str = Field(max_length=200)


class EvalResult(BaseModel):
    """Evaluation output for one golden question."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        frozen=True,
    )

    question_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1)
    ground_truth: str = Field(min_length=1)

    generated_answer: str | None = None
    model: str | None = None
    error_type: str | None = None

    retrieved_sources: list[RetrievedSource] = Field(default_factory=list)
    duration_seconds: float = Field(ge=0.0)

    hit_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    reciprocal_rank: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        )
    source_page_accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        )

    faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_relevancy: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def success_and_failure_fields_must_be_consistent(self) -> Self:
        retrieval_metrics = (
            self.hit_rate,
            self.reciprocal_rank,
            self.source_page_accuracy,
        )

        if self.error_type is not None:
            if self.generated_answer is not None or self.model is not None:
                raise ValueError(
                    "Failed results cannot contain an answer or model."
                )

            if any(metric is not None for metric in retrieval_metrics):
                raise ValueError(
                    "Failed results cannot contain retrieval metrics."
                )

            return self

        if not self.generated_answer or not self.model:
            raise ValueError(
                "Successful results require an answer and model."
            )

        is_negative = self.ground_truth == NOT_IN_CONTEXT
        if is_negative:
            if any(metric is not None for metric in retrieval_metrics):
                raise ValueError(
                    "Negative questions cannot contain retrieval metrics."
                )
        elif any(metric is None for metric in retrieval_metrics):
            raise ValueError(
                "Positive questions require all retrieval metrics."
            )

        return self

    @computed_field(return_type=bool)
    @property
    def succeeded(self) -> bool:
        """Return whether evaluation completed successfully."""

        return self.error_type is None

    @computed_field(return_type=list[float])
    @property
    def retrieval_scores(self) -> list[float]:
        """Return scores without storing duplicate mutable state."""

        return [source.score for source in self.retrieved_sources]


class EvalSummary(BaseModel):
    """Aggregated metrics and per-question evaluation results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_questions: int = Field(ge=0)
    successful_questions: int = Field(ge=0)
    failed_questions: int = Field(ge=0)
    top_k: int = Field(ge=1)

    avg_hit_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    avg_reciprocal_rank: float | None = Field(default=None, ge=0.0, le=1.0)
    avg_source_page_accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    avg_duration_seconds: float = Field(ge=0.0)

    avg_faithfulness: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    avg_answer_relevancy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    results: list[EvalResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def counts_must_match_results(self) -> Self:
        if self.total_questions != len(self.results):
            raise ValueError("total_questions must match results length.")

        if (
            self.successful_questions + self.failed_questions
            != self.total_questions
        ):
            raise ValueError(
                "Successful and failed counts must equal total_questions."
            )

        actual_successes = sum(result.succeeded for result in self.results)
        if self.successful_questions != actual_successes:
            raise ValueError(
                "successful_questions does not match result statuses."
            )

        return self