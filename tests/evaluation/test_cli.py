from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts import run_evaluation
from src.evaluation.exceptions import DatasetLoadError
from src.evaluation.models import EvalResult, EvalSummary


def _summary(*, failed_questions: int = 0) -> EvalSummary:
    if failed_questions:
        result = EvalResult(
            question_id="q1",
            question="Test sorusu",
            ground_truth="Test cevabı",
            error_type="GenerationError",
            duration_seconds=0.25,
        )
        metric_value = None
    else:
        result = EvalResult(
            question_id="q1",
            question="Test sorusu",
            ground_truth="Test cevabı",
            generated_answer="Üretilen cevap",
            model="test-model",
            duration_seconds=0.25,
            hit_rate=1.0,
            reciprocal_rank=1.0,
            source_page_accuracy=1.0,
        )
        metric_value = 1.0

    return EvalSummary(
        top_k=5,
        total_questions=1,
        successful_questions=1 - failed_questions,
        failed_questions=failed_questions,
        avg_hit_rate=metric_value,
        avg_reciprocal_rank=metric_value,
        avg_source_page_accuracy=metric_value,
        avg_duration_seconds=0.25,
        avg_faithfulness=None,
        avg_answer_relevancy=None,
        results=[result],
    )


def test_parse_args_uses_default_paths() -> None:
    args = run_evaluation.parse_args([])

    assert args.dataset == Path("data/eval/golden_qa.yaml")
    assert args.output == Path("data/eval/results.json")
    assert args.top_k is None
    assert args.min_hit_rate == 0.80
    assert args.min_reciprocal_rank == 0.60
    assert args.min_source_page_accuracy == 0.50


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--min-hit-rate", "-0.1"),
        ("--min-reciprocal-rank", "1.1"),
        ("--min-source-page-accuracy", "invalid"),
    ],
)
def test_parse_args_rejects_invalid_thresholds(
    option: str,
    value: str,
) -> None:
    with pytest.raises(SystemExit):
        run_evaluation.parse_args([option, value])


def test_parse_args_accepts_custom_values() -> None:
    args = run_evaluation.parse_args(
        [
            "--dataset",
            "custom.yaml",
            "--output",
            "custom.json",
            "--top-k",
            "8",
        ]
    )

    assert args.dataset == Path("custom.yaml")
    assert args.output == Path("custom.json")
    assert args.top_k == 8


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_parse_args_rejects_invalid_top_k(value: str) -> None:
    with pytest.raises(SystemExit):
        run_evaluation.parse_args(["--top-k", value])


def test_format_metric_handles_value_and_none() -> None:
    assert run_evaluation._format_metric(0.75) == "75.00%"
    assert run_evaluation._format_metric(None) == "N/A"


def test_main_returns_zero_for_successful_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(run_evaluation, "get_settings", Mock())
    monkeypatch.setattr(
        run_evaluation,
        "run_evaluation",
        Mock(return_value=_summary()),
    )

    exit_code = run_evaluation.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Evaluation tamamlandı" in captured.out
    assert "Başarısız çalışma: 0" in captured.out


def test_main_returns_one_when_a_question_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_evaluation, "get_settings", Mock())
    monkeypatch.setattr(
        run_evaluation,
        "run_evaluation",
        Mock(return_value=_summary(failed_questions=1)),
    )

    assert run_evaluation.main([]) == 1


def test_main_returns_two_for_evaluation_setup_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(run_evaluation, "get_settings", Mock())
    monkeypatch.setattr(
        run_evaluation,
        "run_evaluation",
        Mock(side_effect=DatasetLoadError("Dataset okunamadı.")),
    )

    exit_code = run_evaluation.main([])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Evaluation çalıştırılamadı" in captured.err


def test_main_returns_three_when_quality_gate_fails(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    regressed_summary = _summary().model_copy(update={"avg_hit_rate": 0.79})

    monkeypatch.setattr(run_evaluation, "get_settings", Mock())
    monkeypatch.setattr(
        run_evaluation,
        "run_evaluation",
        Mock(return_value=regressed_summary),
    )

    exit_code = run_evaluation.main([])

    captured = capsys.readouterr()
    assert exit_code == 3
    assert "Quality gate failed" in captured.err
    assert "avg_hit_rate" in captured.err
