from pathlib import Path

import yaml

from src.evaluation.models import EvaluationDataset

_DATASET_PATH = Path("data/eval/golden_qa.yaml")
_EXPECTED_SOURCE = "nt1100-2025-kullanim-kitapcigi-erVacc.pdf"


def test_golden_dataset_is_valid_and_balanced() -> None:
    raw_dataset = yaml.safe_load(_DATASET_PATH.read_text(encoding="utf-8"))
    dataset = EvaluationDataset.model_validate(raw_dataset)

    positive_questions = [
        question for question in dataset.questions if not question.expects_no_answer
    ]
    negative_questions = [question for question in dataset.questions if question.expects_no_answer]

    assert len(dataset.questions) == 20
    assert len(positive_questions) == 16
    assert len(negative_questions) == 4

    assert all(
        expected.source == _EXPECTED_SOURCE
        for question in positive_questions
        for expected in question.expected_sources
    )
    assert all(not question.expected_sources for question in negative_questions)
