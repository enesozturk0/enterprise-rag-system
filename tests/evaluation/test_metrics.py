import pytest

from src.evaluation.metrics import (
    hit_rate,
    mean_reciprocal_rank,
    source_page_accuracy,
)
from src.evaluation.models import ExpectedSource, RetrievedSource


def _retrieved(
    source: str,
    page: int,
    score: float = 0.8,
) -> RetrievedSource:
    return RetrievedSource(
        source=source,
        page=page,
        score=score,
        content_preview="Test preview",
    )


def test_hit_rate_found() -> None:
    retrieved = [
        _retrieved("other.pdf", 1),
        _retrieved("manual.pdf", 263),
    ]
    expected = [ExpectedSource(source="manual.pdf", pages=[263])]

    assert hit_rate(retrieved, expected) == 1.0


def test_hit_rate_not_found() -> None:
    retrieved = [_retrieved("other.pdf", 1)]
    expected = [ExpectedSource(source="manual.pdf", pages=[263])]

    assert hit_rate(retrieved, expected) == 0.0


def test_hit_rate_is_not_applicable_to_negative_question() -> None:
    assert hit_rate([_retrieved("other.pdf", 1)], []) is None


def test_reciprocal_rank_first_position() -> None:
    retrieved = [
        _retrieved("manual.pdf", 263),
        _retrieved("other.pdf", 1),
    ]
    expected = [ExpectedSource(source="manual.pdf", pages=[263])]

    assert mean_reciprocal_rank(retrieved, expected) == 1.0


def test_reciprocal_rank_third_position() -> None:
    retrieved = [
        _retrieved("a.pdf", 1),
        _retrieved("b.pdf", 2),
        _retrieved("manual.pdf", 263),
    ]
    expected = [ExpectedSource(source="manual.pdf", pages=[263])]

    assert mean_reciprocal_rank(
        retrieved,
        expected,
    ) == pytest.approx(1 / 3)


def test_reciprocal_rank_not_found() -> None:
    retrieved = [_retrieved("other.pdf", 1)]
    expected = [ExpectedSource(source="manual.pdf", pages=[263])]

    assert mean_reciprocal_rank(retrieved, expected) == 0.0


def test_reciprocal_rank_is_not_applicable_to_negative_question() -> None:
    assert mean_reciprocal_rank([], []) is None


def test_source_page_accuracy_full_match() -> None:
    retrieved = [
        _retrieved("manual.pdf", 262),
        _retrieved("manual.pdf", 263),
    ]
    expected = [ExpectedSource(source="manual.pdf", pages=[262, 263])]

    assert source_page_accuracy(retrieved, expected) == 1.0


def test_source_page_accuracy_partial_match() -> None:
    retrieved = [_retrieved("manual.pdf", 262)]
    expected = [ExpectedSource(source="manual.pdf", pages=[262, 263, 264])]

    assert source_page_accuracy(
        retrieved,
        expected,
    ) == pytest.approx(1 / 3)


def test_source_page_accuracy_no_match() -> None:
    retrieved = [_retrieved("other.pdf", 1)]
    expected = [ExpectedSource(source="manual.pdf", pages=[263])]

    assert source_page_accuracy(retrieved, expected) == 0.0


def test_source_page_accuracy_is_not_applicable_to_negative_question() -> None:
    assert source_page_accuracy([], []) is None
