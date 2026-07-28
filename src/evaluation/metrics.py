"""Deterministic retrieval metrics that require no LLM judge."""

from __future__ import annotations

from collections.abc import Sequence

from src.evaluation.models import ExpectedSource, RetrievedSource


def hit_rate(
    retrieved_sources: Sequence[RetrievedSource],
    expected_sources: Sequence[ExpectedSource],
) -> float | None:
    """Return 1 when any expected document appears in retrieved results."""

    if not expected_sources:
        # Negatif sorular ayrı bir abstention metriğiyle ölçülecek.
        return None

    expected_files = {source.source for source in expected_sources}
    return float(any(result.source in expected_files for result in retrieved_sources))


def mean_reciprocal_rank(
    retrieved_sources: Sequence[RetrievedSource],
    expected_sources: Sequence[ExpectedSource],
) -> float | None:
    """Return the reciprocal rank of the first expected document."""

    if not expected_sources:
        return None

    expected_files = {source.source for source in expected_sources}

    for rank, result in enumerate(retrieved_sources, start=1):
        if result.source in expected_files:
            return 1.0 / rank

    return 0.0


def source_page_accuracy(
    retrieved_sources: Sequence[RetrievedSource],
    expected_sources: Sequence[ExpectedSource],
) -> float | None:
    """Return the fraction of expected document-page pairs retrieved."""

    if not expected_sources:
        return None

    expected_pairs = {(source.source, page) for source in expected_sources for page in source.pages}
    retrieved_pairs = {(source.source, source.page) for source in retrieved_sources}

    return len(expected_pairs & retrieved_pairs) / len(expected_pairs)
