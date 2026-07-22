"""Text preprocessing: cleaning rules applied before chunking."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import structlog

logger = structlog.get_logger(__name__)


class TextPreprocessor:
    """Cleans raw page text before chunking.
    Header/footer detection is a corpus-level operation
    (requires comparing all pages of a document) and is
    intentionally kept separate from the per-text 'clean()'
    method - see 'detect_repeated_lines'.
    """

    def __init__(self, known_headers_footers: set[str] | None = None):
        self._known_headers_footers = known_headers_footers or set()

    def clean(self, text: str) -> str:
        """Apply all cleaning rules in sequence to a single page's text."""

        # Order matters: hyphen-joining needs raw \n positions, header
        # removal needs line structure, whitespace normalization must
        # run last since it collapses that structure.
        text = self._join_hyphenated_words(text)
        text = self._remove_headers_footers(text)
        text = self._strip_control_characters(text)
        text = self._normalize_whitespace(text)
        return text

    @staticmethod
    def _join_hyphenated_words(text: str) -> str:
        """Merge words split across a line break by a hyphen e.g. 'keli-\\nmeler'"""
        return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    def _remove_headers_footers(self, text: str) -> str:
        """Strip lines that were pre-identified as repeating headers/footers."""
        if not self._known_headers_footers:
            return text

        lines = text.split("\n")
        cleaned_lines = [line for line in lines if line.strip() not in self._known_headers_footers]
        return "\n".join(cleaned_lines)

    @staticmethod
    def _strip_control_characters(text: str) -> str:
        """Remove non-printable control characters while preserving all
        legitimate Unicode letters (Turkish, etc.)."""
        return "".join(
            ch for ch in text if ch in ("\n", "\t") or unicodedata.category(ch)[0] != "C"
        )

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse runs of spaces/tabs into one space, and runs of
        blank lines into at most one blank line (two newlines)."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def detect_repeated_lines(
        page_texts: list[str],
        scan_lines: int = 2,
        min_repeat_ratio: float = 0.8,
    ) -> set[str]:
        """Detect lines that repeat across pages (likely headers/footers)."""

        if len(page_texts) < 2:
            return set()

        line_counts: Counter[str] = Counter()
        for text in page_texts:
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            candidates = lines[:scan_lines] + lines[-scan_lines:]
            line_counts.update(set(candidates))

        threshold = len(page_texts) * min_repeat_ratio
        repeated = {line for line, count in line_counts.items() if count >= threshold}

        logger.info(
            "headers_footers_detected",
            candidates_found=len(repeated),
            total_pages=len(page_texts),
        )
        return repeated
