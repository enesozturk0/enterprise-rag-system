from __future__ import annotations

import time
from typing import ClassVar, Protocol

import httpx
import structlog

from src.vectorstore.exceptions import EmbeddingError

logger = structlog.get_logger(__name__)


class EmbeddingClient(Protocol):
    """Protocol for embedding clients."""

    DIMENSION: ClassVar[int]

    def embed_text(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class OllamaEmbeddingClient:
    """Embedding client using Ollama /api/embed endpoint"""

    DIMENSION: ClassVar[int] = 1024

    def __init__(
        self,
        base_url: str,
        model_name: str,
        batch_size: int,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._batch_size = batch_size
        self._client = client or httpx.Client(timeout=60.0)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        embeddings = self._embed(texts=[text])
        return embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            started_at = time.perf_counter()

            embeddings = self._embed(texts=batch)
            all_embeddings.extend(embeddings)

            logger.debug(
                "batch_embedded",
                batch_size=len(batch),
                duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )

        return all_embeddings

    def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.post(
                f"{self._base_url}/api/embed",
                json={"model": self._model_name, "input": texts},
            )
            response.raise_for_status()
            embeddings: list[list[float]] = response.json()["embeddings"]
        except (httpx.HTTPError, KeyError, TypeError) as exc:
            raise EmbeddingError(
                "Did not receive valid embeddings from Ollama.",
                details={"model": self._model_name, "cause": str(exc)},
            ) from exc

        if len(embeddings) != len(texts):
            raise EmbeddingError(
                "Ollama failed to return the expected number of embeddings.",
                details={"expected": len(texts), "received": len(embeddings)},
            )

        for vector in embeddings:
            if len(vector) != self.DIMENSION:
                raise EmbeddingError(
                    "Embedding size does not match expected dimension.",
                    details={"expected_dimension": self.DIMENSION, "actual_dimension": len(vector)},
                )

        return embeddings
