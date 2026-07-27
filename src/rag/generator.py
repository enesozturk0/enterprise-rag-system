"""Send prompts to Ollama and return the generated response."""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog

from src.rag.exceptions import GenerationError

logger = structlog.get_logger(__name__)


class LLMGenerator:
    """Call Ollama /api/generate for non-streaming completions."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url cannot be empty.")
        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")
        if temperature < 0:
            raise ValueError("temperature cannot be negative.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")

        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = client or httpx.Client(timeout=120.0)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not system_prompt.strip():
            raise GenerationError("System prompt cannot be empty.")
        if not user_prompt.strip():
            raise GenerationError("User prompt cannot be empty.")

        started_at = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self._model_name,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }

        try:
            response = self._client.post(
                f"{self._base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()

            data = response.json()
            if not isinstance(data, dict):
                raise TypeError("Ollama response must be a JSON object.")

            answer = data["response"]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise GenerationError(
                "Failed to get a valid response from Ollama.",
                details={"model": self._model_name, "cause": str(exc)},
            ) from exc

        if not isinstance(answer, str) or not answer.strip():
            raise GenerationError(
                "Ollama returned an empty response.",
                details={"model": self._model_name},
            )

        answer = answer.strip()
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)

        logger.info(
            "generation_completed",
            model=self._model_name,
            prompt_length=len(user_prompt),
            answer_length=len(answer),
            duration_ms=duration_ms,
        )

        return answer
