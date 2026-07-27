from typing import Any

import pytest

from src.rag.chain import RAGChain, RAGResponse
from src.rag.exceptions import ChainError, GenerationError
from src.rag.prompt import RAGPromptBuilder
from src.vectorstore.store import SearchResult


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.received_top_k: int | None = None
        self.received_filters: dict[str, Any] | None = None

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        self.received_top_k = top_k
        self.received_filters = filters
        return self.results


class FakeGenerator:
    def __init__(
        self, answer: str = "Üretilen cevap", error: GenerationError | None = None
    ) -> None:
        self.answer = answer
        self.error = error
        self.received_user_prompt: str | None = None
        self.call_count = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        self.received_user_prompt = user_prompt

        if self.error is not None:
            raise self.error

        return self.answer


def make_result() -> SearchResult:
    return SearchResult(
        id="manual.pdf::10::0",
        content="Motor yağı düz zeminde kontrol edilmelidir." * 10,
        metadata={"source": "manual.pdf", "page": 10},
        score=0.91,
    )


def test_query_returns_response_with_sources_and_options() -> None:
    retriever = FakeRetriever([make_result()])
    generator = FakeGenerator()
    chain = RAGChain(
        retriever=retriever,  # type: ignore[arg-type]
        prompt_builder=RAGPromptBuilder(),
        generator=generator,  # type: ignore[arg-type]
        model_name="llama3.1:8b",
    )

    response = chain.query(
        question="Motor yağı nasıl kontrol edilir?",
        top_k=3,
        filters={"source": "manual.pdf"},
    )

    assert isinstance(response, RAGResponse)
    assert response.answer == "Üretilen cevap"
    assert response.model == "llama3.1:8b"
    assert response.retrieval_scores == [0.91]
    assert response.duration_seconds > 0

    assert len(response.sources) == 1
    assert response.sources[0].source == "manual.pdf"
    assert response.sources[0].page == 10
    assert response.sources[0].score == 0.91
    assert len(response.sources[0].content_preview) == 200

    assert retriever.received_top_k == 3
    assert retriever.received_filters == {"source": "manual.pdf"}


def test_query_without_results_still_generates_answer() -> None:
    generator = FakeGenerator(answer="İlgili bilgi bulunamadı.")
    chain = RAGChain(
        retriever=FakeRetriever([]),  # type: ignore[arg-type]
        prompt_builder=RAGPromptBuilder(),
        generator=generator,  # type: ignore[arg-type]
        model_name="test-model",
    )

    response = chain.query("Bilinmeyen bilgi nedir?")

    assert response.answer == "İlgili bilgi bulunamadı."
    assert response.sources == []
    assert response.retrieval_scores == []
    assert generator.received_user_prompt is not None
    assert "İlgili bağlam bulunamadı." in generator.received_user_prompt


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_raises_chain_error(question: str) -> None:
    chain = RAGChain(
        retriever=FakeRetriever([]),  # type: ignore[arg-type]
        prompt_builder=RAGPromptBuilder(),
        generator=FakeGenerator(),  # type: ignore[arg-type]
        model_name="test-model",
    )

    with pytest.raises(ChainError, match="Question cannot be empty"):
        chain.query(question)


def test_invalid_source_metadata_fails_before_generation() -> None:
    invalid_result = SearchResult(
        id="invalid-result",
        content="İçerik",
        metadata={"page": 1},
        score=0.5,
    )
    generator = FakeGenerator()
    chain = RAGChain(
        retriever=FakeRetriever([invalid_result]),  # type: ignore[arg-type]
        prompt_builder=RAGPromptBuilder(),
        generator=generator,  # type: ignore[arg-type]
        model_name="test-model",
    )

    with pytest.raises(ChainError, match="invalid source metadata"):
        chain.query("Test sorusu")

    assert generator.call_count == 0


@pytest.mark.parametrize("page", [None, 0, -1, True, "3"])
def test_invalid_page_metadata_fails_before_generation(page: object) -> None:
    invalid_result = SearchResult(
        id="invalid-page",
        content="İçerik",
        metadata={"source": "manual.pdf", "page": page},
        score=0.5,
    )
    generator = FakeGenerator()
    chain = RAGChain(
        retriever=FakeRetriever([invalid_result]),  # type: ignore[arg-type]
        prompt_builder=RAGPromptBuilder(),
        generator=generator,  # type: ignore[arg-type]
        model_name="test-model",
    )

    with pytest.raises(ChainError, match="invalid page metadata"):
        chain.query("Test sorusu")

    assert generator.call_count == 0


def test_generation_error_propagates_unchanged() -> None:
    original_error = GenerationError("Ollama unavailable")
    chain = RAGChain(
        retriever=FakeRetriever([make_result()]),  # type: ignore[arg-type]
        prompt_builder=RAGPromptBuilder(),
        generator=FakeGenerator(error=original_error),  # type: ignore[arg-type]
        model_name="test-model",
    )

    with pytest.raises(GenerationError) as captured:
        chain.query("Test sorusu")

    assert captured.value is original_error


def test_constructor_rejects_empty_model_name() -> None:
    with pytest.raises(ValueError, match="model_name cannot be empty"):
        RAGChain(
            retriever=FakeRetriever([]),  # type: ignore[arg-type]
            prompt_builder=RAGPromptBuilder(),
            generator=FakeGenerator(),  # type: ignore[arg-type]
            model_name="   ",
        )
