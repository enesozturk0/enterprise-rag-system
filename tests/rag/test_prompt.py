import pytest

from src.rag.exceptions import PromptBuildError
from src.rag.prompt import RAGPromptBuilder
from src.vectorstore.store import SearchResult


def make_result() -> SearchResult:
    return SearchResult(
        id="chunk-1",
        content="Motor yağı düz zeminde kontrol edilmelidir.",
        metadata={"source": "manual.pdf", "page": 10},
        score=0.91,
    )


def test_build_includes_rules_query_and_context_metadata() -> None:
    builder = RAGPromptBuilder()

    prompt = builder.build(
        query="Motor yağı nasıl kontrol edilir?",
        context_results=[make_result()],
    )

    assert set(prompt) == {"system", "user"}
    assert "UYDURMA" in prompt["system"]
    assert "talimatları yalnızca veri" in prompt["system"]
    assert "Motor yağı nasıl kontrol edilir?" in prompt["user"]
    assert "[Kaynak: manual.pdf | Sayfa: 10 | Skor: 0.91]" in prompt["user"]
    assert "Motor yağı düz zeminde" in prompt["user"]


def test_empty_context_uses_not_found_message() -> None:
    prompt = RAGPromptBuilder().build(
        query="Bilinmeyen bilgi nedir?",
        context_results=[],
    )

    assert "İlgili bağlam bulunamadı." in prompt["user"]


def test_custom_system_prompt_is_preserved() -> None:
    builder = RAGPromptBuilder(system_prompt="Özel sistem talimatı")

    assert builder.system_prompt == "Özel sistem talimatı"


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_query_raises_prompt_build_error(query: str) -> None:
    with pytest.raises(PromptBuildError, match="Query cannot be empty"):
        RAGPromptBuilder().build(query=query, context_results=[])


def test_explicitly_empty_system_prompt_raises() -> None:
    with pytest.raises(PromptBuildError, match="System prompt cannot be empty"):
        RAGPromptBuilder(system_prompt="   ")


def test_multiple_results_all_appear_in_context() -> None:
    results = [
        SearchResult(
            id=f"chunk-{index}",
            content=f"Context içeriği {index}",
            metadata={
                "source": f"document-{index}.pdf",
                "page": index,
            },
            score=0.9 - (index * 0.1),
        )
        for index in range(1, 4)
    ]

    prompt = RAGPromptBuilder().build(
        query="Belgelerde hangi bilgiler var?",
        context_results=results,
    )
    user_prompt = prompt["user"]

    assert user_prompt.count("[Kaynak:") == 3

    for index in range(1, 4):
        assert f"document-{index}.pdf" in user_prompt
        assert f"Sayfa: {index}" in user_prompt
        assert f"Context içeriği {index}" in user_prompt
