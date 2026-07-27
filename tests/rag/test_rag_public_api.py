from src.rag import (
    DocumentRetriever,
    LLMGenerator,
    RAGChain,
    RAGPromptBuilder,
    RAGResponse,
    SourceInfo,
)


def test_rag_public_api_exports_expected_types() -> None:
    exported_types = (
        DocumentRetriever,
        LLMGenerator,
        RAGChain,
        RAGPromptBuilder,
        RAGResponse,
        SourceInfo,
    )

    assert all(exported_type is not None for exported_type in exported_types)
