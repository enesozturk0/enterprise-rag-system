from src.vectorstore import (
    ChromaVectorStore,
    DocumentIndexer,
    EmbeddingClient,
    OllamaEmbeddingClient,
    SearchResult,
    VectorStore,
)


def test_vectorstore_public_api_exports_expected_types() -> None:
    exported_types = (
        EmbeddingClient,
        OllamaEmbeddingClient,
        VectorStore,
        ChromaVectorStore,
        DocumentIndexer,
        SearchResult,
    )

    assert all(exported_type is not None for exported_type in exported_types)
