from pathlib import Path
from unittest.mock import Mock

from src.api import dependencies
from src.config.settings import Settings
from src.ingestion.pipeline import IngestionPipeline
from src.rag.chain import RAGChain
from src.vectorstore.indexer import DocumentIndexer


def test_expensive_dependency_builders_reuse_instances(monkeypatch) -> None:
    cases = [
        (
            "_build_embedding_client",
            "OllamaEmbeddingClient",
            ("http://localhost:11434", "bge-m3", 32),
        ),
        (
            "_build_vector_store",
            "ChromaVectorStore",
            (Path("data/test-chroma"), "test-collection"),
        ),
        (
            "_build_llm_generator",
            "LLMGenerator",
            ("http://localhost:11434", "test-model", 0.1, 128),
        ),
    ]

    for builder_name, constructor_name, arguments in cases:
        builder = getattr(dependencies, builder_name)
        builder.cache_clear()

        instance = object()
        constructor = Mock(return_value=instance)
        monkeypatch.setattr(dependencies, constructor_name, constructor)

        try:
            first = builder(*arguments)
            second = builder(*arguments)

            assert first is second
            assert constructor.call_count == 1
        finally:
            builder.cache_clear()


def test_service_factories_wire_injected_dependencies() -> None:
    settings = Settings(_env_file=None)
    embedding_client = Mock()
    vector_store = Mock()
    generator = Mock()

    chain = dependencies.get_rag_chain(
        settings,
        embedding_client,
        vector_store,
        generator,
    )
    indexer = dependencies.get_document_indexer(
        settings,
        embedding_client,
        vector_store,
    )
    pipeline = dependencies.get_ingestion_pipeline(settings)

    assert isinstance(chain, RAGChain)
    assert isinstance(indexer, DocumentIndexer)
    assert isinstance(pipeline, IngestionPipeline)

    assert chain._retriever._embedding_client is embedding_client
    assert chain._retriever._vector_store is vector_store
    assert chain._generator is generator
    assert indexer._embedding_client is embedding_client
    assert indexer._vector_store is vector_store
