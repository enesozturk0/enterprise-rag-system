"""Run ingestion, indexing, and similarity search end to end."""

from __future__ import annotations

import time
from pathlib import Path

import httpx

from src.config.settings import get_settings
from src.ingestion.pipeline import IngestionPipeline
from src.vectorstore import (
    ChromaVectorStore,
    DocumentIndexer,
    OllamaEmbeddingClient,
)


def main() -> None:
    started_at = time.perf_counter()
    settings = get_settings()
    source = Path("data/raw")

    ingestion_pipeline = IngestionPipeline(settings)
    ingestion_result = ingestion_pipeline.run(source)

    print("\nIngestion completed")
    print(f"Files: {ingestion_result.successful_files}/{ingestion_result.total_files}")
    print(f"Chunks: {ingestion_result.total_chunks}")

    if ingestion_result.failed_files:
        print("\nFailed files:")
        for failure in ingestion_result.failed_files:
            print(f"- {failure.file_name}: {failure.error_detail}")

    if not ingestion_result.documents:
        print("\nNo chunks were produced; indexing was skipped.")
        return

    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )

    # Context manager gerçek HTTP bağlantısının işlem sonunda kapanmasını sağlar.
    with httpx.Client(timeout=60.0) as http_client:
        embedding_client = OllamaEmbeddingClient(
            base_url=settings.ollama_base_url,
            model_name=settings.embedding_model_name,
            batch_size=settings.embedding_batch_size,
            client=http_client,
        )
        indexer = DocumentIndexer(
            embedding_client=embedding_client,
            vector_store=vector_store,
            batch_size=settings.embedding_batch_size,
        )

        indexing_result = indexer.index(ingestion_result.documents)

        query = "Bu doküman ne hakkında?"
        query_embedding = embedding_client.embed_text(query)
        search_results = vector_store.search(
            query_embedding=query_embedding,
            k=3,
        )

    print("\nIndexing completed")
    print(f"Indexed: {indexing_result.indexed_documents}")
    print(f"Failed: {indexing_result.failed_documents}")
    print(f"Batches: {indexing_result.total_batches}")
    print(f"Stored vectors: {vector_store.count()}")

    print(f"\nQuery: {query}")
    if not search_results:
        print("No search results found.")

    for position, result in enumerate(search_results, start=1):
        print(f"\nResult {position}")
        print(f"Source: {result.metadata.get('source')}")
        print(f"Page: {result.metadata.get('page')}")
        print(f"Score: {result.score:.4f}")
        print(f"Content: {result.content[:500]}")

    total_duration = time.perf_counter() - started_at
    print(f"\nTotal duration: {total_duration:.2f}s")


if __name__ == "__main__":
    main()
