"""Run an end-to-end RAG query against the indexed documents."""

from __future__ import annotations

import httpx

from src.config.settings import get_settings
from src.rag import (
    DocumentRetriever,
    LLMGenerator,
    RAGChain,
    RAGPromptBuilder,
)
from src.vectorstore import ChromaVectorStore, OllamaEmbeddingClient


def main() -> None:
    settings = get_settings()

    vector_store = ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )

    stored_vectors = vector_store.count()
    if stored_vectors == 0:
        print("No indexed documents found.")
        print("Run: .venv\\Scripts\\python.exe -m scripts.run_indexing")
        return

    with (
        httpx.Client(timeout=60.0) as embedding_http_client,
        httpx.Client(timeout=120.0) as generation_http_client,
    ):
        embedding_client = OllamaEmbeddingClient(
            base_url=settings.ollama_base_url,
            model_name=settings.embedding_model_name,
            batch_size=settings.embedding_batch_size,
            client=embedding_http_client,
        )
        retriever = DocumentRetriever(
            embedding_client=embedding_client,
            vector_store=vector_store,
            default_top_k=settings.retrieval_top_k,
        )
        prompt_builder = RAGPromptBuilder()
        generator = LLMGenerator(
            base_url=settings.ollama_base_url,
            model_name=settings.llm_model_name,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            client=generation_http_client,
        )
        chain = RAGChain(
            retriever=retriever,
            prompt_builder=prompt_builder,
            generator=generator,
            model_name=settings.llm_model_name,
        )

        question = "Motor yağı nasıl kontrol edilir?"
        response = chain.query(question)

    print(f"\nQuestion: {question}")
    print(f"\nAnswer:\n{response.answer}")
    print(f"\nModel: {response.model}")
    print(f"Duration: {response.duration_seconds:.2f}s")
    print(f"Indexed vectors: {stored_vectors}")

    print(f"\nSources ({len(response.sources)}):")
    for position, source in enumerate(response.sources, start=1):
        print(f"{position}. {source.source} | Page {source.page} | Score {source.score:.2f}")
        print(f"   {source.content_preview}")


if __name__ == "__main__":
    main()
