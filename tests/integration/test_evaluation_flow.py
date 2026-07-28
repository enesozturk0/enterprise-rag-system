"""Service-independent end-to-end test for the evaluation flow."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
import httpx

from src.evaluation.runner import EvaluationRunner
from src.ingestion.models import Document, DocumentMetadata
from src.rag.chain import RAGChain
from src.rag.generator import LLMGenerator
from src.rag.prompt import RAGPromptBuilder
from src.rag.retriever import DocumentRetriever
from src.vectorstore.embeddings import OllamaEmbeddingClient
from src.vectorstore.indexer import DocumentIndexer
from src.vectorstore.store import ChromaVectorStore

_VECTOR = [1.0] + [0.0] * (
    OllamaEmbeddingClient.DIMENSION - 1
)


def test_evaluation_flow_from_indexing_to_json(
    tmp_path: Path,
) -> None:
    request_counts = {
        "embed": 0,
        "generate": 0,
    }

    def ollama_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            request_counts["embed"] += 1
            payload: dict[str, Any] = json.loads(request.content)
            inputs = payload["input"]

            return httpx.Response(
                status_code=200,
                json={
                    "embeddings": [
                        _VECTOR.copy()
                        for _ in inputs
                    ]
                },
            )

        if request.url.path == "/api/generate":
            request_counts["generate"] += 1

            return httpx.Response(
                status_code=200,
                json={
                    "response": (
                        "Motor yağı düz zeminde yağ seviye "
                        "çubuğu kullanılarak kontrol edilir."
                    )
                },
            )

        return httpx.Response(status_code=404)

    transport = httpx.MockTransport(ollama_handler)

    with (
        httpx.Client(transport=transport) as embedding_http,
        httpx.Client(transport=transport) as generation_http,
    ):
        embedding_client = OllamaEmbeddingClient(
            base_url="http://ollama.test",
            model_name="test-embedding-model",
            batch_size=2,
            client=embedding_http,
        )

        vector_store = ChromaVectorStore(
            persist_dir=tmp_path / "unused",
            collection_name="evaluation_flow",
            client=chromadb.EphemeralClient(),
        )

        document = Document(
            content=(
                "Motor yağını kontrol etmek için motosikleti "
                "düz zemine alın ve yağ seviyesini kontrol edin."
            ),
            metadata=DocumentMetadata(
                source="manual.pdf",
                page=263,
                total_pages=300,
                chunk_index=0,
            ),
        )

        indexer = DocumentIndexer(
            embedding_client=embedding_client,
            vector_store=vector_store,
            batch_size=2,
        )
        indexing_result = indexer.index([document])

        assert indexing_result.indexed_documents == 1
        assert indexing_result.failed_documents == 0
        assert vector_store.count() == 1

        retriever = DocumentRetriever(
            embedding_client=embedding_client,
            vector_store=vector_store,
            default_top_k=1,
        )
        generator = LLMGenerator(
            base_url="http://ollama.test",
            model_name="test-generation-model",
            temperature=0.1,
            max_tokens=128,
            client=generation_http,
        )
        chain = RAGChain(
            retriever=retriever,
            prompt_builder=RAGPromptBuilder(),
            generator=generator,
            model_name="test-generation-model",
        )

        dataset_path = tmp_path / "golden.yaml"
        dataset_path.write_text(
            """
questions:
  - id: motor-oil
    question: Motor yağı nasıl kontrol edilir?
    ground_truth: Motor yağı düz zeminde kontrol edilir.
    expected_sources:
      - source: manual.pdf
        pages: [263]
""".strip(),
            encoding="utf-8",
        )

        dataset = EvaluationRunner.load_dataset(dataset_path)
        runner = EvaluationRunner(chain=chain, top_k=1)
        summary = runner.evaluate(dataset.questions)

        output_path = tmp_path / "results.json"
        runner.save_results(summary, output_path)

    saved_result = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert summary.total_questions == 1
    assert summary.successful_questions == 1
    assert summary.failed_questions == 0
    assert summary.avg_hit_rate == 1.0
    assert summary.avg_reciprocal_rank == 1.0
    assert summary.avg_source_page_accuracy == 1.0

    result = summary.results[0]
    assert result.generated_answer is not None
    assert result.retrieved_sources[0].source == "manual.pdf"
    assert result.retrieved_sources[0].page == 263

    assert saved_result["successful_questions"] == 1
    assert saved_result["results"][0]["question_id"] == "motor-oil"

    # Bir embedding indeksleme, diğeri sorgu için yapılır.
    assert request_counts == {
        "embed": 2,
        "generate": 1,
    }