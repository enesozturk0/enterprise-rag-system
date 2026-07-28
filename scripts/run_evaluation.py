"""Golden dataset üzerinde RAG evaluation çalıştırır."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import httpx
from chromadb.errors import ChromaError

from src.config.settings import Settings, get_settings
from src.evaluation.exceptions import (
    BaseEvaluationError,
    EvaluationPrerequisiteError,
)
from src.evaluation.models import EvalSummary
from src.evaluation.runner import EvaluationRunner
from src.rag.chain import RAGChain
from src.rag.generator import LLMGenerator
from src.rag.prompt import RAGPromptBuilder
from src.rag.retriever import DocumentRetriever
from src.vectorstore.embeddings import OllamaEmbeddingClient
from src.vectorstore.store import ChromaVectorStore

from src.evaluation.quality_gate import (
    DEFAULT_QUALITY_THRESHOLDS,
    QualityThresholds,
    QualityViolation,
    check_quality_gate,
)


DEFAULT_DATASET_PATH = Path("data/eval/golden_qa.yaml")
DEFAULT_OUTPUT_PATH = Path("data/eval/results.json")


def _positive_integer(value: str) -> int:
    """Argparse seviyesinde pozitif tamsayı doğrulaması yapar."""
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
              f"Pozitif bir tamsayı bekleniyordu: {value!r}"
        ) from exc

    if parsed_value < 1:
        raise argparse.ArgumentTypeError(
            "Değer sıfırdan büyük olmalıdır."
        )

    return parsed_value

def _unit_interval(value: str) -> float:
    """Parse a floating-point threshold between zero and one."""
    try:
        parsed_value = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a decimal value: {value!r}"
        ) from exc

    if not 0.0 <= parsed_value <= 1.0:
        raise argparse.ArgumentTypeError(
            "Threshold must be between 0 and 1."
        )

    return parsed_value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI argümanlarını ayrıştırır."""
    parser = argparse.ArgumentParser(
        description="""Golden dataset üzerinde RAG evaluation çalıştırır."""
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Golden dataset yolu (varsayılan: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Sonuç JSON yolu (varsayılan: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--top-k",
        type=_positive_integer,
        default=None,
        help="Her soru için getirilecek kaynak sayısı.",
    )
    parser.add_argument(
        "--min-hit-rate",
        type=_unit_interval,
        default=DEFAULT_QUALITY_THRESHOLDS.min_hit_rate,
        help="Minimum acceptable average hit rate.",
    )
    parser.add_argument(
        "--min-reciprocal-rank",
        type=_unit_interval,
        default=(DEFAULT_QUALITY_THRESHOLDS.min_reciprocal_rank),
        help="Minimum acceptable average reciprocal rank.",
    )
    parser.add_argument(
        "--min-source-page-accuracy",
        type=_unit_interval,
        default=(DEFAULT_QUALITY_THRESHOLDS.min_source_page_accuracy),
        help="Minimum acceptable source-page accuracy.",
    )

    return parser.parse_args(argv)


@contextmanager
def build_chain(
    settings: Settings,
) -> Iterator[tuple[RAGChain, ChromaVectorStore]]:
    """CLI ömrü boyunca kullanılacak RAG bağımlılıklarını oluşturur."""
    with (
        httpx.Client(timeout=60.0) as embedding_http,
        httpx.Client(timeout=120.0) as generation_http,
    ):
        embedding_client = OllamaEmbeddingClient(
            base_url=settings.ollama_base_url,
            model_name=settings.embedding_model_name,
            batch_size=settings.embedding_batch_size,
            client=embedding_http,
        )

        vector_store = ChromaVectorStore(
            persist_dir=settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )

        retriever = DocumentRetriever(
            embedding_client=embedding_client,
            vector_store=vector_store,
            default_top_k=settings.retrieval_top_k,
        )

        generator = LLMGenerator(
            base_url=settings.ollama_base_url,
            model_name=settings.llm_model_name,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            client=generation_http,
        )

        chain = RAGChain(
            retriever=retriever,
            prompt_builder=RAGPromptBuilder(),
            generator=generator,
            model_name=settings.llm_model_name,
        )

        yield chain, vector_store


def run_evaluation(
    args: argparse.Namespace,
    settings: Settings,
) -> EvalSummary:
    """Dataset’i yükler, evaluation çalıştırır ve sonucu kaydeder."""
    # Geçersiz dataset varsa Ollama ve Chroma bağlantılarını kurmadan hata verir.
    dataset = EvaluationRunner.load_dataset(args.dataset)
    top_k = args.top_k or settings.retrieval_top_k

    try:
        with build_chain(settings) as (chain, vector_store):
            if vector_store.count() == 0:
                raise EvaluationPrerequisiteError(
                    "Vector store boş. Önce dokümanları indeksleyin."
                )

            runner = EvaluationRunner(chain=chain, top_k=top_k)
            summary = runner.evaluate(dataset.questions)
            runner.save_results(summary, args.output)

    except ChromaError as exc:
        # Chroma'nın iç hata ayrıntılarını kullanıcı çıktısına taşımıyoruz.
        raise EvaluationPrerequisiteError(
            "Vector store bağlantısı kurulamadı."
        ) from exc

    return summary


def _format_metric(value: float | None) -> str:
    """Pozitif sorulardaki oranı veya negatif sorulardaki N/A değerini gösterir."""
    return "N/A" if value is None else f"{value:.2%}"


def print_summary(
    summary: EvalSummary,
    dataset_path: Path,
    output_path: Path,
) -> None:
    """Evaluation özetini terminale yazdırır."""
    print("\nEvaluation tamamlandı")
    print(f"Dataset: {dataset_path}")
    print(f"Top-k: {summary.top_k}")
    print(f"Toplam soru: {summary.total_questions}")
    print(f"Başarılı çalışma: {summary.successful_questions}")
    print(f"Başarısız çalışma: {summary.failed_questions}")
    print(f"Ortalama süre: {summary.avg_duration_seconds:.2f} saniye")
    print(f"Hit rate: {_format_metric(summary.avg_hit_rate)}")
    print(
        "Reciprocal rank: "
        f"{_format_metric(summary.avg_reciprocal_rank)}"
    )
    print(
        "Kaynak-sayfa doğruluğu: "
        f"{_format_metric(summary.avg_source_page_accuracy)}"
    )
    print(
        "Faithfulness: "
        f"{_format_metric(summary.avg_faithfulness)}"
    )
    print(
        "Cevap ilgililiği: "
        f"{_format_metric(summary.avg_answer_relevancy)}"
    )
    print(f"Sonuç dosyası: {output_path}")

def print_quality_violations(
    violations: tuple[QualityViolation, ...],
) -> None:
    """Print quality regressions without hiding missing metrics."""
    print("\nQuality gate failed:", file=sys.stderr)

    for violation in violations:
        actual = (
            "N/A"
            if violation.actual is None
            else f"{violation.actual:.2%}"
        )
        print(
            f"- {violation.metric}: "
            f"actual={actual}, minimum={violation.minimum:.2%}",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI giriş noktası ve işletim sistemi çıkış kodu."""
    args = parse_args(argv)

    try:
        summary = run_evaluation(args, get_settings())
    except BaseEvaluationError as exc:
        print(f"Evaluation çalıştırılamadı: {exc}", file=sys.stderr)
        return 2

    print_summary(
        summary,
        dataset_path=args.dataset,
        output_path=args.output,
    )

    if summary.failed_questions:
        return 1

    thresholds = QualityThresholds(
        min_hit_rate=args.min_hit_rate,
        min_reciprocal_rank=args.min_reciprocal_rank,
        min_source_page_accuracy=args.min_source_page_accuracy,
    )
    violations = check_quality_gate(summary, thresholds)

    if violations:
        print_quality_violations(violations)
        return 3

    print("\nQuality gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())