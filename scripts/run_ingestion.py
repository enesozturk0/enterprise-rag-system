"""Run the ingestion pipeline manually for local testing."""

from pathlib import Path

from src.config.settings import get_settings
from src.ingestion.pipeline import IngestionPipeline


def main() -> None:
    settings = get_settings()
    pipeline = IngestionPipeline(settings)

    source = Path("data/raw")

    print(f"Running ingestion pipeline for: {source.resolve()}")

    result = pipeline.run(source)

    print("\nIngestion completed")
    print(f"Duration: {result.processing_time_seconds:.2f}s")
    print(f"Files processed: {result.successful_files}/{result.total_files}")
    print(f"Chunks created: {result.total_chunks}")
    print(f"Failed files: {len(result.failed_files)}")

    if result.failed_files:
        print("\nFailures:")
        for failed in result.failed_files:
            print(f"- {failed.file_name}: {failed.error_detail}")

    if result.documents:
        sample = result.documents[0]

        print("\nFirst chunk")
        print(f"Source: {sample.metadata.source}")
        print(f"Page: {sample.metadata.page}")
        print(f"Chunk: {sample.metadata.chunk_index}")
        print(f"Preview: {sample.content[:1000]}...")


if __name__ == "__main__":
    main()
