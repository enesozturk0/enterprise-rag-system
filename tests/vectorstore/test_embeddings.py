import httpx
import pytest

from src.config.settings import get_settings
from src.vectorstore.embeddings import OllamaEmbeddingClient
from src.vectorstore.exceptions import EmbeddingError


class FakeResponse:
    def __init__(self, embedding_count: int) -> None:
        self._embedding_count = embedding_count

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, list[list[float]]]:
        return {"embeddings": [[0.1] * 1024 for _ in range(self._embedding_count)]}


class FakeHttpClient:
    def post(self, url: str, json: dict[str, object]) -> FakeResponse:
        inputs = json["input"]
        assert isinstance(inputs, list)

        return FakeResponse(embedding_count=len(inputs))


class FailingHttpClient:
    def post(self, url: str, json: dict[str, object]) -> FakeResponse:
        request = httpx.Request("POST", url)
        raise httpx.ConnectError("connection refused", request=request)


def test_embed_batch_empty_input_returns_empty_list() -> None:
    client = OllamaEmbeddingClient(
        base_url="http://unused",
        model_name="bge-m3",
        batch_size=32,
        client=FakeHttpClient(),  # type: ignore[arg-type]
    )

    assert client.embed_batch([]) == []


def test_embed_text_returns_vector_with_expected_dimension() -> None:
    client = OllamaEmbeddingClient(
        base_url="http://unused",
        model_name="bge-m3",
        batch_size=32,
        client=FakeHttpClient(),  # type: ignore[arg-type]
    )

    vector = client.embed_text("Hello, world!")

    assert len(vector) == 1024
    assert all(isinstance(value, float) for value in vector)


def test_embed_batch_returns_matching_count() -> None:
    client = OllamaEmbeddingClient(
        base_url="http://unused",
        model_name="bge-m3",
        batch_size=2,
        client=FakeHttpClient(),  # type: ignore[arg-type]
    )

    embeddings = client.embed_batch(["bir", "iki", "üç"])

    assert len(embeddings) == 3
    assert all(len(vector) == 1024 for vector in embeddings)


def test_connection_error_raises_embedding_error() -> None:
    client = OllamaEmbeddingClient(
        base_url="http://unreachable",
        model_name="bge-m3",
        batch_size=32,
        client=FailingHttpClient(),  # type: ignore[arg-type]
    )

    with pytest.raises(EmbeddingError, match="valid embeddings"):
        client.embed_text("test")


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        OllamaEmbeddingClient(
            base_url="http://unused",
            model_name="bge-m3",
            batch_size=0,
        )


@pytest.mark.integration
def test_integration_embed_text_returns_float_vector() -> None:
    settings = get_settings()

    with httpx.Client(timeout=60.0) as http_client:
        client = OllamaEmbeddingClient(
            base_url=settings.ollama_base_url,
            model_name=settings.embedding_model_name,
            batch_size=settings.embedding_batch_size,
            client=http_client,
        )

        vector = client.embed_text("enterprise RAG system")

    assert len(vector) == 1024
    assert all(isinstance(value, float) for value in vector)


@pytest.mark.integration
def test_integration_embed_batch_returns_matching_count() -> None:
    settings = get_settings()
    texts = ["birinci metin", "ikinci metin", "üçüncü metin"]

    with httpx.Client(timeout=60.0) as http_client:
        client = OllamaEmbeddingClient(
            base_url=settings.ollama_base_url,
            model_name=settings.embedding_model_name,
            batch_size=settings.embedding_batch_size,
            client=http_client,
        )

        embeddings = client.embed_batch(texts)

    assert len(embeddings) == len(texts)
    assert all(len(vector) == 1024 for vector in embeddings)
