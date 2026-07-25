from src.vectorstore.embeddings import OllamaEmbeddingClient


class FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, list[list[float]]]:
        return {"embeddings": [[0.1] * 1024]}


class FakeHttpClient:
    def post(self, url: str, json: dict[str, object]) -> FakeResponse:
        return FakeResponse()


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
