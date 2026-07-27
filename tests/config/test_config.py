import pydantic
import pytest

from src.config.settings import Settings


def test_default_values():
    settings = Settings(_env_file=None)

    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 50
    assert settings.llm_model_name == "llama3.1:8b"
    assert settings.embedding_model_name == "bge-m3"
    assert settings.api_port == 8000
    assert settings.retrieval_top_k == 5
    assert settings.llm_temperature == 0.1
    assert settings.llm_max_tokens == 1024


def test_env_override(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "1024")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "3")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.0")
    monkeypatch.setenv("LLM_MAX_TOKENS", "512")

    settings = Settings(_env_file=None)

    assert settings.chunk_size == 1024
    assert settings.retrieval_top_k == 3
    assert settings.llm_temperature == 0.0
    assert settings.llm_max_tokens == 512


def test_invalid_chunk_size():
    with pytest.raises(pydantic.ValidationError):
        Settings(_env_file=None, chunk_size=-1)


def test_api_key_is_masked():
    settings = Settings(_env_file=None, api_key="secret-value-123")

    masked = str(settings.api_key)

    assert "secret-value-123" not in masked
    assert settings.api_key.get_secret_value() == "secret-value-123"


def test_cors_origins_are_normalized_and_deduplicated() -> None:
    settings = Settings(
        _env_file=None,
        api_cors_origins=[
            " http://localhost:8501/ ",
            "http://localhost:8501",
        ],
    )

    assert settings.api_cors_origins == ["http://localhost:8501"]

    with pytest.raises(pydantic.ValidationError):
        Settings(_env_file=None, api_cors_origins=[" * "])
