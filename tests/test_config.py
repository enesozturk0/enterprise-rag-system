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

def test_env_override(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "1024")

    settings = Settings(_env_file=None)

    assert settings.chunk_size == 1024

def test_invalid_chunk_size():
    with pytest.raises(pydantic.ValidationError):
        Settings(_env_file=None, chunk_size=-1)

def test_api_key_is_masked():
    settings = Settings(_env_file=None, api_key="secret-value-123")

    masked = str(settings.api_key)

    assert "secret-value-123" not in masked
    assert settings.api_key.get_secret_value() == "secret-value-123"