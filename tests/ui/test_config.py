import pytest
from pydantic import ValidationError

from src.ui.config import UISettings, get_ui_settings


def test_ui_settings_load_prefixed_environment(monkeypatch) -> None:
    monkeypatch.setenv("RAG_API_URL", "http://127.0.0.1:9000/")
    monkeypatch.setenv("RAG_API_KEY", "test-ui-api-key-123456")
    monkeypatch.setenv("RAG_DEFAULT_TOP_K", "3")
    monkeypatch.setenv("RAG_MAX_TOP_K", "10")
    monkeypatch.setenv("RAG_MAX_UPLOAD_SIZE_MB", "30")
    get_ui_settings.cache_clear()

    try:
        settings = get_ui_settings()

        assert settings.api_base_url == "http://127.0.0.1:9000"
        assert settings.default_top_k == 3
        assert settings.max_top_k == 10
        assert "test-ui-api-key" not in str(settings.api_key)
        assert settings.max_upload_size_mb == 30
    finally:
        get_ui_settings.cache_clear()


def test_ui_settings_require_api_key(monkeypatch) -> None:
    monkeypatch.delenv("RAG_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        UISettings(_env_file=None)


def test_default_top_k_cannot_exceed_maximum() -> None:
    with pytest.raises(ValidationError):
        UISettings(
            _env_file=None,
            api_key="test-ui-api-key-123456",
            default_top_k=10,
            max_top_k=5,
        )
