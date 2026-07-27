from unittest.mock import Mock

from streamlit.testing.v1 import AppTest

from src.ui.api_client import (
    APIClient,
    APIConnectionError,
    QueryResponse,
)
from src.ui.app import (
    _check_backend_health,
    _create_assistant_message,
    _initialize_session_state,
)
from src.ui.config import get_ui_settings


def test_session_initialization_preserves_existing_messages() -> None:
    state: dict[str, object] = {}

    _initialize_session_state(state)
    messages = state["messages"]
    assert isinstance(messages, list)

    messages.append({"role": "user", "content": "İlk soru"})
    _initialize_session_state(state)

    assert state["messages"] == [
        {"role": "user", "content": "İlk soru"}
    ]
    assert state["total_chunks"] is None
    assert state["backend_healthy"] is None


def test_app_shell_starts_without_exception(monkeypatch) -> None:
    monkeypatch.setenv("RAG_API_KEY", "test-ui-api-key-123456")
    monkeypatch.setenv("RAG_PAGE_TITLE", "Test RAG")
    get_ui_settings.cache_clear()

    try:
        app = AppTest.from_file("src/ui/app.py").run()

        assert not app.exception
        assert app.title[0].value == "Test RAG"
        assert app.session_state["messages"] == []
        assert [button.label for button in app.button] == [
            "Yükle ve İndeksle",
            "Dokümanı Sil",
            "İstatistikleri Yenile",
            "Bağlantıyı Kontrol Et",
        ]
        assert app.slider[0].value == 5
        assert app.chat_input[0].placeholder == "Sorunuzu yazın..."
    finally:
        get_ui_settings.cache_clear()

def test_assistant_message_preserves_answer_sources_and_metadata() -> None:
    client = Mock(spec=APIClient)
    client.query.return_value = QueryResponse.model_validate(
        {
            "answer": "Yağ seviyesi düz zeminde kontrol edilir.",
            "sources": [
                {
                    "source": "manual.pdf",
                    "page": 263,
                    "score": 0.84,
                    "content_preview": "Motor yağının kontrol edilmesi...",
                }
            ],
            "model": "llama3.1:8b",
            "duration_seconds": 1.2,
        }
    )

    message = _create_assistant_message(
        client,
        question="Motor yağı nasıl kontrol edilir?",
        top_k=3,
        source_filter="manual.pdf",
    )

    client.query.assert_called_once_with(
        "Motor yağı nasıl kontrol edilir?",
        top_k=3,
        source_filter="manual.pdf",
    )
    assert message["content"].startswith("Yağ seviyesi")
    assert message["sources"][0].page == 263
    assert message["model"] == "llama3.1:8b"
    assert message["duration_seconds"] == 1.2


def test_connection_error_becomes_safe_assistant_message() -> None:
    client = Mock(spec=APIClient)
    client.query.side_effect = APIConnectionError(
        "Sensitive connection detail."
    )

    message = _create_assistant_message(
        client,
        question="Test sorusu",
        top_k=5,
        source_filter=None,
    )

    assert message == {
        "role": "assistant",
        "content": "Backend servisine ulaşılamıyor.",
        "is_error": True,
    }
    assert "Sensitive" not in message["content"]

def test_backend_health_is_stored_without_leaking_errors() -> None:
    client = Mock(spec=APIClient)
    state: dict[str, object] = {}

    client.check_health.return_value = True
    _check_backend_health(client, state)

    assert state["backend_healthy"] is True

    client.check_health.side_effect = APIConnectionError(
        "Sensitive network detail."
    )
    _check_backend_health(client, state)

    assert state["backend_healthy"] is False