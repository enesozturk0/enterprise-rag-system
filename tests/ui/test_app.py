from contextlib import nullcontext
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

import src.ui.app as app_module
from src.ui.api_client import (
    APIAuthenticationError,
    APIClient,
    APIConnectionError,
    APIResponseError,
    DeleteResponse,
    DocumentStats,
    QueryResponse,
    SourceDetail,
    UploadResponse,
)
from src.ui.app import (
    _api_error_message,
    _check_backend_health,
    _create_assistant_message,
    _handle_delete,
    _handle_upload,
    _initialize_session_state,
    _refresh_stats,
    _render_chat,
    _render_message_content,
    _render_sidebar,
    _render_sources,
)
from src.ui.config import UISettings, get_ui_settings


def _ui_settings() -> UISettings:
    return UISettings(
        _env_file=None,
        api_url="http://testserver",
        api_key="test-ui-api-key-123456",
        max_upload_size_mb=1,
    )


def _install_fake_streamlit(
    monkeypatch: pytest.MonkeyPatch,
) -> Mock:
    fake_st = Mock()
    fake_st.session_state = {
        "messages": [],
        "total_chunks": None,
        "backend_healthy": None,
    }
    fake_st.sidebar = nullcontext()
    fake_st.form.side_effect = lambda *args, **kwargs: nullcontext()
    fake_st.spinner.side_effect = lambda *args, **kwargs: nullcontext()
    fake_st.chat_message.side_effect = lambda *args, **kwargs: nullcontext()
    fake_st.expander.side_effect = lambda *args, **kwargs: nullcontext()
    monkeypatch.setattr(app_module, "st", fake_st)
    return fake_st


def test_session_initialization_preserves_existing_messages() -> None:
    state: dict[str, object] = {}

    _initialize_session_state(state)
    messages = state["messages"]
    assert isinstance(messages, list)

    messages.append({"role": "user", "content": "İlk soru"})
    _initialize_session_state(state)

    assert state["messages"] == [{"role": "user", "content": "İlk soru"}]
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
    client.query.side_effect = APIConnectionError("Sensitive connection detail.")

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

    client.check_health.side_effect = APIConnectionError("Sensitive network detail.")
    _check_backend_health(client, state)

    assert state["backend_healthy"] is False


def test_api_error_messages_are_safe_and_operation_specific() -> None:
    authentication_message = _api_error_message(
        APIAuthenticationError("Sensitive authentication detail."),
        operation="Sorgu",
    )
    response_message = _api_error_message(
        APIResponseError("Safe public detail."),
        operation="Yükleme",
    )

    assert "RAG_API_KEY" in authentication_message
    assert "Sensitive" not in authentication_message
    assert response_message == "Yükleme tamamlanamadı: Safe public detail."


def test_document_handlers_update_state_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _install_fake_streamlit(monkeypatch)
    client = Mock(spec=APIClient)
    uploaded_file = Mock()
    uploaded_file.name = "manual.pdf"
    uploaded_file.size = 100
    uploaded_file.getvalue.return_value = b"%PDF-test"

    client.upload_document.return_value = UploadResponse(
        file_name="manual.pdf",
        total_chunks=2,
        indexed_chunks=2,
        failed_chunks=0,
        successful_files=1,
        failed_files=0,
        processing_time_seconds=0.5,
        message="Document uploaded.",
    )
    client.delete_document.return_value = DeleteResponse(
        source="manual.pdf",
        chunks_deleted=2,
        message="Document deleted.",
    )
    client.get_document_stats.return_value = DocumentStats(total_chunks=7)

    _handle_upload(client, uploaded_file, _ui_settings())
    _handle_delete(client, "manual.pdf")
    _refresh_stats(client)

    client.upload_document.assert_called_once_with(
        "manual.pdf",
        b"%PDF-test",
    )
    client.delete_document.assert_called_once_with("manual.pdf")
    assert fake_st.session_state["total_chunks"] == 7
    assert fake_st.success.call_count == 2


def test_document_handlers_render_safe_api_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _install_fake_streamlit(monkeypatch)
    client = Mock(spec=APIClient)
    uploaded_file = Mock()
    uploaded_file.name = "manual.pdf"
    uploaded_file.size = 100
    uploaded_file.getvalue.return_value = b"%PDF-test"
    error = APIConnectionError("Sensitive network detail.")

    client.upload_document.side_effect = error
    client.delete_document.side_effect = error
    client.get_document_stats.side_effect = error

    _handle_upload(client, uploaded_file, _ui_settings())
    _handle_delete(client, "manual.pdf")
    _refresh_stats(client)

    assert fake_st.error.call_count == 3
    assert all("Sensitive" not in call.args[0] for call in fake_st.error.call_args_list)


def test_oversized_upload_is_rejected_before_api_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _install_fake_streamlit(monkeypatch)
    client = Mock(spec=APIClient)
    uploaded_file = Mock()
    uploaded_file.size = 2 * 1024 * 1024

    _handle_upload(client, uploaded_file, _ui_settings())

    client.upload_document.assert_not_called()
    fake_st.warning.assert_called_once()


def test_sidebar_dispatches_confirmed_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _install_fake_streamlit(monkeypatch)
    fake_st.slider.return_value = 4
    fake_st.text_input.side_effect = [
        " manual.pdf ",
        "delete.pdf",
    ]
    uploaded_file = Mock()
    fake_st.file_uploader.return_value = uploaded_file
    fake_st.form_submit_button.side_effect = [True, True]
    fake_st.checkbox.return_value = True
    fake_st.button.side_effect = [True, True]
    fake_st.session_state["backend_healthy"] = True

    client = Mock(spec=APIClient)
    upload_handler = Mock()
    delete_handler = Mock()
    stats_handler = Mock()
    health_handler = Mock()
    monkeypatch.setattr(app_module, "_handle_upload", upload_handler)
    monkeypatch.setattr(app_module, "_handle_delete", delete_handler)
    monkeypatch.setattr(app_module, "_refresh_stats", stats_handler)
    monkeypatch.setattr(
        app_module,
        "_check_backend_health",
        health_handler,
    )

    top_k, source_filter = _render_sidebar(_ui_settings(), client)

    assert (top_k, source_filter) == (4, "manual.pdf")
    upload_handler.assert_called_once_with(
        client,
        uploaded_file,
        _ui_settings(),
    )
    delete_handler.assert_called_once_with(client, "delete.pdf")
    stats_handler.assert_called_once_with(client)
    health_handler.assert_called_once_with(client, fake_st.session_state)
    fake_st.success.assert_called_once()


@pytest.mark.parametrize(
    (
        "uploaded_file",
        "delete_source",
        "confirmed",
        "expected_warning",
    ),
    [
        (None, "", False, "Önce bir PDF dosyası seçin."),
        (Mock(), "manual.pdf", False, "Silme işlemini önce onaylayın."),
    ],
)
def test_sidebar_validates_document_actions(
    monkeypatch: pytest.MonkeyPatch,
    uploaded_file: Mock | None,
    delete_source: str,
    confirmed: bool,
    expected_warning: str,
) -> None:
    fake_st = _install_fake_streamlit(monkeypatch)
    fake_st.slider.return_value = 5
    fake_st.text_input.side_effect = ["", delete_source]
    fake_st.file_uploader.return_value = uploaded_file
    fake_st.form_submit_button.side_effect = [True, True]
    fake_st.checkbox.return_value = confirmed
    fake_st.button.side_effect = [False, False]
    fake_st.session_state["backend_healthy"] = False
    monkeypatch.setattr(app_module, "_handle_upload", Mock())
    monkeypatch.setattr(app_module, "_handle_delete", Mock())

    _render_sidebar(_ui_settings(), Mock(spec=APIClient))

    warning_messages = [call.args[0] for call in fake_st.warning.call_args_list]
    assert expected_warning in warning_messages
    fake_st.error.assert_called_once()


def test_sidebar_renders_unknown_backend_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _install_fake_streamlit(monkeypatch)
    fake_st.slider.return_value = 5
    fake_st.text_input.side_effect = ["", ""]
    fake_st.file_uploader.return_value = None
    fake_st.form_submit_button.side_effect = [False, False]
    fake_st.checkbox.return_value = False
    fake_st.button.side_effect = [False, False]

    _render_sidebar(_ui_settings(), Mock(spec=APIClient))

    fake_st.caption.assert_called_once()


def test_sources_and_message_variants_are_rendered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _install_fake_streamlit(monkeypatch)
    sources = [
        SourceDetail(
            source="manual.pdf",
            page=1,
            score=0.9,
            content_preview="First source",
        ),
        SourceDetail(
            source="manual.pdf",
            page=2,
            score=0.8,
            content_preview="Second source",
        ),
    ]

    _render_sources(sources)
    _render_message_content(
        {
            "role": "user",
            "content": "User question",
        }
    )
    _render_message_content(
        {
            "role": "assistant",
            "content": "Assistant answer",
            "sources": sources,
            "model": "test-model",
            "duration_seconds": 0.5,
        }
    )
    _render_message_content(
        {
            "role": "assistant",
            "content": "Safe error",
            "is_error": True,
        }
    )

    assert fake_st.text.call_count >= 3
    fake_st.markdown.assert_called_once_with("Assistant answer")
    assert fake_st.error.call_args.args == ("Safe error",)
    assert fake_st.divider.call_count >= 2


def test_chat_appends_user_and_assistant_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _install_fake_streamlit(monkeypatch)
    fake_st.session_state["messages"] = [{"role": "assistant", "content": "Previous answer"}]
    fake_st.chat_input.return_value = "  New question  "
    assistant_message = {
        "role": "assistant",
        "content": "New answer",
    }
    create_message = Mock(return_value=assistant_message)
    monkeypatch.setattr(
        app_module,
        "_create_assistant_message",
        create_message,
    )

    _render_chat(
        Mock(spec=APIClient),
        top_k=3,
        source_filter="manual.pdf",
    )

    assert fake_st.session_state["messages"] == [
        {"role": "assistant", "content": "Previous answer"},
        {"role": "user", "content": "New question"},
        assistant_message,
    ]
    create_message.assert_called_once()


def test_chat_rejects_whitespace_only_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _install_fake_streamlit(monkeypatch)
    fake_st.chat_input.return_value = "   "

    _render_chat(
        Mock(spec=APIClient),
        top_k=5,
        source_filter=None,
    )

    assert fake_st.session_state["messages"] == []
    fake_st.warning.assert_called_once()
