"""Streamlit entry point for the Enterprise RAG UI."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, Protocol, TypedDict, cast

import streamlit as st

from src.ui.api_client import (
    APIAuthenticationError,
    APIClient,
    APIClientError,
    APIConnectionError,
    SourceDetail,
)
from src.ui.config import UISettings, get_ui_settings


class ChatMessage(TypedDict):
    """One message persisted between Streamlit reruns."""

    role: Literal["user", "assistant"]
    content: str
    sources: NotRequired[list[SourceDetail]]
    model: NotRequired[str]
    duration_seconds: NotRequired[float]
    is_error: NotRequired[bool]


class SessionStateLike(Protocol):
    """Minimum state interface required by UI helpers."""

    def __contains__(self, key: object) -> bool: ...

    def __setitem__(self, key: str, value: Any) -> None: ...


class UploadedPDF(Protocol):
    """Required interface of a Streamlit uploaded PDF."""

    name: str
    size: int

    def getvalue(self) -> bytes:
        """Return the uploaded file content."""

        ...


def _initialize_session_state(
    state: SessionStateLike,
) -> None:
    """Create session values without overwriting the existing history."""

    if "messages" not in state:
        state["messages"] = []

    if "total_chunks" not in state:
        state["total_chunks"] = None

    if "backend_healthy" not in state:
        state["backend_healthy"] = None


@st.cache_resource(show_spinner=False)
def get_api_client() -> APIClient:
    """Create one reusable HTTP connection pool per Streamlit process."""

    return APIClient(get_ui_settings())


def _api_error_message(
    error: APIClientError,
    *,
    operation: str,
) -> str:
    """Return a safe message suitable for display and chat history."""

    if isinstance(error, APIAuthenticationError):
        return "API kimlik doğrulaması başarısız. RAG_API_KEY ayarını kontrol edin."

    if isinstance(error, APIConnectionError):
        return "Backend servisine ulaşılamıyor."

    return f"{operation} tamamlanamadı: {error}"


def _render_api_error(
    error: APIClientError,
    *,
    operation: str,
) -> None:
    """Display a safe, user-oriented API error."""

    st.error(_api_error_message(error, operation=operation))


def _handle_upload(
    client: APIClient,
    uploaded_file: UploadedPDF,
    settings: UISettings,
) -> None:
    """Validate and upload one selected PDF."""

    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024

    if uploaded_file.size > max_size_bytes:
        st.warning(f"PDF en fazla {settings.max_upload_size_mb} MB olabilir.")
        return

    with st.spinner("Doküman işleniyor ve indeksleniyor..."):
        try:
            result = client.upload_document(
                uploaded_file.name,
                uploaded_file.getvalue(),
            )
        except APIClientError as error:
            _render_api_error(error, operation="Yükleme")
            return

    st.session_state["total_chunks"] = None
    st.success(
        f"{result.file_name} yüklendi. "
        f"{result.indexed_chunks} chunk indekslendi "
        f"({result.processing_time_seconds:.1f} saniye)."
    )


def _handle_delete(
    client: APIClient,
    source: str,
) -> None:
    """Delete one document after explicit confirmation."""

    with st.spinner(f"{source} siliniyor..."):
        try:
            result = client.delete_document(source)
        except APIClientError as error:
            _render_api_error(error, operation="Silme")
            return

    st.session_state["total_chunks"] = None
    st.success(f"{result.source} silindi. {result.chunks_deleted} chunk kaldırıldı.")


def _refresh_stats(client: APIClient) -> None:
    """Fetch current index statistics after a user action."""

    try:
        stats = client.get_document_stats()
    except APIClientError as error:
        _render_api_error(error, operation="İstatistik sorgusu")
        return

    st.session_state["total_chunks"] = stats.total_chunks


def _check_backend_health(
    client: APIClient,
    state: SessionStateLike,
) -> None:
    """Update backend health only after an explicit user action."""

    try:
        state["backend_healthy"] = client.check_health()
    except APIClientError:
        # Teknik bağlantı ayrıntısını kullanıcıya sızdırmıyoruz.
        state["backend_healthy"] = False


def _render_sidebar(
    settings: UISettings,
    client: APIClient,
) -> tuple[int, str | None]:
    """Render controlled document-management operations."""

    with st.sidebar:
        st.header("Sorgu Ayarları")

        top_k = st.slider(
            "Getirilecek Chunk sayısı",
            min_value=1,
            max_value=settings.max_top_k,
            value=settings.default_top_k,
        )
        source_filter_input = st.text_input(
            "Kaynak Filtresi",
            placeholder="kullanim-kilavuzu.pdf",
            key="query_source_filter",
        )

        st.divider()
        st.header("Doküman Yönetimi")

        with st.form("upload_document", clear_on_submit=True):
            uploaded_file = st.file_uploader(
                "PDF dosyası",
                type=["pdf"],
            )
            upload_submitted = st.form_submit_button(
                "Yükle ve İndeksle",
                use_container_width=True,
            )

        if upload_submitted:
            if uploaded_file is None:
                st.warning("Önce bir PDF dosyası seçin.")
            else:
                _handle_upload(client, uploaded_file, settings)

        st.divider()

        with st.form("delete_document", clear_on_submit=True):
            source = st.text_input(
                "Silinecek dosya adı",
                placeholder="kullanim-kilavuzu.pdf",
            )
            confirmed = st.checkbox("Bu dokümanın tüm chunk'larını silmeyi onaylıyorum.")
            delete_submitted = st.form_submit_button(
                "Dokümanı Sil",
                type="secondary",
                use_container_width=True,
            )

        if delete_submitted:
            if not source.strip():
                st.warning("Silinecek dosyanın adını girin.")
            elif not confirmed:
                st.warning("Silme işlemini önce onaylayın.")
            else:
                _handle_delete(client, source.strip())

        st.divider()
        st.subheader("İstatistikler")

        if st.button(
            "İstatistikleri Yenile",
            use_container_width=True,
        ):
            _refresh_stats(client)

        total_chunks = st.session_state["total_chunks"]
        st.metric(
            "Toplam chunk",
            total_chunks if total_chunks is not None else "—",
        )
        st.divider()
        st.subheader("Backend Durumu")

        if st.button(
            "Bağlantıyı Kontrol Et",
            use_container_width=True,
        ):
            _check_backend_health(client, st.session_state)

        backend_healthy = st.session_state["backend_healthy"]
        if backend_healthy is True:
            st.success("Backend bağlantısı aktif.")
        elif backend_healthy is False:
            st.error("Backend servisine ulaşılamıyor.")
        else:
            st.caption("Bağlantı henüz kontrol edilmedi.")

        source_filter = source_filter_input.strip() or None
        return top_k, source_filter


def _get_messages() -> list[ChatMessage]:
    """Return the typed chat history of the current user session."""

    return cast(list[ChatMessage], st.session_state["messages"])


def _create_assistant_message(
    client: APIClient,
    *,
    question: str,
    top_k: int,
    source_filter: str | None,
) -> ChatMessage:
    """Request one answer and convert it into session-state data."""

    try:
        response = client.query(
            question,
            top_k=top_k,
            source_filter=source_filter,
        )
    except APIClientError as error:
        return {
            "role": "assistant",
            "content": _api_error_message(error, operation="Sorgu"),
            "is_error": True,
        }

    return {
        "role": "assistant",
        "content": response.answer,
        "sources": response.sources,
        "model": response.model,
        "duration_seconds": response.duration_seconds,
    }


def _render_sources(sources: list[SourceDetail]) -> None:
    """Render the document chunks supporting one answer."""

    with st.expander(f"Kaynaklar ({len(sources)})"):
        for index, source in enumerate(sources):
            # st.text kullanmak kaynak adını Markdown olarak yorumlamaz.
            st.text(f"{source.source} · Sayfa {source.page} · Skor {source.score:.2f}")
            st.caption(source.content_preview)

            if index < len(sources) - 1:
                st.divider()


def _render_message_content(message: ChatMessage) -> None:
    """Render the content and metadata of one chat message."""

    if message.get("is_error"):
        st.error(message["content"])
        return

    if message["role"] == "user":
        # Kullanıcı girdisini Markdown olarak çalıştırmıyoruz.
        st.text(message["content"])
    else:
        st.markdown(message["content"])

    sources = message.get("sources", [])
    if sources:
        _render_sources(sources)

    model = message.get("model")
    duration = message.get("duration_seconds")
    if model is not None and duration is not None:
        st.caption(f"Model: {model} · Süre: {duration:.2f} saniye · Kaynak: {len(sources)}")


def _render_chat(
    client: APIClient,
    *,
    top_k: int,
    source_filter: str | None,
) -> None:
    """Render chat history and process one newly submitted question."""

    messages = _get_messages()

    for message in messages:
        with st.chat_message(message["role"]):
            _render_message_content(message)

    prompt = st.chat_input("Sorunuzu yazın...")
    if prompt is None:
        return

    question = prompt.strip()
    if not question:
        st.warning("Lütfen boş olmayan bir soru yazın.")
        return

    user_message: ChatMessage = {
        "role": "user",
        "content": question,
    }
    messages.append(user_message)

    with st.chat_message("user"):
        _render_message_content(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Dokümanlarda aranıyor..."):
            assistant_message = _create_assistant_message(
                client,
                question=question,
                top_k=top_k,
                source_filter=source_filter,
            )

        # Render sırasında başka bir rerun olsa bile mesaj artık korunur.
        messages.append(assistant_message)
        _render_message_content(assistant_message)


def main() -> None:
    """Configure and render the Streamlit application shell."""

    settings = get_ui_settings()

    # Bu, uygulamanın ilk görsel Streamlit çağrısı olmalıdır.
    st.set_page_config(
        page_title=settings.page_title,
        page_icon=settings.page_icon,
        layout="wide",
    )

    _initialize_session_state(st.session_state)

    st.title(settings.page_title)
    st.caption("İndekslenen PDF dokümanları hakkında kaynaklı cevaplar alın.")

    client = get_api_client()
    top_k, source_filter = _render_sidebar(settings, client)
    _render_chat(
        client,
        top_k=top_k,
        source_filter=source_filter,
    )


if __name__ == "__main__":
    main()
