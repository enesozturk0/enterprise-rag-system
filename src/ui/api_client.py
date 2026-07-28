"""Typed contracts and errors for communication with the RAG API."""

from __future__ import annotations

from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.api.schemas.documents import (
    DeleteRequest,
    DeleteResponse,
    DocumentStats,
    UploadResponse,
)
from src.api.schemas.query import (
    QueryRequest,
    QueryResponse,
    SourceDetail,
)
from src.ui.config import UISettings

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class APIErrorPayload(BaseModel):
    """Error response returned by FastAPI."""

    model_config = ConfigDict(extra="ignore")

    # Domain hatalarında string, FastAPI doğrulama hatalarında liste gelir.
    detail: str | list[dict[str, Any]]
    error_type: str | None = Field(default=None)


class HealthResponse(BaseModel):
    """Response returned by the health endpoint."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(min_length=1)


class APIClientError(Exception):
    """Base error raised by the Streamlit API client."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type


class APIConnectionError(APIClientError):
    """Raised when the backend cannot be reached."""


class APIAuthenticationError(APIClientError):
    """Raised when the backend rejects the API key."""


class APIResponseError(APIClientError):
    """Raised for unsuccessful or invalid backend responses."""


class APIClient:
    """Synchronous HTTP client for the RAG API."""

    def __init__(
        self,
        settings: UISettings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        timeout = httpx.Timeout(
            settings.request_timeout_seconds,
            connect=settings.connect_timeout_seconds,
        )

        self._client = httpx.Client(
            base_url=settings.api_base_url,
            headers={
                # SecretStr yalnızca isteğin gönderildiği noktada açılır.
                "X-API-Key": settings.api_key.get_secret_value(),
                "Accept": "application/json",
            },
            timeout=timeout,
            transport=transport,
        )

    def __enter__(self) -> APIClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying connection pool."""

        self._client.close()

    def check_health(self) -> bool:
        """Return whether the backend reports a healthy state."""

        response = self._request(
            "GET",
            "/health",
            response_model=HealthResponse,
        )
        return response.status == "healthy"

    def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> QueryResponse:
        """Submit a question to the RAG backend."""

        payload = self._validated_payload(
            QueryRequest,
            question=question,
            top_k=top_k,
            source_filter=source_filter,
        )
        return self._request(
            "POST",
            "/api/v1/query",
            response_model=QueryResponse,
            json=payload,
        )

    def upload_document(
        self,
        file_name: str,
        content: bytes,
    ) -> UploadResponse:
        """Upload and index one PDF document."""

        return self._request(
            "POST",
            "/api/v1/documents/upload",
            response_model=UploadResponse,
            files={
                "file": (file_name, content, "application/pdf"),
            },
        )

    def delete_document(self, source: str) -> DeleteResponse:
        """Delete all indexed chunks belonging to a document."""

        payload = self._validated_payload(
            DeleteRequest,
            source=source,
        )
        return self._request(
            "DELETE",
            "/api/v1/documents",
            response_model=DeleteResponse,
            json=payload,
        )

    def get_document_stats(self) -> DocumentStats:
        """Return current vector index statistics."""

        return self._request(
            "GET",
            "/api/v1/documents/stats",
            response_model=DocumentStats,
        )

    @staticmethod
    def _validated_payload(
        model_type: type[BaseModel],
        **values: Any,
    ) -> dict[str, Any]:
        """Validate an outgoing request before network communication."""

        try:
            return model_type.model_validate(values).model_dump(exclude_none=True)
        except ValidationError as exc:
            raise APIResponseError(
                "Request data is invalid.",
                error_type="ClientValidationError",
            ) from exc

    def _request(
        self,
        method: str,
        path: str,
        *,
        response_model: type[ResponseModelT],
        **kwargs: Any,
    ) -> ResponseModelT:
        """Send one request and validate its successful response."""

        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            raise APIConnectionError(
                "The API could not be reached.",
                error_type="ConnectionError",
            ) from exc

        if not response.is_success:
            self._raise_api_error(response)

        try:
            return response_model.model_validate(response.json())
        except ValueError as exc:
            raise APIResponseError(
                "The API returned an invalid response.",
                status_code=response.status_code,
                error_type="InvalidResponseError",
            ) from exc

    @staticmethod
    def _raise_api_error(response: httpx.Response) -> None:
        """Convert an unsuccessful HTTP response into a client error."""

        try:
            payload = APIErrorPayload.model_validate(response.json())
        except ValueError:
            payload = None

        error_type = payload.error_type if payload else None
        detail = payload.detail if payload else None

        if response.status_code in {401, 403}:
            raise APIAuthenticationError(
                "API authentication failed.",
                status_code=response.status_code,
                error_type=error_type or "AuthenticationError",
            )

        # FastAPI validation listelerini doğrudan kullanıcıya göstermiyoruz.
        message = (
            detail
            if isinstance(detail, str)
            else f"API request failed with status {response.status_code}."
        )
        raise APIResponseError(
            message,
            status_code=response.status_code,
            error_type=error_type,
        )


__all__ = [
    "APIAuthenticationError",
    "APIClient",
    "APIClientError",
    "APIConnectionError",
    "APIErrorPayload",
    "APIResponseError",
    "DeleteResponse",
    "DocumentStats",
    "QueryResponse",
    "SourceDetail",
    "UploadResponse",
    "HealthResponse",
]
