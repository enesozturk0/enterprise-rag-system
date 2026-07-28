import httpx
import pytest

from src.ui.api_client import (
    APIAuthenticationError,
    APIClient,
    APIClientError,
    APIConnectionError,
    APIErrorPayload,
    APIResponseError,
    QueryResponse,
)
from src.ui.config import UISettings


def test_query_response_uses_backend_contract() -> None:
    response = QueryResponse.model_validate(
        {
            "answer": "Motor yağı düz zeminde kontrol edilir.",
            "sources": [
                {
                    "source": "manual.pdf",
                    "page": 263,
                    "score": 0.84,
                    "content_preview": "Motor yağının kontrol edilmesi...",
                }
            ],
            "model": "llama3.1:8b",
            "duration_seconds": 1.25,
        }
    )

    assert response.sources[0].page == 263
    assert response.sources[0].score == 0.84


def test_error_payload_accepts_fastapi_validation_details() -> None:
    payload = APIErrorPayload.model_validate(
        {
            "detail": [
                {
                    "type": "missing",
                    "loc": ["body", "question"],
                    "msg": "Field required",
                    "input": {},
                }
            ]
        }
    )

    assert isinstance(payload.detail, list)
    assert payload.error_type is None


def test_client_errors_share_status_metadata() -> None:
    error = APIAuthenticationError(
        "Authentication failed.",
        status_code=401,
        error_type="AuthenticationError",
    )

    assert isinstance(error, APIClientError)
    assert error.status_code == 401
    assert error.error_type == "AuthenticationError"
    assert str(error) == "Authentication failed."


def test_response_error_can_represent_server_failure() -> None:
    error = APIResponseError(
        "Document retrieval service is unavailable.",
        status_code=502,
        error_type="RetrievalError",
    )

    assert error.status_code == 502
    assert error.error_type == "RetrievalError"


def _settings() -> UISettings:
    return UISettings(
        _env_file=None,
        api_url="http://testserver",
        api_key="test-ui-api-key-123456",
    )


def test_query_sends_auth_and_returns_typed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/query"
        assert request.headers["X-API-Key"] == "test-ui-api-key-123456"
        assert request.read().decode() == (
            '{"question":"Motor yağı nasıl kontrol edilir?","top_k":3}'
        )
        return httpx.Response(
            200,
            json={
                "answer": "Motor yağı düz zeminde kontrol edilir.",
                "sources": [],
                "model": "llama3.1:8b",
                "duration_seconds": 1.0,
            },
        )

    transport = httpx.MockTransport(handler)

    with APIClient(_settings(), transport=transport) as client:
        result = client.query(
            "Motor yağı nasıl kontrol edilir?",
            top_k=3,
        )

    assert isinstance(result, QueryResponse)
    assert result.model == "llama3.1:8b"


def test_authentication_failure_is_mapped() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            401,
            json={"detail": "Invalid API key."},
        )
    )

    with APIClient(_settings(), transport=transport) as client:  # noqa: SIM117
        with pytest.raises(APIAuthenticationError) as captured:
            client.get_document_stats()

    assert captured.value.status_code == 401


def test_connection_failure_is_mapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused.", request=request)

    with (
        APIClient(
            _settings(),
            transport=httpx.MockTransport(handler),
        ) as client,
        pytest.raises(APIConnectionError),
    ):
        client.check_health()


def test_document_operations_use_expected_endpoints() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/upload"):
            assert request.method == "POST"
            assert b"manual.pdf" in request.read()
            return httpx.Response(
                201,
                json={
                    "file_name": "manual.pdf",
                    "total_chunks": 2,
                    "indexed_chunks": 2,
                    "failed_chunks": 0,
                    "successful_files": 1,
                    "failed_files": 0,
                    "processing_time_seconds": 0.5,
                    "message": "Document uploaded.",
                },
            )

        if request.url.path.endswith("/stats"):
            return httpx.Response(200, json={"total_chunks": 2})

        assert request.method == "DELETE"
        assert request.read() == b'{"source":"manual.pdf"}'
        return httpx.Response(
            200,
            json={
                "source": "manual.pdf",
                "chunks_deleted": 2,
                "message": "Document deleted.",
            },
        )

    with APIClient(
        _settings(),
        transport=httpx.MockTransport(handler),
    ) as client:
        upload = client.upload_document("manual.pdf", b"%PDF-test")
        stats = client.get_document_stats()
        deletion = client.delete_document("manual.pdf")

    assert upload.indexed_chunks == 2
    assert stats.total_chunks == 2
    assert deletion.chunks_deleted == 2


def test_health_returns_false_for_non_healthy_status() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"status": "degraded"},
        )
    )

    with APIClient(_settings(), transport=transport) as client:
        assert client.check_health() is False


def test_invalid_outgoing_payload_is_rejected_before_request() -> None:
    def unexpected_request(request: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid payload must not reach the network.")

    with (
        APIClient(
            _settings(),
            transport=httpx.MockTransport(unexpected_request),
        ) as client,
        pytest.raises(APIResponseError) as captured,
    ):
        client.query("   ")

    assert captured.value.status_code is None
    assert captured.value.error_type == "ClientValidationError"


def test_invalid_success_response_is_mapped() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"total_chunks": -1},
        )
    )

    with (
        APIClient(
            _settings(),
            transport=transport,
        ) as client,
        pytest.raises(APIResponseError) as captured,
    ):
        client.get_document_stats()

    assert captured.value.status_code == 200
    assert captured.value.error_type == "InvalidResponseError"


@pytest.mark.parametrize(
    ("response", "expected_message", "expected_error_type"),
    [
        (
            httpx.Response(503, text="not-json"),
            "API request failed with status 503.",
            None,
        ),
        (
            httpx.Response(
                502,
                json={
                    "detail": "Document retrieval service is unavailable.",
                    "error_type": "RetrievalError",
                },
            ),
            "Document retrieval service is unavailable.",
            "RetrievalError",
        ),
        (
            httpx.Response(
                422,
                json={
                    "detail": [
                        {
                            "type": "missing",
                            "loc": ["body", "question"],
                            "msg": "Field required",
                            "input": {},
                        }
                    ]
                },
            ),
            "API request failed with status 422.",
            None,
        ),
    ],
)
def test_non_success_responses_are_safely_mapped(
    response: httpx.Response,
    expected_message: str,
    expected_error_type: str | None,
) -> None:
    transport = httpx.MockTransport(lambda request: response)

    with (
        APIClient(
            _settings(),
            transport=transport,
        ) as client,
        pytest.raises(APIResponseError) as captured,
    ):
        client.get_document_stats()

    assert str(captured.value) == expected_message
    assert captured.value.status_code == response.status_code
    assert captured.value.error_type == expected_error_type
