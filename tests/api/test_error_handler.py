import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.api.middleware.error_handler import register_exception_handlers
from src.ingestion.exceptions import PdfReadError
from src.rag.exceptions import ChainError, GenerationError, RetrievalError
from src.vectorstore.exceptions import StoreConnectionError


@pytest.mark.parametrize(
    ("exception", "expected_status", "expected_type"),
    [
        (
            RetrievalError("sensitive retrieval detail"),
            502,
            "RetrievalError",
        ),
        (
            GenerationError("sensitive generation detail"),
            502,
            "GenerationError",
        ),
        (
            ChainError("sensitive chain detail"),
            500,
            "ChainError",
        ),
        (
            PdfReadError("sensitive PDF detail"),
            422,
            "PdfReadError",
        ),
        (
            StoreConnectionError("sensitive store detail"),
            503,
            "StoreConnectionError",
        ),
        (
            RuntimeError("sensitive internal detail"),
            500,
            "InternalServerError",
        ),
    ],
)
def test_domain_errors_are_mapped_without_leaking_details(
    exception: Exception,
    expected_status: int,
    expected_type: str,
) -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/failure")
    def failure_endpoint() -> None:
        raise exception

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/failure")

    assert response.status_code == expected_status
    assert response.json()["error_type"] == expected_type
    assert "sensitive" not in response.text


def test_http_exception_keeps_fastapi_default_handler() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/missing")
    def missing_endpoint() -> None:
        raise HTTPException(status_code=404, detail="Resource not found.")

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/missing")

    assert response.status_code == 404
    assert response.json() == {"detail": "Resource not found."}