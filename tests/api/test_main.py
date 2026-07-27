from fastapi.testclient import TestClient

from src.api.main import create_app
from src.config.settings import get_settings
from src.rag.exceptions import RetrievalError


def test_create_app_wires_routes_metrics_handlers_and_cors(monkeypatch) -> None:
    allowed_origin = "http://test-ui.local"
    monkeypatch.setenv("API_CORS_ORIGINS", f'["{allowed_origin}"]')
    get_settings.cache_clear()

    try:
        application = create_app()

        with TestClient(application) as client:
            health_response = client.get("/health")
            metrics_response = client.get("/metrics")
            openapi_response = client.get("/openapi.json")

            allowed_preflight = client.options(
                "/api/v1/query",
                headers={
                    "Origin": allowed_origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "X-API-Key, Content-Type",
                },
            )
            blocked_preflight = client.options(
                "/api/v1/query",
                headers={
                    "Origin": "https://untrusted.example",
                    "Access-Control-Request-Method": "POST",
                },
            )

        paths = openapi_response.json()["paths"]

        assert health_response.status_code == 200
        assert metrics_response.status_code == 200
        assert "/api/v1/query" in paths
        assert "/api/v1/documents/upload" in paths
        assert "/api/v1/documents" in paths
        assert "/api/v1/documents/stats" in paths

        assert RetrievalError in application.exception_handlers
        assert (
            allowed_preflight.headers["access-control-allow-origin"]
            == allowed_origin
        )
        assert "access-control-allow-origin" not in blocked_preflight.headers
    finally:
        get_settings.cache_clear()