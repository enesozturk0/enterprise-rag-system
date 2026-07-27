from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api.middleware.auth import verify_api_key
from src.config.settings import Settings, get_settings

_TEST_API_KEY = "test-only-api-key-123456789"


def test_api_key_dependency_accepts_only_the_configured_key() -> None:
    app = FastAPI()
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        api_key=_TEST_API_KEY,
    )

    @app.get("/protected", dependencies=[Depends(verify_api_key)])
    def protected_endpoint() -> dict[str, str]:
        return {"status": "allowed"}

    with TestClient(app) as client:
        missing_response = client.get("/protected")
        invalid_response = client.get(
            "/protected",
            headers={"X-API-Key": "wrong-key"},
        )
        valid_response = client.get(
            "/protected",
            headers={"X-API-Key": _TEST_API_KEY},
        )

    assert missing_response.status_code == 401
    assert invalid_response.status_code == 401
    assert valid_response.status_code == 200
    assert valid_response.json() == {"status": "allowed"}