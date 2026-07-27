from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.health import router

app = FastAPI()
app.include_router(router)
client = TestClient(app)


def test_health_returns_200_without_authentication() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}