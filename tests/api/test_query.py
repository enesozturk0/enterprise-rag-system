from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import get_rag_chain
from src.api.routes.query import router
from src.config.settings import Settings, get_settings
from src.rag.chain import RAGResponse, SourceInfo

_TEST_API_KEY = "test-only-api-key-123456789"


class FakeRAGChain:
    def __init__(self) -> None:
        self.last_call: (
            tuple[
                str,
                int | None,
                dict[str, Any] | None,
            ]
            | None
        ) = None

    def query(
        self,
        question: str,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RAGResponse:
        self.last_call = (question, top_k, filters)

        return RAGResponse(
            answer="Test cevabı.",
            sources=[
                SourceInfo(
                    source="kilavuz.pdf",
                    page=3,
                    score=0.92,
                    content_preview="İlgili test içeriği.",
                )
            ],
            retrieval_scores=[0.92],
            model="test-model",
            duration_seconds=0.1,
        )


def test_query_maps_request_and_response() -> None:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None,
        api_key=_TEST_API_KEY,
    )

    fake_chain = FakeRAGChain()
    app.dependency_overrides[get_rag_chain] = lambda: fake_chain

    with TestClient(app) as client:
        unauthorized_response = client.post(
            "/api/v1/query",
            json={"question": "Motor yağı nasıl kontrol edilir?"},
        )
        response = client.post(
            "/api/v1/query",
            json={
                "question": "Motor yağı nasıl kontrol edilir?",
                "top_k": 3,
                "source_filter": "kilavuz.pdf",
            },
            headers={"X-API-Key": _TEST_API_KEY},
        )

    assert unauthorized_response.status_code == 401
    assert response.status_code == 200
    assert fake_chain.last_call == (
        "Motor yağı nasıl kontrol edilir?",
        3,
        {"source": "kilavuz.pdf"},
    )
    assert response.json() == {
        "answer": "Test cevabı.",
        "sources": [
            {
                "source": "kilavuz.pdf",
                "page": 3,
                "score": 0.92,
                "content_preview": "İlgili test içeriği.",
            }
        ],
        "model": "test-model",
        "duration_seconds": 0.1,
    }
