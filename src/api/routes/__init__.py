"""Public API route exports."""

from src.api.routes.documents import router as documents_router
from src.api.routes.health import router as health_router
from src.api.routes.query import router as query_router

__all__ = [
    "documents_router",
    "health_router",
    "query_router",
]
