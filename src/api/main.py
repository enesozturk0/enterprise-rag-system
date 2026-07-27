"""FastAPI application factory and ASGI entry point."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.middleware import register_exception_handlers
from src.api.routes import documents_router, health_router, query_router
from src.config.settings import get_settings


def create_app() -> FastAPI:
    """Build and configure a new FastAPI application."""

    settings = get_settings()

    application = FastAPI(
        title="Enterprise RAG API",
        description="Retrieval-Augmented Generation over indexed PDF documents.",
        version="0.1.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    register_exception_handlers(application)

    application.include_router(health_router)
    application.include_router(query_router)
    application.include_router(documents_router)

    Instrumentator(
        excluded_handlers=[r"^/health$", r"^/metrics$"],
    ).instrument(application).expose(
        application,
        include_in_schema=False,
        should_gzip=True,
    )

    return application


app = create_app()
