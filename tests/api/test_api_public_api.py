from fastapi import APIRouter

from src.api.middleware import register_exception_handlers, verify_api_key
from src.api.routes import documents_router, health_router, query_router


def test_api_packages_expose_expected_public_components() -> None:
    assert isinstance(documents_router, APIRouter)
    assert isinstance(health_router, APIRouter)
    assert isinstance(query_router, APIRouter)
    assert callable(register_exception_handlers)
    assert callable(verify_api_key)
