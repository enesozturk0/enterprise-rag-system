"""API-key authentication dependency."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.config.settings import Settings, get_settings

_api_key_header = APIKeyHeader(
    name="X-API-Key",
    scheme_name="ApiKeyAuth",
    description="API key required for protected endpoints.",
    auto_error=False,
)


def verify_api_key(
    api_key: Annotated[str | None, Security(_api_key_header)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Reject requests whose API key is missing or invalid."""

    expected_key = settings.api_key.get_secret_value()

    # compare_digest avoids content-dependent comparison timing.
    if api_key is None or not secrets.compare_digest(api_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )