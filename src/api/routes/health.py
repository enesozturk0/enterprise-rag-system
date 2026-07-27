"""Application liveness endpoint."""

from __future__ import annotations

from fastapi import APIRouter, status

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=dict[str, str],
    status_code=status.HTTP_200_OK,
)
async def health_check() -> dict[str, str]:
    """Report whether the API process is running."""

    return {"status": "healthy"}
