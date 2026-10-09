"""System health endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from ..config import settings
from ..schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check. Returns safe, non-secret service information."""
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.version,
        environment=settings.environment,
    )
