"""Health endpoint.

``GET /api/health`` is the contract the frontend uses to confirm that the API is
reachable, and it reports which optional AI components are configured.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import HealthServiceDep
from app.schemas.common import ErrorResponse
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health and dependency status",
    responses={503: {"model": ErrorResponse}},
)
def get_health(service: HealthServiceDep) -> HealthResponse:
    """Return service metadata and the status of each dependency."""
    return service.get_status()
