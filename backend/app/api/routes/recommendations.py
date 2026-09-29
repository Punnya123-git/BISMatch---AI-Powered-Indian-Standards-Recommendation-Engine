"""Recommendation endpoints.

``POST /api/recommendations/analyze`` is fully wired end to end. The deterministic
stage retrieves and verifies candidate standards; the reasoning layer turns the
verified candidates plus their evidence into the final applicability
classification and ranking. Until a real Indian Standards dataset exists, the
response explicitly reports that state instead of returning invented standards.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import RecommendationServiceDep
from app.schemas.common import ErrorResponse
from app.schemas.recommendation import RecommendationRequest, RecommendationResponse

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post(
    "/analyze",
    response_model=RecommendationResponse,
    summary="Analyse a procurement requirement and recommend applicable standards",
    responses={503: {"model": ErrorResponse, "description": "Engine not ready"}},
)
def analyze_requirement(
    payload: RecommendationRequest,
    service: RecommendationServiceDep,
) -> RecommendationResponse:
    """Return recommendations, evidence and the current pipeline status."""
    return service.analyze(payload)
