"""FastAPI dependencies.

Routes depend on these typed aliases instead of importing services directly, so
services can be swapped in tests with ``app.dependency_overrides``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.services.document_service import DocumentService, get_document_service
from app.services.health_service import HealthService, get_health_service
from app.services.recommendation_service import (
    RecommendationService,
    get_recommendation_service,
)
from app.services.standards_service import StandardsService, get_standards_service

DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
HealthServiceDep = Annotated[HealthService, Depends(get_health_service)]
RecommendationServiceDep = Annotated[
    RecommendationService, Depends(get_recommendation_service)
]
StandardsServiceDep = Annotated[StandardsService, Depends(get_standards_service)]

__all__ = [
    "DocumentServiceDep",
    "HealthServiceDep",
    "RecommendationServiceDep",
    "StandardsServiceDep",
]
