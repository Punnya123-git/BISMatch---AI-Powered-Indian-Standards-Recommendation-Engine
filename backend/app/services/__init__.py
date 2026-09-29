"""Re-export the service classes for convenient imports."""

from app.services.document_service import DocumentService, get_document_service
from app.services.health_service import HealthService
from app.services.recommendation_service import (
    RecommendationService,
    get_recommendation_service,
)
from app.services.standards_service import StandardsService, get_standards_service

__all__ = [
    "DocumentService",
    "HealthService",
    "RecommendationService",
    "StandardsService",
    "get_document_service",
    "get_recommendation_service",
    "get_standards_service",
]
