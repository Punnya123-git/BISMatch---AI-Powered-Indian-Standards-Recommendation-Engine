"""Pydantic schemas: the API/domain data contracts.

Grouping the exports here keeps imports short in services and routes
(``from app.schemas import StandardRecord``).
"""

from app.schemas.common import ErrorResponse, MessageResponse
from app.schemas.document import (
    DocumentMetadata,
    DocumentUploadResponse,
    ExtractedDocument,
    ExtractedPage,
)
from app.schemas.health import DependencyStatus, HealthResponse
from app.schemas.recommendation import (
    EvidenceItem,
    PipelineInfo,
    RecommendationRequest,
    RecommendationResponse,
    RecommendationStatus,
    RecommendedStandard,
)
from app.schemas.standard import (
    CertificationRequirement,
    DatasetStatusResponse,
    StandardMatch,
    StandardRecord,
    StandardReference,
    StandardSource,
    StandardsCatalogResponse,
    StandardVersion,
)

__all__ = [
    "CertificationRequirement",
    "DatasetStatusResponse",
    "DependencyStatus",
    "DocumentMetadata",
    "DocumentUploadResponse",
    "ErrorResponse",
    "EvidenceItem",
    "ExtractedDocument",
    "ExtractedPage",
    "HealthResponse",
    "MessageResponse",
    "RecommendationRequest",
    "RecommendationResponse",
    "RecommendationStatus",
    "RecommendedStandard",
    "StandardMatch",
    "StandardRecord",
    "StandardReference",
    "StandardSource",
    "StandardVersion",
    "StandardsCatalogResponse",
]
