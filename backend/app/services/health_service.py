"""Health reporting.

The endpoint must keep working even when the AI stack is not configured, so
every check here is defensive: failures become ``configured=False`` entries
instead of raised exceptions.
"""

from __future__ import annotations

from app.ai.llm.factory import get_llm_provider
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.database.connection import get_database_manager
from app.rag.pipeline import (
    PipelineReadiness,
    RecommendationPipeline,
    get_recommendation_pipeline,
)
from app.schemas.health import DependencyStatus, HealthResponse
from app.services.standards_service import (
    DatasetStatus,
    StandardsService,
    get_standards_service,
)

logger = get_logger(__name__)


class HealthService:
    """Builds the payload returned by ``GET /api/health``."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        pipeline: RecommendationPipeline | None = None,
        standards_service: StandardsService | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._pipeline = pipeline or get_recommendation_pipeline()
        self._standards = standards_service or get_standards_service()

    def _dependencies(
        self,
        *,
        readiness: PipelineReadiness | None = None,
        dataset: DatasetStatus | None = None,
    ) -> list[DependencyStatus]:
        """Report the state of every external/optional component."""
        dependencies: list[DependencyStatus] = []

        llm_provider = get_llm_provider()
        dependencies.append(
            DependencyStatus(
                name="llm_provider",
                configured=llm_provider.is_configured,
                detail=llm_provider.describe(),
            )
        )

        readiness = readiness or self._pipeline.readiness()
        dependencies.append(
            DependencyStatus(
                name="embedding_provider",
                configured=self._pipeline.embedding_provider.is_configured,
                detail=self._pipeline.embedding_provider.describe(),
            )
        )
        dependencies.append(
            DependencyStatus(
                name="vector_store",
                configured=readiness.vector_store != "unavailable",
                detail=f"{readiness.vector_store} ({readiness.indexed_chunks} records)",
            )
        )

        dataset = dataset or self._standards.dataset_status()
        dependencies.append(
            DependencyStatus(
                name="standards_dataset",
                configured=dataset.loaded,
                detail=dataset.message,
            )
        )

        database = get_database_manager()
        dependencies.append(
            DependencyStatus(
                name="sql_database",
                configured=database.is_configured,
                detail=database.describe,
            )
        )
        return dependencies

    def get_status(self) -> HealthResponse:
        """Return service metadata plus the status of each dependency."""
        readiness = self._pipeline.readiness()
        dataset = self._standards.dataset_status()
        return HealthResponse(
            status="ok",
            service=self._settings.app_name,
            version=self._settings.app_version,
            environment=self._settings.environment,
            dependencies=self._dependencies(readiness=readiness, dataset=dataset),
            standards_available=self._standards.repository.count(),
            indexed_chunks=readiness.indexed_chunks,
            dataset=dataset.to_response(),
        )


def get_health_service() -> HealthService:
    """Return a health service bound to the current configuration."""
    return HealthService()
