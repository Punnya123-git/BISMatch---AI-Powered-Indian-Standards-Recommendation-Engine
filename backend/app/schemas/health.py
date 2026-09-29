"""Health check schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.standard import DatasetStatusResponse


class DependencyStatus(BaseModel):
    """Status of a single downstream dependency."""

    name: str
    configured: bool = Field(
        description="Whether the dependency is configured (e.g. credentials present)."
    )
    detail: str | None = None


class HealthResponse(BaseModel):
    """Payload returned by ``GET /api/health``."""

    status: str = "ok"
    service: str
    version: str
    environment: str
    dependencies: list[DependencyStatus] = Field(default_factory=list)
    # --- standards catalogue / index ---------------------------------------
    standards_available: int = Field(
        default=0,
        description="Indian Standards records currently loaded in the catalogue.",
    )
    indexed_chunks: int = Field(
        default=0, description="Chunks currently stored in the vector store."
    )
    dataset: DatasetStatusResponse | None = Field(
        default=None, description="Availability and size of the standards dataset."
    )
