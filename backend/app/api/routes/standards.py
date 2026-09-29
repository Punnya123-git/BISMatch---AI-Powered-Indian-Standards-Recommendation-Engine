"""Standards catalogue endpoint.

``GET /api/standards`` exposes the loaded catalogue as safe metadata: how many
standards are available, the dataset version and (a page of) the records
themselves. Filesystem paths and other operational details are never exposed.

The endpoint reports what the dataset actually holds - records without a
verified title or scope simply have ``null`` there, they are never filled in.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.dependencies import StandardsServiceDep
from app.schemas.common import ErrorResponse
from app.schemas.standard import StandardsCatalogResponse

router = APIRouter(tags=["standards"])


@router.get(
    "/standards",
    response_model=StandardsCatalogResponse,
    summary="Loaded Indian Standards catalogue",
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def list_standards(
    service: StandardsServiceDep,
    limit: int = Query(50, ge=1, le=200, description="Maximum records to return."),
    offset: int = Query(0, ge=0, description="Records to skip."),
    q: str | None = Query(
        None,
        max_length=200,
        description="Optional text query (designation, or wording present in the dataset).",
    ),
) -> StandardsCatalogResponse:
    """Return catalogue metadata plus a page of standard records.

    Fails with ``standards_dataset_unavailable`` / ``standards_dataset_invalid``
    when no usable dataset is configured - an empty catalogue is never disguised
    as a successful lookup.
    """
    return service.catalog(limit=limit, offset=offset, query=q)
