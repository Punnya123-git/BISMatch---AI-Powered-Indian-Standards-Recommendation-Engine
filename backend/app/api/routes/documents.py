"""Document upload endpoints.

The route only translates HTTP into a service call: validation, storage,
extraction and cleaning all live in :mod:`app.services.document_service`.
"""

from __future__ import annotations

from fastapi import APIRouter, File, UploadFile, status

from app.api.dependencies import DocumentServiceDep
from app.schemas.common import ErrorResponse
from app.schemas.document import DocumentUploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a tender/specification document (PDF or text)",
    responses={
        400: {"model": ErrorResponse, "description": "Unsupported file type"},
        413: {"model": ErrorResponse, "description": "File too large"},
        422: {"model": ErrorResponse, "description": "Document could not be parsed"},
    },
)
async def upload_document(
    service: DocumentServiceDep,
    file: UploadFile = File(..., description="Procurement/tender document."),
) -> DocumentUploadResponse:
    """Store the upload, extract its text and return the cleaned result."""
    content = await file.read()
    return service.process_upload(
        filename=file.filename or "upload",
        content=content,
        content_type=file.content_type,
    )
