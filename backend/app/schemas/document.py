"""Schemas describing uploaded procurement / tender documents."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ExtractedPage(BaseModel):
    """Text extracted from a single page of a source document."""

    page_number: int = Field(ge=1)
    text: str
    character_count: int = Field(ge=0)


class DocumentMetadata(BaseModel):
    """Metadata of a stored upload."""

    document_id: str
    filename: str
    content_type: str | None = None
    size_bytes: int = Field(ge=0)
    uploaded_at: datetime
    stored_path: str


class ExtractedDocument(BaseModel):
    """Result of the document processing pipeline for one upload."""

    document_id: str
    filename: str
    page_count: int = Field(ge=0)
    character_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    text: str = Field(description="Cleaned full text.")
    pages: list[ExtractedPage] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DocumentUploadResponse(BaseModel):
    """Payload returned by ``POST /api/documents/upload``."""

    metadata: DocumentMetadata
    document: ExtractedDocument
