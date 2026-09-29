"""Document service: validate, store, extract and clean uploads.

Business logic lives here so the API route only deals with HTTP concerns
(multipart parsing and status codes).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    DocumentProcessingError,
    FileTooLargeError,
    NotFoundError,
    UnsupportedFileTypeError,
)
from app.core.logging import get_logger
from app.document_processing import clean_pages, get_extractor
from app.document_processing.entities import ExtractedDocument
from app.document_processing.factory import supported_extensions
from app.schemas.document import DocumentMetadata, DocumentUploadResponse
from app.schemas.document import ExtractedDocument as ExtractedDocumentSchema
from app.schemas.document import ExtractedPage as ExtractedPageSchema

logger = get_logger(__name__)
_BYTES_PER_MB = 1024 * 1024


class DocumentService:
    """Handles the lifecycle of an uploaded procurement document."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def settings(self) -> Settings:
        return self._settings

    # --- validation -------------------------------------------------------
    def _validate(self, filename: str, content: bytes) -> str:
        suffix = Path(filename).suffix.lower()
        allowed = [ext.lower() for ext in self._settings.allowed_upload_extensions]
        if suffix not in allowed:
            raise UnsupportedFileTypeError(
                f"'{suffix or filename}' is not an accepted file type. "
                f"Accepted types: {', '.join(supported_extensions())}."
            )
        if not content:
            raise DocumentProcessingError("The uploaded file is empty.")
        limit = self._settings.max_upload_size_mb * _BYTES_PER_MB
        if len(content) > limit:
            raise FileTooLargeError(
                f"File is {len(content) / _BYTES_PER_MB:.1f} MB; the limit is "
                f"{self._settings.max_upload_size_mb} MB."
            )
        return suffix

    # --- storage ----------------------------------------------------------
    def _store_raw(self, document_id: str, suffix: str, content: bytes) -> Path:
        path = self._settings.raw_data_dir / f"{document_id}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def _store_processed(
        self, document_id: str, document: ExtractedDocument, metadata: DocumentMetadata
    ) -> None:
        directory = self._settings.processed_data_dir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{document_id}.txt").write_text(document.text, encoding="utf-8")
        (directory / f"{document_id}.json").write_text(
            json.dumps(
                {
                    "metadata": metadata.model_dump(mode="json"),
                    "page_count": document.page_count,
                    "character_count": document.character_count,
                    "word_count": document.word_count,
                    "warnings": document.warnings,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # --- extraction -------------------------------------------------------
    def extract(self, path: Path, *, document_id: str) -> ExtractedDocument:
        """Extract and clean the text of a previously stored file."""
        extractor = get_extractor(path)
        raw_pages = extractor.extract(path)
        cleaned_pages = clean_pages(raw_pages)
        document = ExtractedDocument(filename=path.name, pages=cleaned_pages)

        if len(cleaned_pages) < len(raw_pages):
            document.warnings.append(
                f"{len(raw_pages) - len(cleaned_pages)} page(s) produced no "
                "extractable text (possibly scanned images)."
            )
        if document.is_empty() or document.word_count < 5:
            document.warnings.append(
                "Almost no text could be extracted. If this is a scanned PDF, OCR "
                "support is required before the document can be analysed."
            )
        logger.info(
            "Extracted document %s: %d page(s), %d word(s)",
            document_id,
            document.page_count,
            document.word_count,
        )
        return document

    def process_upload(
        self,
        *,
        filename: str,
        content: bytes,
        content_type: str | None = None,
    ) -> DocumentUploadResponse:
        """Full pipeline: validate -> store -> extract -> clean -> persist."""
        suffix = self._validate(filename, content)
        document_id = uuid.uuid4().hex
        stored_path = self._store_raw(document_id, suffix, content)

        document = self.extract(stored_path, document_id=document_id)
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            content_type=content_type,
            size_bytes=len(content),
            uploaded_at=datetime.now(timezone.utc),
            stored_path=str(stored_path),
        )
        self._store_processed(document_id, document, metadata)

        return DocumentUploadResponse(
            metadata=metadata,
            document=ExtractedDocumentSchema(
                document_id=document_id,
                filename=filename,
                page_count=document.page_count,
                character_count=document.character_count,
                word_count=document.word_count,
                text=document.text,
                pages=[
                    ExtractedPageSchema(
                        page_number=page.page_number,
                        text=page.text,
                        character_count=page.character_count,
                    )
                    for page in document.pages
                ],
                warnings=document.warnings,
            ),
        )

    # --- retrieval of processed text --------------------------------------
    def get_processed_text(self, document_id: str) -> str:
        """Return the cleaned text of an earlier upload."""
        path = self._settings.processed_data_dir / f"{document_id}.txt"
        if not path.exists():
            raise NotFoundError(
                f"No processed document found for id '{document_id}'. Upload the "
                "document first via /api/documents/upload."
            )
        return path.read_text(encoding="utf-8")


def get_document_service() -> DocumentService:
    """Return a document service bound to the current settings."""
    return DocumentService()
