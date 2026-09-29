"""PDF text extraction.

Backed by :mod:`pypdf` (pure Python, no external services). Scanned/image-only
tenders therefore produce empty pages and are reported as warnings instead of
silently yielding nothing useful.
"""

from __future__ import annotations

from pathlib import Path

from app.core.exceptions import DocumentProcessingError
from app.core.logging import get_logger
from app.document_processing.base import DocumentExtractor
from app.document_processing.entities import ExtractedPage

logger = get_logger(__name__)


class PdfExtractor(DocumentExtractor):
    """Page-by-page text extraction for ``.pdf`` files."""

    supported_extensions = (".pdf",)

    def extract(self, path: Path) -> list[ExtractedPage]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise DocumentProcessingError(
                "PDF support requires the 'pypdf' package. Install backend requirements."
            ) from exc

        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted:
                # Try the empty-password decryption many tenders use.
                reader.decrypt("")
            pages: list[ExtractedPage] = []
            for index, page in enumerate(reader.pages, start=1):
                pages.append(
                    ExtractedPage(page_number=index, text=page.extract_text() or "")
                )
        except Exception as exc:  # noqa: BLE001 - surfaced as a domain error
            logger.exception("Failed to read PDF %s", path)
            raise DocumentProcessingError(f"Could not read PDF '{path.name}': {exc}") from exc

        if not pages:
            raise DocumentProcessingError(f"PDF '{path.name}' contains no pages.")
        return pages
