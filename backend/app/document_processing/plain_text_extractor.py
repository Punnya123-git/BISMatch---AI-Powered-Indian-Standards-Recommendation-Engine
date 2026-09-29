"""Extractor for already machine-readable text files (``.txt``, ``.md``)."""

from __future__ import annotations

from pathlib import Path

from app.core.exceptions import DocumentProcessingError
from app.document_processing.base import DocumentExtractor
from app.document_processing.entities import ExtractedPage


class PlainTextExtractor(DocumentExtractor):
    """Reads the whole file as a single logical page."""

    supported_extensions = (".txt", ".md")

    def extract(self, path: Path) -> list[ExtractedPage]:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise DocumentProcessingError(
                f"Could not read text file '{path.name}': {exc}"
            ) from exc
        return [ExtractedPage(page_number=1, text=text)]
