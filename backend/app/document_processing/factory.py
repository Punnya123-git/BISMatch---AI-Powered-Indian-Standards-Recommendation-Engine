"""Selects the right extractor for an uploaded file."""

from __future__ import annotations

from pathlib import Path

from app.core.exceptions import UnsupportedFileTypeError
from app.document_processing.base import DocumentExtractor
from app.document_processing.pdf_extractor import PdfExtractor
from app.document_processing.plain_text_extractor import PlainTextExtractor

_EXTRACTORS: tuple[DocumentExtractor, ...] = (PdfExtractor(), PlainTextExtractor())


def supported_extensions() -> tuple[str, ...]:
    """All extensions the document pipeline can currently handle."""
    extensions: list[str] = []
    for extractor in _EXTRACTORS:
        extensions.extend(extractor.supported_extensions)
    return tuple(extensions)


def get_extractor(path: Path) -> DocumentExtractor:
    """Return the extractor registered for ``path``'s extension."""
    suffix = path.suffix.lower()
    for extractor in _EXTRACTORS:
        if suffix in extractor.supported_extensions:
            return extractor
    raise UnsupportedFileTypeError(
        f"Unsupported file type '{suffix or path.name}'. "
        f"Supported types: {', '.join(supported_extensions())}."
    )
