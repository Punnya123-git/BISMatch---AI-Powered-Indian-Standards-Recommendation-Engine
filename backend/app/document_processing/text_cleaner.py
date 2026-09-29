"""Text normalisation utilities.

Tender documents are messy: PDF extraction introduces hard line breaks,
duplicated whitespace, non-breaking spaces and ligature artefacts. Cleaning is
kept deterministic and dependency-free so the same input always produces the
same text (important for reproducible retrieval).
"""

from __future__ import annotations

import re
import unicodedata

from app.document_processing.entities import ExtractedPage

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HORIZONTAL_WHITESPACE = re.compile(r"[ \t\f\v]+")
# Soft hyphen used by PDF generators when splitting words across lines.
_SOFT_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?%)\]])")


def normalize_whitespace(text: str) -> str:
    """Collapse redundant whitespace while preserving paragraph breaks."""
    lines = [_HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in text.splitlines()]
    collapsed = "\n".join(lines)
    return _EXCESS_BLANK_LINES.sub("\n\n", collapsed).strip()


def clean_text(text: str) -> str:
    """Return a normalised version of ``text``."""
    if not text:
        return ""

    normalised = unicodedata.normalize("NFKC", text)
    normalised = normalised.replace("\xa0", " ").replace("\u200b", "")
    normalised = _CONTROL_CHARS.sub("", normalised)
    normalised = _SOFT_HYPHEN_BREAK.sub(r"\1\2", normalised)
    normalised = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", normalised)
    return normalize_whitespace(normalised)


def clean_pages(pages: list[ExtractedPage]) -> list[ExtractedPage]:
    """Clean every page, dropping pages that end up empty."""
    cleaned: list[ExtractedPage] = []
    for page in pages:
        text = clean_text(page.text)
        if text:
            cleaned.append(ExtractedPage(page_number=page.page_number, text=text))
    return cleaned
