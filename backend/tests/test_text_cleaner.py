"""Unit tests for the text cleaner."""

from __future__ import annotations

from app.document_processing.entities import ExtractedPage
from app.document_processing.text_cleaner import clean_pages, clean_text


def test_clean_text_collapses_whitespace() -> None:
    assert clean_text("  Supply   of\tcement  ") == "Supply of cement"


def test_clean_text_removes_soft_hyphen_line_breaks() -> None:
    assert clean_text("conform-\ning to") == "conforming to"


def test_clean_text_handles_non_breaking_spaces() -> None:
    assert clean_text("50\u00a0kg") == "50 kg"


def test_clean_text_collapses_excess_blank_lines() -> None:
    assert clean_text("A\n\n\n\nB") == "A\n\nB"


def test_clean_pages_drops_empty_pages() -> None:
    pages = [
        ExtractedPage(page_number=1, text="Content"),
        ExtractedPage(page_number=2, text="   \n  "),
        ExtractedPage(page_number=3, text="More content"),
    ]
    cleaned = clean_pages(pages)

    assert [page.page_number for page in cleaned] == [1, 3]
