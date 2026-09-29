"""Document processing: turning uploaded files into clean, page-aware text.

This package must stay independent from the AI/RAG packages: it only knows about
files and text, never about embeddings, LLMs or vector stores.
"""

from app.document_processing.entities import ExtractedDocument, ExtractedPage
from app.document_processing.factory import get_extractor
from app.document_processing.text_cleaner import clean_pages, clean_text

__all__ = [
    "ExtractedDocument",
    "ExtractedPage",
    "clean_pages",
    "clean_text",
    "get_extractor",
]
