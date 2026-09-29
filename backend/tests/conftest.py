"""Shared test fixtures.

Environment variables are set *before* importing the application so tests write
into a temporary data directory and never touch the real ``data/`` folder.

``STANDARDS_DATASET_PATH`` is pointed at the *shipped* dataset so the API level
tests exercise real, verified records. Tests that need the "no catalogue" case
inject their own :class:`StandardsService` / loader through FastAPI dependency
overrides instead of changing the environment.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import pytest

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="sih-tests-"))
_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _BACKEND_ROOT.parent
_SHIPPED_DATASET = _PROJECT_ROOT / "data" / "standards" / "standards.json"

os.environ.update(
    {
        "ENVIRONMENT": "test",
        "DATA_DIR": str(_TMP_ROOT),
        "VECTOR_DB_PATH": str(_TMP_ROOT / "vector_store"),
        "VECTOR_STORE_PROVIDER": "chroma",
        "LLM_PROVIDER": "unconfigured",
        "EMBEDDING_PROVIDER": "unconfigured",
        "STANDARDS_DATASET_PATH": str(_SHIPPED_DATASET),
        "LOG_LEVEL": "WARNING",
    }
)

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402
from app.standards.dataset_schema import designation_year  # noqa: E402

# Make sure the (possibly already cached) settings pick up the test env vars.
get_settings.cache_clear()


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Test client with application lifespan handling enabled."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def shipped_dataset_path() -> Path:
    """Path to the dataset that ships with the repository."""
    return _SHIPPED_DATASET


@pytest.fixture
def standard_record_factory():
    """Build a *test-only* dataset record dict (never shipped catalogue data).

    ``year`` is derived from the designation unless the caller overrides it, so
    fixtures stay consistent with the validator's year/designation rule.
    """

    def _build(**overrides: object) -> dict:
        record: dict = {
            "standard_number": "IS 9999:2000",
            "title": None,
            "scope": None,
            "product_category": "rotating electrical machines",
            "status": None,
            "revision": None,
            "year": 2000,
            "amendments": [],
            "related_standards": [],
            "references": [],
            "certification": None,
            "source": {
                "organization": "Bureau of Indian Standards",
                "url": None,
                "document": None,
            },
        }
        record.update(overrides)
        if "year" not in overrides:
            record["year"] = designation_year(record["standard_number"])
        return record

    return _build


@pytest.fixture
def dataset_payload_factory(standard_record_factory):
    """Build a dataset envelope (metadata + records) for validator tests."""

    def _build(*records: dict, **metadata: object) -> dict:
        entries = list(records) or [standard_record_factory()]
        meta: dict = {
            "name": "test-only dataset",
            "version": "9.9.9",
            "record_count": len(entries),
        }
        meta.update(metadata)
        return {"schema_version": "1.0", "dataset": meta, "standards": entries}

    return _build


@pytest.fixture(scope="session")
def sample_text_bytes() -> bytes:
    """A small, clearly synthetic procurement requirement used by tests."""
    return (
        "Supply of ordinary Portland cement, 43 grade, conforming to the "
        "relevant Indian Standard. Packing shall be in moisture-proof bags of "
        "50 kg net weight with clear marking of the grade and batch number."
    ).encode("utf-8")


@pytest.fixture(scope="session")
def sample_pdf_bytes() -> bytes:
    """A one-page PDF containing a single line of text, generated with pypdf."""
    from pypdf import PdfWriter
    from pypdf.generic import (
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
    )

    writer = PdfWriter()
    page = writer.add_blank_page(width=595, height=842)

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)  # noqa: SLF001 - only public-ish option
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )

    content = DecodedStreamObject()
    content.set_data(
        b"BT /F1 12 Tf 72 760 Td (Cement shall conform to the applicable Indian Standard.) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(content)  # noqa: SLF001

    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
