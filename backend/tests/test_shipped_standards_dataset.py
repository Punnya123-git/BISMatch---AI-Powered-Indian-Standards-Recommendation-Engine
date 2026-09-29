"""Guards for the dataset that ships with the repository.

These tests are the reason the prototype subset stays honest: they fail if a
record gains information that was never verified in the source material.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.standards.dataset_schema import (
    designation_year,
    validate_standards_dataset,
)

#: The designations supplied by the source material (rotating electrical machines).
EXPECTED_DESIGNATIONS = {
    "IS/IEC 60034-1:2022",
    "IS 12615:2018",
    "IS 996:2009",
    "IS 9283:2024",
    "IS 7538:1996",
    "IS 14582:2021",
    "IS 14578:1999",
    "IS 8151:2024",
    "IS 18073:2023",
    "IS/IEC 60034-2-1:2024",
    "IS 4029:2010",
    "IS 9320:2025",
    "IS 7572:1974",
    "IS 12075:2024",
    "IS 12065:2025",
    "IS 1231:2019",
    "IS 2223:1983",
    "IS 6362:1995",
    "IS/IEC 60034-5:2020",
    "IS 13529:2021",
}


def _load(shipped_dataset_path: Path):
    payload = json.loads(shipped_dataset_path.read_text(encoding="utf-8"))
    return validate_standards_dataset(payload, path=str(shipped_dataset_path))


def test_shipped_dataset_is_valid(shipped_dataset_path: Path) -> None:
    dataset = _load(shipped_dataset_path)
    assert dataset.count == 20
    assert dataset.version
    assert dataset.metadata.organization == "Bureau of Indian Standards"
    assert dataset.metadata.product_category == "rotating electrical machines"


def test_shipped_dataset_holds_exactly_the_expected_designations(
    shipped_dataset_path: Path,
) -> None:
    dataset = _load(shipped_dataset_path)
    numbers = [record.standard_number for record in dataset.records]
    assert set(numbers) == EXPECTED_DESIGNATIONS
    assert len(numbers) == len(set(numbers))


def test_shipped_records_contain_only_verified_information(
    shipped_dataset_path: Path,
) -> None:
    """Anything the source material did not provide must still be empty."""
    dataset = _load(shipped_dataset_path)
    for record in dataset.records:
        assert record.title is None
        assert record.scope is None
        assert record.status is None
        assert record.revision is None
        assert record.amendments == []
        assert record.related_standards == []
        assert record.references == []
        assert record.certification is None
        assert record.product_category == "rotating electrical machines"
        assert record.source.organization == "Bureau of Indian Standards"
        assert record.source.url is None
        assert record.source.document is None
        # The year must be the one carried by the designation itself.
        assert record.year == designation_year(record.standard_number)
