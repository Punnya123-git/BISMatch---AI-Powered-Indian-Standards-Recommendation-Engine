"""Standards dataset schema and validation tests.

The validator exists to keep unverified or malformed catalogue data away from the
engine, so every test here asserts that a problem *fails loudly* instead of being
skipped or silently repaired.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import DatasetValidationError
from app.standards.dataset_schema import (
    designation_key,
    designation_year,
    validate_standards_dataset,
)


def _issues(excinfo: pytest.ExceptionInfo) -> str:
    """Flatten the collected problems into one searchable string."""
    return " | ".join(str(issue) for issue in (excinfo.value.details or []))


# --- accepted shapes ------------------------------------------------------


def test_valid_envelope_is_accepted(dataset_payload_factory) -> None:
    dataset = validate_standards_dataset(dataset_payload_factory())
    assert dataset.count == 1
    assert dataset.version == "9.9.9"
    assert dataset.records[0].standard_number == "IS 9999:2000"


def test_bare_array_is_accepted_without_metadata(standard_record_factory) -> None:
    dataset = validate_standards_dataset([standard_record_factory()])
    assert dataset.count == 1
    assert dataset.version is None
    assert dataset.metadata.name is None


def test_unverified_fields_stay_empty(dataset_payload_factory) -> None:
    record = validate_standards_dataset(dataset_payload_factory()).records[0]
    assert record.title is None
    assert record.scope is None
    assert record.status is None
    assert record.revision is None
    assert record.amendments == []
    assert record.related_standards == []
    assert record.references == []
    assert record.certification is None
    assert record.source.organization == "Bureau of Indian Standards"
    assert record.source.url is None


def test_fully_populated_record_is_accepted(
    dataset_payload_factory, standard_record_factory
) -> None:
    record = standard_record_factory(
        title="Test-only title",
        scope="Test-only scope text",
        status="current",
        revision="Second Revision",
        amendments=["Amendment No. 1", "Amendment No. 2"],
        related_standards=["IS 1234:2001"],
        references=[{"code": "IS 5678:1990", "title": None, "relation": "normative"}],
        certification={"scheme": "BIS certification scheme", "marking": "ISI Mark"},
    )
    dataset = validate_standards_dataset(dataset_payload_factory(record))
    assert dataset.records[0].amendments == ["Amendment No. 1", "Amendment No. 2"]
    assert dataset.records[0].certification.marking == "ISI Mark"


# --- required fields and uniqueness ---------------------------------------


def test_missing_standard_number_fails(
    dataset_payload_factory, standard_record_factory
) -> None:
    record = standard_record_factory()
    del record["standard_number"]
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset(dataset_payload_factory(record))
    assert "standard_number" in _issues(excinfo)


def test_blank_standard_number_fails(
    dataset_payload_factory, standard_record_factory
) -> None:
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset(
            dataset_payload_factory(standard_record_factory(standard_number="   "))
        )
    assert "standard_number" in _issues(excinfo)


def test_duplicate_standard_numbers_fail(
    dataset_payload_factory, standard_record_factory
) -> None:
    payload = dataset_payload_factory(
        standard_record_factory(standard_number="IS 12615:2018"),
        standard_record_factory(standard_number="IS 12615:2018"),
    )
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset(payload)
    assert "duplicate standard_number" in _issues(excinfo)


def test_duplicate_detection_ignores_case_and_spacing(
    dataset_payload_factory, standard_record_factory
) -> None:
    payload = dataset_payload_factory(
        standard_record_factory(standard_number="IS 12615:2018"),
        standard_record_factory(standard_number="is  12615:2018"),
    )
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset(payload)
    assert "duplicate standard_number" in _issues(excinfo)


# --- types ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("amendments", "Amendment No. 1"),
        ("related_standards", "IS 1234:2001"),
        ("references", {"code": "IS 1234:2001"}),
        ("related_standards", None),
    ],
)
def test_non_array_collections_fail(
    dataset_payload_factory, standard_record_factory, field: str, value: object
) -> None:
    payload = dataset_payload_factory(standard_record_factory(**{field: value}))
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset(payload)
    assert field in _issues(excinfo)


@pytest.mark.parametrize(
    ("field", "value"),
    [("year", "2000"), ("product_category", 42), ("title", 7), ("amendments", [1, 2])],
)
def test_wrong_types_fail(
    dataset_payload_factory, standard_record_factory, field: str, value: object
) -> None:
    payload = dataset_payload_factory(standard_record_factory(**{field: value}))
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset(payload)
    assert field in _issues(excinfo)


def test_source_must_be_structured(
    dataset_payload_factory, standard_record_factory
) -> None:
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset(
            dataset_payload_factory(
                standard_record_factory(source="Bureau of Indian Standards")
            )
        )
    assert "source" in _issues(excinfo)


def test_source_requires_an_organization(
    dataset_payload_factory, standard_record_factory
) -> None:
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset(
            dataset_payload_factory(standard_record_factory(source={"url": None}))
        )
    assert "organization" in _issues(excinfo)


def test_unknown_record_field_fails(
    dataset_payload_factory, standard_record_factory
) -> None:
    payload = dataset_payload_factory(standard_record_factory(invented_field="guess"))
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset(payload)
    assert "invented_field" in _issues(excinfo)


def test_unknown_top_level_key_fails(dataset_payload_factory) -> None:
    payload = dataset_payload_factory()
    payload["extra_section"] = {}
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset(payload)
    assert "extra_section" in _issues(excinfo)


def test_unsupported_schema_version_fails(dataset_payload_factory) -> None:
    payload = dataset_payload_factory()
    payload["schema_version"] = "99"
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset(payload)
    assert "schema_version" in _issues(excinfo)


def test_empty_standards_array_fails() -> None:
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset({"standards": []})
    assert "at least one record" in _issues(excinfo)


def test_missing_standards_key_fails() -> None:
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset({"dataset": {"name": "empty"}})
    assert "standards" in _issues(excinfo)


def test_non_object_root_fails() -> None:
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset("IS 12615:2018")
    assert "root" in _issues(excinfo)


def test_record_count_mismatch_fails(dataset_payload_factory) -> None:
    payload = dataset_payload_factory()
    payload["dataset"]["record_count"] = 5
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset(payload)
    assert "record_count" in _issues(excinfo)


# --- consistency rules ----------------------------------------------------


def test_year_must_match_the_designation(
    dataset_payload_factory, standard_record_factory
) -> None:
    payload = dataset_payload_factory(standard_record_factory(year=1999))
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset(payload)
    assert "contradicts the designation" in _issues(excinfo)


def test_year_without_a_designation_year_fails(
    dataset_payload_factory, standard_record_factory
) -> None:
    payload = dataset_payload_factory(
        standard_record_factory(standard_number="IS 9999", year=2000)
    )
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset(payload)
    assert "no year suffix" in _issues(excinfo)


def test_certification_without_information_fails(
    dataset_payload_factory, standard_record_factory
) -> None:
    payload = dataset_payload_factory(
        standard_record_factory(certification={"scheme": None, "marking": None})
    )
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset(payload)
    assert "certification" in _issues(excinfo)


def test_self_references_fail(dataset_payload_factory, standard_record_factory) -> None:
    payload = dataset_payload_factory(
        standard_record_factory(
            related_standards=["IS 9999:2000"],
            references=[{"code": "IS 9999:2000"}],
        )
    )
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset(payload)
    issues = _issues(excinfo)
    assert "related_standards must not reference the standard itself" in issues
    assert "references must not point at the standard itself" in issues


def test_duplicate_amendments_and_references_fail(
    dataset_payload_factory, standard_record_factory
) -> None:
    payload = dataset_payload_factory(
        standard_record_factory(
            amendments=["Amendment No. 1", "Amendment No. 1"],
            references=[{"code": "IS 1:1990"}, {"code": "is 1:1990"}],
        )
    )
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset(payload)
    issues = _issues(excinfo)
    assert "duplicate entry 'Amendment No. 1'" in issues
    assert "duplicate reference 'is 1:1990'" in issues


def test_error_reports_every_problem_at_once(
    dataset_payload_factory, standard_record_factory
) -> None:
    payload = dataset_payload_factory(
        standard_record_factory(standard_number="IS 9999:2000", year=1999),
        standard_record_factory(standard_number="IS 8888:2001", title=7),
    )
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_standards_dataset(payload, path="C:/somewhere/standards.json")
    assert excinfo.value.code == "standards_dataset_invalid"
    assert "C:/somewhere/standards.json" in excinfo.value.message
    details = excinfo.value.details or []
    assert len(details) >= 2
    assert any("year" in issue for issue in details)
    assert any("title" in issue for issue in details)


def test_designation_helpers() -> None:
    assert designation_year("IS 12615:2018") == 2018
    assert designation_year("IS/IEC 60034-2-1:2024") == 2024
    assert designation_year("IS 9999") is None
    assert designation_key("  is  12615:2018 ") == "IS 12615:2018"


