"""Standards dataset loader tests: path resolution, reading and entity mapping."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.exceptions import DatasetNotAvailableError, DatasetValidationError
from app.standards.dataset_loader import (
    StandardsDatasetLoader,
    get_standards_dataset_loader,
    to_standard_entity,
    to_standard_entities,
)
from app.standards.dataset_schema import validate_standards_dataset


def _write(path: Path, payload: object) -> Path:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(text, encoding="utf-8")
    return path


def _settings_without_dataset() -> SimpleNamespace:
    """Minimal stand-in for Settings with no dataset configured."""
    return SimpleNamespace(standards_dataset_path=None)


def test_missing_configuration_is_reported() -> None:
    loader = StandardsDatasetLoader(settings=_settings_without_dataset())
    assert loader.dataset_path is None
    with pytest.raises(DatasetNotAvailableError) as excinfo:
        loader.load()
    assert "STANDARDS_DATASET_PATH" in excinfo.value.message


def test_missing_file_is_reported(tmp_path: Path) -> None:
    loader = StandardsDatasetLoader(path=tmp_path / "absent.json")
    with pytest.raises(DatasetNotAvailableError) as excinfo:
        loader.load()
    assert "not found" in excinfo.value.message


def test_invalid_json_is_reported(tmp_path: Path) -> None:
    path = _write(tmp_path / "broken.json", "{not json")
    with pytest.raises(DatasetValidationError) as excinfo:
        StandardsDatasetLoader(path=path).load()
    assert "not valid JSON" in excinfo.value.message


def test_invalid_dataset_is_reported_with_details(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "bad.json",
        {"standards": [{"standard_number": "", "product_category": "x"}]},
    )
    with pytest.raises(DatasetValidationError) as excinfo:
        StandardsDatasetLoader(path=path).load()
    assert excinfo.value.details


def test_valid_dataset_loads(
    tmp_path: Path, dataset_payload_factory, standard_record_factory
) -> None:
    payload = dataset_payload_factory(
        standard_record_factory(standard_number="IS 1111:2001"),
        standard_record_factory(standard_number="IS 2222:2002"),
    )
    path = _write(tmp_path / "ok.json", payload)

    dataset = StandardsDatasetLoader(path=path).load()
    assert dataset.count == 2
    assert dataset.version == "9.9.9"
    assert dataset.get("is  1111:2001").standard_number == "IS 1111:2001"


def test_relative_path_is_resolved_from_the_backend_root() -> None:
    loader = StandardsDatasetLoader(path="../data/standards/standards.json")
    resolved = loader.resolve_path()
    assert resolved.is_file()
    assert resolved.parent.name == "standards"


def test_shipped_dataset_loads(shipped_dataset_path: Path) -> None:
    dataset, entities = StandardsDatasetLoader(path=shipped_dataset_path).load_entities()
    assert dataset.count == 20
    assert len(entities) == 20
    assert all(entity.code for entity in entities)


def test_loader_factory_uses_the_configured_path(
    tmp_path: Path, dataset_payload_factory
) -> None:
    path = _write(tmp_path / "configured.json", dataset_payload_factory())
    settings = SimpleNamespace(standards_dataset_path=path)

    dataset, entities = StandardsDatasetLoader(settings=settings).load_entities()
    assert dataset.count == 1
    assert entities[0].code == "IS 9999:2000"

    # The factory itself is bound to the real configuration.
    assert get_standards_dataset_loader().resolve_path().is_file()


# --- entity mapping -------------------------------------------------------


def test_mapping_keeps_only_verified_fields(dataset_payload_factory) -> None:
    record = validate_standards_dataset(dataset_payload_factory()).records[0]
    entity = to_standard_entity(record, dataset_version="9.9.9")

    assert entity.code == "IS 9999:2000"
    assert entity.id == "IS 9999:2000"
    assert entity.title is None
    assert entity.summary is None
    assert entity.scope is None
    assert entity.keywords == []
    assert entity.category == "rotating electrical machines"
    assert entity.certification == []
    assert entity.references == []
    assert entity.source == {
        "organization": "Bureau of Indian Standards",
        "url": None,
        "document": None,
    }
    assert entity.version is not None
    assert entity.version.year == 2000
    assert entity.version.amendments == []
    assert entity.metadata["dataset_version"] == "9.9.9"
    assert entity.metadata["related_standards"] == []


def test_mapping_carries_populated_fields(
    dataset_payload_factory, standard_record_factory
) -> None:
    payload = dataset_payload_factory(
        standard_record_factory(
            title="Test-only title",
            scope="Test-only scope",
            status="current",
            revision="First Revision",
            amendments=["Amendment No. 1"],
            related_standards=["IS 1111:2001"],
            references=[{"code": "IS 5678:1990", "relation": "normative"}],
            certification={"marking": "ISI Mark"},
        )
    )
    entity = to_standard_entities(validate_standards_dataset(payload))[0]

    assert entity.title == "Test-only title"
    assert entity.scope == "Test-only scope"
    assert entity.version.status == "current"
    assert entity.version.revision == "First Revision"
    assert entity.version.amendments == ["Amendment No. 1"]
    assert entity.certification == [
        {"scheme": None, "marking": "ISI Mark", "notes": None}
    ]
    assert entity.references[0]["code"] == "IS 5678:1990"
    assert entity.metadata["related_standards"] == ["IS 1111:2001"]


def test_version_is_omitted_when_nothing_is_recorded(
    dataset_payload_factory, standard_record_factory
) -> None:
    payload = dataset_payload_factory(
        standard_record_factory(standard_number="IS 9999", year=None)
    )
    entity = to_standard_entities(validate_standards_dataset(payload))[0]
    assert entity.version is None

