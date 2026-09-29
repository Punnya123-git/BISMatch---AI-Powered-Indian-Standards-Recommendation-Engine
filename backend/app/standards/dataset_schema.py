"""Schema and validation for the verified Indian Standards dataset.

Rules encoded here (they exist to protect the project's honesty contract):

* A dataset file is a JSON object with ``schema_version``, ``dataset`` metadata
  and a ``standards`` array (a bare JSON array is also accepted, but then no
  dataset metadata such as a version is available).
* Every field that is not verified in the source material must be ``null`` or an
  empty array. The validator never fills anything in.
* ``year``, when present, must equal the year suffix of ``standard_number``, so a
  transcribed year can never contradict the designation it came from.
* Broken datasets fail loudly: :class:`~app.core.exceptions.DatasetValidationError`
  carries the list of concrete problems. No record is ever skipped silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.core.exceptions import DatasetValidationError

#: Version of the *file format* (not of the standards it contains).
DATASET_SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({DATASET_SCHEMA_VERSION})

_ENVELOPE_KEYS = frozenset({"schema_version", "dataset", "standards"})

#: Keys used by the enriched BIS draft format
#: (``data/standards/standards_enriched_bis_verified.json``). That draft describes
#: the catalogue with top level ``dataset_version`` / ``description`` /
#: ``source_note`` instead of a ``dataset`` object; they are accepted and mapped
#: onto :class:`StandardsDatasetMetadata` by
#: :func:`_draft_envelope_to_metadata` - no field is invented while doing so.
DRAFT_ENVELOPE_KEYS = frozenset({"dataset_version", "description", "source_note"})

_YEAR_SUFFIX = re.compile(r":\s*(\d{4})\s*$")

_MIN_DESIGNATION_LENGTH = 3
_MAX_DESIGNATION_LENGTH = 64


def designation_key(standard_number: str) -> str:
    """Return a comparison key for a standard designation.

    Only used to detect duplicates and to look records up case/space
    insensitively; the stored value is never rewritten.
    """
    return re.sub(r"\s+", " ", (standard_number or "").strip()).upper()


def designation_year(standard_number: str) -> int | None:
    """Return the year suffix of a designation (``IS 12615:2018`` -> ``2018``).

    Returns ``None`` when the designation carries no year. The value is read out
    of the designation itself, which is the only place a year is verified.
    """
    match = _YEAR_SUFFIX.search(standard_number or "")
    return int(match.group(1)) if match else None


class StandardSourceRecord(BaseModel):
    """Where a record came from (required: every record needs provenance)."""

    model_config = ConfigDict(extra="forbid")

    organization: str = Field(min_length=1, description="Publishing body, e.g. BIS.")
    url: str | None = None
    document: str | None = None


class StandardReferenceRecord(BaseModel):
    """A reference from one standard to another standard."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    title: str | None = None
    relation: str | None = None


class CertificationRecord(BaseModel):
    """Certification requirement as stated in the source material."""

    model_config = ConfigDict(extra="forbid")

    scheme: str | None = None
    marking: str | None = None
    notes: str | None = None

    def carries_information(self) -> bool:
        """True when at least one field is actually filled in."""
        return any(
            isinstance(value, str) and value.strip()
            for value in (self.scheme, self.marking, self.notes)
        )


class StandardDatasetRecord(BaseModel):
    """One Indian Standard exactly as it is stored in the dataset file."""

    model_config = ConfigDict(extra="forbid")

    standard_number: str = Field(strict=True)
    title: str | None = None
    scope: str | None = None
    product_category: str = Field(strict=True, min_length=1)
    status: str | None = None
    revision: str | None = None
    year: int | None = Field(default=None, ge=1900, le=2100, strict=True)
    amendments: list[str] = Field(default_factory=list)
    related_standards: list[str] = Field(default_factory=list)
    references: list[StandardReferenceRecord] = Field(default_factory=list)
    certification: CertificationRecord | None = None
    source: StandardSourceRecord

    @field_validator("references", mode="before")
    @classmethod
    def _references_from_designation_strings(cls, value: Any) -> Any:
        """Accept cross-references written as bare designation strings.

        The enriched BIS draft lists references as plain designations
        (``"IEC 60034-1:2022"``), while the canonical schema keeps them as objects
        so a title or relation can be recorded later. Each string becomes
        ``{"code": <string>}`` and no other field is filled in. A non-list value is
        passed through untouched so it still fails type validation.
        """
        if isinstance(value, list):
            return [
                {"code": item} if isinstance(item, str) else item for item in value
            ]
        return value

    @field_validator("certification", mode="before")
    @classmethod
    def _certification_from_note_string(cls, value: Any) -> Any:
        """Accept a certification statement written as one free-text note.

        The string becomes ``{"notes": <string>}``. The rule that a certification
        must carry at least one real value still applies, so a blank statement
        keeps failing validation instead of being silently accepted.
        """
        if isinstance(value, str):
            return {"notes": value}
        return value


class StandardsDatasetMetadata(BaseModel):
    """Dataset level description (never standard data itself)."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    version: str | None = None
    generated: str | None = None
    organization: str | None = None
    product_category: str | None = None
    record_count: int | None = Field(default=None, ge=0)
    notes: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class StandardsDataset:
    """A validated dataset: metadata plus immutable-by-convention records."""

    metadata: StandardsDatasetMetadata
    records: list[StandardDatasetRecord]
    schema_version: str = DATASET_SCHEMA_VERSION
    source_path: str | None = None

    @property
    def count(self) -> int:
        """Number of standards in the dataset."""
        return len(self.records)

    @property
    def version(self) -> str | None:
        """Dataset version, used as vector metadata (may be ``None``)."""
        return self.metadata.version

    @property
    def organization(self) -> str | None:
        """Publishing organization, if the dataset declares one."""
        if self.metadata.organization:
            return self.metadata.organization
        organizations = {
            record.source.organization for record in self.records if record.source
        }
        if len(organizations) == 1:
            return next(iter(organizations))
        return None

    def get(self, standard_number: str) -> StandardDatasetRecord | None:
        """Look a record up by its designation (case/space insensitive)."""
        key = designation_key(standard_number)
        for record in self.records:
            if designation_key(record.standard_number) == key:
                return record
        return None


def validate_standards_dataset(
    payload: Any, *, path: str | None = None
) -> StandardsDataset:
    """Validate a decoded dataset payload and return it as a typed object.

    Raises:
        DatasetValidationError: with a ``details`` list of every problem found.
    """
    issues: list[str] = []
    metadata, raw_records = _split_payload(payload, issues)

    if raw_records is not None and not raw_records:
        issues.append("'standards' must contain at least one record.")

    records = _validate_records(raw_records or [], issues)
    _validate_uniqueness(records, issues)

    if metadata is not None and metadata.record_count is not None:
        expected = len(raw_records or [])
        if metadata.record_count != expected:
            issues.append(
                "dataset.record_count does not match the number of 'standards' "
                f"entries ({metadata.record_count} != {expected})."
            )

    if issues:
        raise _validation_error(path, issues)
    return StandardsDataset(
        metadata=metadata or StandardsDatasetMetadata(),
        records=records,
        source_path=path,
    )


def _validation_error(path: str | None, issues: list[str]) -> DatasetValidationError:
    where = f" '{path}'" if path else ""
    message = (
        f"The standards dataset{where} failed validation with "
        f"{len(issues)} problem(s). No records were loaded."
    )
    return DatasetValidationError(message, details=list(issues))


def _draft_envelope_to_metadata(payload: dict[str, Any]) -> StandardsDatasetMetadata:
    """Map the enriched BIS draft envelope onto dataset metadata.

    Only text the draft itself declares is carried over (its version, description
    and source note); nothing is inferred from the records.
    """
    version = payload.get("dataset_version")
    notes: list[str] = []
    for key in ("description", "source_note"):
        text = payload.get(key)
        if isinstance(text, str) and text.strip():
            notes.append(text)
    return StandardsDatasetMetadata(
        version=version.strip() if isinstance(version, str) and version.strip() else None,
        notes=notes,
    )


def _split_payload(
    payload: Any, issues: list[str]
) -> tuple[StandardsDatasetMetadata | None, list[Any] | None]:
    """Return ``(metadata, raw_records)`` from either supported file shape."""
    if isinstance(payload, list):
        return None, payload
    if not isinstance(payload, dict):
        issues.append(
            "The dataset root must be a JSON object with 'standards' (or an array "
            f"of records), got {type(payload).__name__}."
        )
        return None, None

    for key in payload:
        if key not in _ENVELOPE_KEYS and key not in DRAFT_ENVELOPE_KEYS:
            issues.append(
                f"Unknown top level key '{key}'. Allowed keys: "
                f"{', '.join(sorted(_ENVELOPE_KEYS | DRAFT_ENVELOPE_KEYS))}."
            )

    schema_version = payload.get("schema_version", DATASET_SCHEMA_VERSION)
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        issues.append(
            f"Unsupported schema_version '{schema_version}'. Supported: "
            f"{', '.join(sorted(SUPPORTED_SCHEMA_VERSIONS))}."
        )

    metadata: StandardsDatasetMetadata | None = None
    if "dataset" in payload:
        if isinstance(payload["dataset"], dict):
            try:
                metadata = StandardsDatasetMetadata.model_validate(payload["dataset"])
            except ValidationError as exc:
                issues.extend(
                    f"dataset.{_location(error)}: {error['msg']}"
                    for error in exc.errors()
                )
        else:
            issues.append("'dataset' must be a JSON object when present.")
    elif any(key in payload for key in DRAFT_ENVELOPE_KEYS):
        metadata = _draft_envelope_to_metadata(payload)

    raw_records = payload.get("standards")
    if raw_records is None:
        issues.append("The dataset must contain a 'standards' array.")
    elif not isinstance(raw_records, list):
        issues.append("'standards' must be an array.")
        raw_records = None
    return metadata, raw_records


def _validate_records(
    records: list[Any], issues: list[str]
) -> list[StandardDatasetRecord]:
    validated: list[StandardDatasetRecord] = []
    for index, raw in enumerate(records):
        label = f"standards[{index}]"
        if not isinstance(raw, dict):
            issues.append(f"{label}: each record must be a JSON object.")
            continue
        number = raw.get("standard_number")
        if isinstance(number, str) and number.strip():
            label = f"standards[{index}] ({number.strip()})"
        try:
            record = StandardDatasetRecord.model_validate(raw)
        except ValidationError as exc:
            issues.extend(
                f"{label}: {_location(error)}: {error['msg']}" for error in exc.errors()
            )
            continue
        issues.extend(_record_issues(record, label))
        validated.append(record)
    return validated


def _record_issues(record: StandardDatasetRecord, label: str) -> list[str]:
    issues: list[str] = []
    number = record.standard_number

    if number != number.strip():
        issues.append(f"{label}: standard_number must not have surrounding spaces.")
    if re.search(r"\s{2,}", number):
        issues.append(f"{label}: standard_number must not contain repeated spaces.")
    if re.search(r"[\r\n\t]", number):
        issues.append(f"{label}: standard_number must be a single line.")
    if not re.search(r"\d", number):
        issues.append(f"{label}: standard_number must contain a numeric part.")
    if not (_MIN_DESIGNATION_LENGTH <= len(number) <= _MAX_DESIGNATION_LENGTH):
        issues.append(
            f"{label}: standard_number must be {_MIN_DESIGNATION_LENGTH}-"
            f"{_MAX_DESIGNATION_LENGTH} characters long."
        )

    suffix_year = designation_year(number)
    if record.year is not None and suffix_year is None:
        issues.append(
            f"{label}: year {record.year} is set but the designation has no year "
            "suffix to verify it against."
        )
    if record.year is not None and suffix_year is not None and record.year != suffix_year:
        issues.append(
            f"{label}: year {record.year} contradicts the designation, which ends "
            f"with {suffix_year}."
        )

    if record.certification is not None and not record.certification.carries_information():
        issues.append(
            f"{label}: certification must be null or carry at least one value "
            "(scheme/marking/notes)."
        )

    issues.extend(_string_list_issues(record.amendments, f"{label}: amendments"))
    issues.extend(
        _string_list_issues(record.related_standards, f"{label}: related_standards")
    )

    own_key = designation_key(number)
    for related in record.related_standards:
        if designation_key(related) == own_key:
            issues.append(
                f"{label}: related_standards must not reference the standard itself."
            )

    seen_codes: set[str] = set()
    for reference in record.references:
        key = designation_key(reference.code)
        if key == own_key:
            issues.append(f"{label}: references must not point at the standard itself.")
        if key in seen_codes:
            issues.append(f"{label}: duplicate reference '{reference.code}'.")
        seen_codes.add(key)
    return issues


def _string_list_issues(values: list[str], label: str) -> list[str]:
    issues: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            issues.append(f"{label}: entries must be non-empty strings.")
            continue
        if value != value.strip():
            issues.append(f"{label}: entry '{value}' has surrounding whitespace.")
        key = designation_key(value)
        if key in seen:
            issues.append(f"{label}: duplicate entry '{value}'.")
        seen.add(key)
    return issues


def _validate_uniqueness(records: list[StandardDatasetRecord], issues: list[str]) -> None:
    seen: dict[str, str] = {}
    for record in records:
        key = designation_key(record.standard_number)
        if key in seen:
            issues.append(
                f"duplicate standard_number: '{record.standard_number}' is listed more "
                f"than once (first seen as '{seen[key]}')."
            )
        else:
            seen[key] = record.standard_number


def _location(error: dict[str, Any]) -> str:
    return ".".join(str(part) for part in error.get("loc", ())) or "<record>"


