"""Load the verified Indian Standards dataset from disk.

The loader is the only component that knows where the dataset lives. It resolves
``STANDARDS_DATASET_PATH`` (relative paths are tried against ``backend/`` first,
then the project root, then the working directory), reads the JSON, validates it
and maps each record to a :class:`~app.models.standard.StandardEntity`.

Nothing is ever inferred: fields that the dataset leaves ``null`` stay ``null``.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.config import BACKEND_ROOT, PROJECT_ROOT, Settings, get_settings
from app.core.exceptions import DatasetNotAvailableError, DatasetValidationError
from app.core.logging import get_logger
from app.models.standard import StandardEntity, StandardVersionEntity
from app.standards.dataset_schema import (
    StandardDatasetRecord,
    StandardsDataset,
    validate_standards_dataset,
)

logger = get_logger(__name__)


def to_standard_entity(
    record: StandardDatasetRecord, *, dataset_version: str | None = None
) -> StandardEntity:
    """Map one dataset record to the domain entity used by the repositories."""
    has_version_info = bool(
        record.year or record.revision or record.amendments or record.status
    )
    version = (
        StandardVersionEntity(
            year=record.year,
            revision=record.revision,
            amendments=list(record.amendments),
            status=record.status,
        )
        if has_version_info
        else None
    )
    return StandardEntity(
        id=record.standard_number,
        code=record.standard_number,
        title=record.title,
        summary=None,  # never derived from the scope text
        category=record.product_category,
        keywords=[],  # never inferred from the designation
        scope=record.scope,
        version=version,
        certification=(
            [record.certification.model_dump()] if record.certification else []
        ),
        references=[reference.model_dump() for reference in record.references],
        source=record.source.model_dump() if record.source else None,
        metadata={
            "related_standards": list(record.related_standards),
            "dataset_version": dataset_version,
        },
    )


def to_standard_entities(dataset: StandardsDataset) -> list[StandardEntity]:
    """Map every record of a validated dataset to an entity."""
    return [
        to_standard_entity(record, dataset_version=dataset.version)
        for record in dataset.records
    ]


class StandardsDatasetLoader:
    """Reads and validates the standards dataset configured for this process."""

    def __init__(
        self, *, path: Path | str | None = None, settings: Settings | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self._path = Path(path) if path is not None else None

    @property
    def dataset_path(self) -> Path | None:
        """The configured (not yet resolved) dataset path, if any."""
        return self._path or self._settings.standards_dataset_path

    def resolve_path(self) -> Path:
        """Return the dataset file that will be read.

        Raises:
            DatasetNotAvailableError: when no path is configured or the file is
                missing.
        """
        configured = self.dataset_path
        if configured is None:
            raise DatasetNotAvailableError(
                "STANDARDS_DATASET_PATH is not configured, so no Indian Standards "
                "catalogue is available. Example: "
                "STANDARDS_DATASET_PATH=../data/standards/standards.json"
            )

        path = Path(configured).expanduser()
        if path.is_absolute():
            if not path.is_file():
                raise DatasetNotAvailableError(
                    f"Standards dataset file not found: {path}"
                )
            return path

        candidates: list[Path] = [
            BACKEND_ROOT / path,
            PROJECT_ROOT / path,
            Path.cwd() / path,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        checked = "; ".join(str(candidate) for candidate in candidates)
        raise DatasetNotAvailableError(
            f"Standards dataset '{path}' was not found. Locations checked: {checked}"
        )

    def load(self) -> StandardsDataset:
        """Resolve, read and validate the dataset.

        Raises:
            DatasetNotAvailableError: no path configured / file missing / unreadable.
            DatasetValidationError: the file is invalid; ``details`` lists why.
        """
        path = self.resolve_path()
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DatasetNotAvailableError(
                f"Could not read the standards dataset '{path}': {exc}"
            ) from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(
                f"The standards dataset '{path}' is not valid JSON: {exc.msg} "
                f"(line {exc.lineno}, column {exc.colno})."
            ) from exc

        dataset = validate_standards_dataset(payload, path=str(path))
        logger.info(
            "Loaded standards dataset '%s': %d records (version=%s)",
            path.name,
            dataset.count,
            dataset.version,
        )
        return dataset

    def load_entities(self) -> tuple[StandardsDataset, list[StandardEntity]]:
        """Load the dataset and map it to repository entities."""
        dataset = self.load()
        return dataset, to_standard_entities(dataset)


def get_standards_dataset_loader() -> StandardsDatasetLoader:
    """Return a loader bound to the current configuration."""
    return StandardsDatasetLoader()

