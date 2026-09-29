"""Standards catalogue service.

Reads from the repository abstraction and maps entities to API schemas. It never
synthesises standards: an empty catalogue is reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import re

from app.core.config import Settings, get_settings
from app.core.exceptions import AppError, NotFoundError
from app.database.repositories import StandardRepository, get_standard_repository
from app.models.standard import StandardEntity
from app.schemas.standard import (
    CertificationRequirement,
    DatasetStatusResponse,
    StandardRecord,
    StandardReference,
    StandardSource,
    StandardsCatalogResponse,
    StandardVersion,
)
from app.standards.dataset_loader import (
    StandardsDatasetLoader,
    get_standards_dataset_loader,
)
from app.standards.dataset_schema import StandardsDataset


@dataclass(slots=True)
class DatasetStatus:
    """Whether the Indian Standards catalogue has been loaded, and from what.

    Contains no filesystem path: file locations are an operational detail, not
    something the API or the UI should depend on.
    """

    available: bool
    loaded: bool
    count: int
    dataset_version: str | None
    organization: str | None
    product_category: str | None
    message: str
    issues: list[str] = field(default_factory=list)

    def to_response(self) -> DatasetStatusResponse:
        """Map to the API schema used by /api/health and /api/standards."""
        return DatasetStatusResponse(
            available=self.available,
            count=self.count,
            dataset_version=self.dataset_version,
            organization=self.organization,
            product_category=self.product_category,
            message=self.message,
            issues=list(self.issues),
        )


#: A designation such as ``IS 12615:2026`` or ``IS/IEC 60034-1:2022``.
_DESIGNATION_RE = re.compile(
    r"\b(IS/IEC|IS)\s*\.?\s*(\d+(?:\.\d+)?)\s*[-:]\s*(\d{4})\b", re.IGNORECASE
)


def _latest_revision_from_status(
    standard_number: str | None, status: str | None
) -> str | None:
    """A newer edition of *this* standard, when verified catalogue text names one.

    Strictly evidence-bound: the string comes from the verified catalogue ``status``
    field, and it only counts when it is the same standard number with a strictly
    later year. Returns ``None`` otherwise - including when nothing is stated - so
    a catalogue edition is never silently presented as the latest one.
    """
    if not standard_number or not status:
        return None
    current = _DESIGNATION_RE.search(standard_number)
    if current is None:
        return None
    current_key = re.sub(
        r"[^a-z0-9]", "", f"{current.group(1)}{current.group(2)}"
    ).lower()
    current_year = int(current.group(3))

    best_year: int | None = None
    best_text: str | None = None
    for match in _DESIGNATION_RE.finditer(status):
        key = re.sub(
            r"[^a-z0-9]", "", f"{match.group(1)}{match.group(2)}"
        ).lower()
        if key != current_key:
            continue
        year = int(match.group(3))
        if year <= current_year:
            continue
        if best_year is None or year > best_year:
            best_year = year
            best_text = f"{match.group(1)} {match.group(2)}:{year}"
    return best_text


def to_standard_record(entity: StandardEntity) -> StandardRecord:
    """Map a stored entity to the API schema."""
    version = None
    if entity.version is not None:
        version = StandardVersion(
            year=entity.version.year,
            revision=entity.version.revision,
            amendments=list(entity.version.amendments),
            status=entity.version.status,
            latest_revision=_latest_revision_from_status(
                entity.code, entity.version.status
            ),
        )
    return StandardRecord(
        code=entity.code,
        title=entity.title,
        summary=entity.summary,
        category=entity.category,
        keywords=list(entity.keywords),
        scope=entity.scope,
        version=version,
        certification=[
            CertificationRequirement(**item) for item in entity.certification
        ],
        references=[StandardReference(**item) for item in entity.references],
        related_standards=list(entity.metadata.get("related_standards") or []),
        source=StandardSource(**entity.source) if entity.source else None,
    )


class StandardsService:
    """Business logic around the Indian Standards catalogue."""

    def __init__(
        self,
        repository: StandardRepository | None = None,
        settings: Settings | None = None,
        loader: StandardsDatasetLoader | None = None,
    ) -> None:
        self._repository = repository or get_standard_repository()
        self._settings = settings or get_settings()
        self._loader = loader or get_standards_dataset_loader()
        self._dataset: StandardsDataset | None = None

    @property
    def repository(self) -> StandardRepository:
        return self._repository

    @property
    def loader(self) -> StandardsDatasetLoader:
        return self._loader

    @property
    def dataset(self) -> StandardsDataset | None:
        """The dataset currently loaded into the catalogue, if any."""
        return self._dataset

    # --- dataset loading --------------------------------------------------
    def ensure_loaded(self, *, force: bool = False) -> DatasetStatus:
        """Populate the repository from the configured dataset file.

        Idempotent: the file is read once per service instance. Failures are
        raised (``DatasetNotAvailableError`` / ``DatasetValidationError``) so the
        caller can report them explicitly - invalid records are never skipped.
        """
        if self._dataset is not None and not force and self._repository.count():
            return self.dataset_status()

        dataset, entities = self._loader.load_entities()
        if force:
            self._repository.clear()
        self._repository.add_many(entities)
        self._dataset = dataset
        return self.dataset_status()

    def dataset_status(self) -> DatasetStatus:
        """Describe the dataset without pretending it exists and without raising.

        Safe to call from ``/api/health`` even when the dataset is missing or
        invalid: problems are reported in ``message`` / ``issues``.
        """
        dataset = self._dataset
        if dataset is None:
            try:
                dataset = self._loader.load()
            except AppError as exc:
                details = [str(issue) for issue in (exc.details or [])] or [exc.message]
                return DatasetStatus(
                    available=False,
                    loaded=self._repository.count() > 0,
                    count=0,
                    dataset_version=None,
                    organization=None,
                    product_category=None,
                    message=exc.message,
                    issues=details,
                )

        count = self._repository.count()
        loaded = count > 0
        if loaded:
            message = (
                f"{count} standards loaded from dataset version "
                f"{dataset.version or 'unversioned'} "
                f"({dataset.count} record(s) in the dataset file)."
            )
        else:
            message = (
                f"The dataset file holds {dataset.count} record(s) but they have "
                "not been loaded into the catalogue."
            )
        return DatasetStatus(
            available=True,
            loaded=loaded,
            count=count,
            dataset_version=dataset.version,
            organization=dataset.organization,
            product_category=dataset.metadata.product_category,
            message=message,
            issues=[],
        )

    def catalog(
        self, *, limit: int = 50, offset: int = 0, query: str | None = None
    ) -> StandardsCatalogResponse:
        """Return the safe catalogue payload used by ``GET /api/standards``."""
        self.ensure_loaded()
        status = self.dataset_status()
        if query and query.strip():
            standards = self.keyword_search(query, limit=limit)
        else:
            standards = self.list_standards(limit=limit, offset=offset)
        return StandardsCatalogResponse(
            available=status.loaded,
            count=status.count,
            dataset=status.to_response(),
            standards=standards,
        )

    def list_standards(self, *, limit: int = 50, offset: int = 0) -> list[StandardRecord]:
        """List catalogue entries as API schemas."""
        return [
            to_standard_record(entity)
            for entity in self._repository.list_all(limit=limit, offset=offset)
        ]

    def get_standard(self, code: str) -> StandardRecord:
        """Fetch one standard by code, or raise ``NotFoundError``."""
        entity = self._repository.get_by_code(code)
        if entity is None:
            raise NotFoundError(f"Standard '{code}' is not in the catalogue.")
        return to_standard_record(entity)

    def keyword_search(self, query: str, *, limit: int = 10) -> list[StandardRecord]:
        """Lexical search used as a baseline (semantic search lives in app.rag)."""
        return [
            to_standard_record(entity)
            for entity in self._repository.keyword_search(query, limit=limit)
        ]


def get_standards_service() -> StandardsService:
    """Return the process-wide standards service.

    Cached so the dataset is read and mapped once per process; the underlying
    repository is shared with every other consumer of the catalogue.
    """
    return _get_standards_service()


@lru_cache
def _get_standards_service() -> StandardsService:
    return StandardsService()


def reset_standards_service_cache() -> None:
    """Clear the cache (used by tests and after configuration changes)."""
    _get_standards_service.cache_clear()
