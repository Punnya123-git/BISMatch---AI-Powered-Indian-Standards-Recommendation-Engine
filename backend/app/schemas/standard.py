"""Schemas describing Indian Standards records.

These mirror the fields the recommendation engine must be able to surface:
version/amendment details, certification requirements and normative references.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StandardVersion(BaseModel):
    """Version / amendment information for a standard."""

    year: int | None = Field(default=None, description="Year of the edition, e.g. 2018.")
    revision: str | None = Field(default=None, description="e.g. 'Fourth Revision'.")
    amendments: list[str] = Field(
        default_factory=list, description="Amendment identifiers, e.g. 'Amendment No. 2'."
    )
    status: str | None = Field(
        default=None, description="e.g. 'current', 'superseded', 'withdrawn'."
    )
    latest_revision: str | None = Field(
        default=None,
        description=(
            "A newer edition of this same standard, when the verified catalogue "
            "status text explicitly names one. Null when the catalogue does not "
            "say, so the catalogue edition is never silently called 'latest'."
        ),
    )


class CertificationRequirement(BaseModel):
    """Certification / conformity marking requirement attached to a standard."""

    scheme: str | None = Field(
        default=None, description="Compliance scheme, e.g. BIS certification scheme."
    )
    marking: str | None = Field(default=None, description="Marking, e.g. ISI Mark.")
    notes: str | None = None


class StandardReference(BaseModel):
    """A reference from one standard to another standard."""

    code: str = Field(description="Standard code, e.g. 'IS 269'.")
    title: str | None = None
    relation: str | None = Field(
        default=None,
        description="Relation type: normative, test, safety, installation, informative...",
    )


class StandardSource(BaseModel):
    """Provenance of a standards record (where the information came from)."""

    organization: str = Field(description="Publishing body, e.g. Bureau of Indian Standards.")
    url: str | None = None
    document: str | None = None


class StandardRecord(BaseModel):
    """A single Indian Standard in the knowledge base.

    Every field except ``code`` may be ``null``: the dataset only stores verified
    information, so an unknown title is reported as ``null`` instead of a guess.
    """

    code: str = Field(description="Canonical standard identifier, e.g. 'IS 12615:2018'.")
    title: str | None = Field(
        default=None, description="Null when the source material did not provide one."
    )
    summary: str | None = None
    category: str | None = None
    keywords: list[str] = Field(default_factory=list)
    scope: str | None = Field(
        default=None, description="Scope / applicability text from the standard."
    )
    version: StandardVersion | None = None
    certification: list[CertificationRequirement] = Field(default_factory=list)
    references: list[StandardReference] = Field(default_factory=list)
    related_standards: list[str] = Field(
        default_factory=list,
        description="Designations this standard is stated to be related to (dataset field).",
    )
    source: StandardSource | None = None


class DatasetStatusResponse(BaseModel):
    """Safe, non-sensitive description of the loaded standards dataset.

    Deliberately contains no filesystem path: the UI only needs to know whether a
    catalogue is available, how big it is and why it is not.
    """

    available: bool
    count: int = 0
    dataset_version: str | None = None
    organization: str | None = None
    product_category: str | None = None
    message: str
    issues: list[str] = Field(default_factory=list)


class StandardsCatalogResponse(BaseModel):
    """Payload of ``GET /api/standards``."""

    available: bool
    count: int
    dataset: DatasetStatusResponse
    standards: list[StandardRecord] = Field(default_factory=list)


class StandardMatch(BaseModel):
    """A retrieved standard together with its relevance score."""

    standard: StandardRecord
    score: float = Field(description="Similarity score from the vector store.")
    highlights: list[str] = Field(
        default_factory=list, description="Retrieved text chunks used as evidence."
    )
