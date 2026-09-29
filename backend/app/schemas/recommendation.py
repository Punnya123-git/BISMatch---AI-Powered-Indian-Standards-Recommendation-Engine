"""Schemas for the recommendation engine."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field

from app.schemas.standard import (
    CertificationRequirement,
    StandardReference,
    StandardVersion,
)


class RecommendationStatus(str, Enum):
    """Lifecycle state of a recommendation response."""

    OK = "ok"
    AI_UNAVAILABLE = "ai_unavailable"
    PLACEHOLDER = "placeholder"
    NOT_CONFIGURED = "not_configured"
    DATASET_UNAVAILABLE = "dataset_unavailable"


class NoResultReason(str, Enum):
    """Why an empty ``recommendations`` list is empty.

    These are genuinely different facts and collapsing them would mislead a
    procurement user, so the backend states which one applies rather than
    leaving the UI to guess:

    * ``AI_RULED_NONE`` - the AI reviewed every verified candidate that
      retrieval produced and judged that none of them applies. The catalogue
      *does* cover this kind of product; the candidates simply do not match.
    * ``INSUFFICIENT_COVERAGE`` - retrieval produced no catalogued candidate at
      all, so the locally indexed verified knowledge base does not cover this
      requirement. This says nothing about whether an Indian Standard exists:
      the official BIS catalogue is far larger than the local index.
    * ``AI_UNAVAILABLE`` - the reasoning stage failed, so no decision was
      possible either way.
    * ``RETRIEVAL_UNAVAILABLE`` - retrieval itself could not run: either it is
      not configured at all, or the query failed while the pipeline was ready.
      This is never reported as a coverage gap.
    """

    AI_RULED_NONE = "ai_ruled_none"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    AI_UNAVAILABLE = "ai_unavailable"
    RETRIEVAL_UNAVAILABLE = "retrieval_unavailable"


class EvidenceItem(BaseModel):
    """A single piece of evidence backing a recommendation."""

    source: str = Field(description="Where the evidence came from.")
    document_id: str | None = None
    standard_code: str | None = None
    chunk_id: str | None = None
    snippet: str = Field(description="Verbatim text supporting the recommendation.")
    page_number: int | None = None
    score: float | None = None


class RecommendedStandard(BaseModel):
    """One recommended standard, with justification and evidence.

    Every value is derived from the loaded, validated catalogue and the chunks
    retrieved from the index: no title, scope, certification or designation is
    ever produced from memory. ``confidence`` is the relevance score assigned by
    the ranking stage (see :mod:`app.rag.ranking`).

    ``standard_number`` and ``why_it_matches`` are exposed as computed aliases of
    ``code`` and ``reasoning``: the newer, more explicit names are part of the
    response contract while the original keys keep existing clients (and the
    bundled UI) working unchanged.
    """

    code: str
    title: str | None = Field(
        default=None, description="Null when the catalogue holds no verified title."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Relevance/confidence score (0-1) of this standard for the requirement."
        ),
    )
    category: str | None = None
    reasoning: str = Field(description="Why this standard is applicable.")
    applicability_type: Literal["direct", "related", "needs_verification"] = Field(
        default="direct",
        description=(
            "Classification of applicability: 'direct' (directly governs product), "
            "'related' (normative/supporting standard), or 'needs_verification' "
            "(insufficient evidence to determine full applicability)."
        ),
    )
    matched_requirements: list[str] = Field(default_factory=list)
    uncovered_requirements: list[str] = Field(
        default_factory=list,
        description="Requirement attributes or aspects not explicitly covered by this standard.",
    )
    deterministic_rank: int | None = Field(
        default=None,
        description=(
            "1-based position this standard held in the deterministic retrieval "
            "ranking. It is a supporting signal, not the applicability decision, "
            "and it stays attached after the reasoning layer reorders the list so "
            "the candidate order and the final order remain comparable."
        ),
    )
    version: StandardVersion | None = None
    certification: list[CertificationRequirement] = Field(default_factory=list)
    related_standards: list[StandardReference] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def standard_number(self) -> str:
        """Catalogued designation; identical to :attr:`code`."""
        return self.code

    @computed_field  # type: ignore[prop-decorator]
    @property
    def why_it_matches(self) -> str:
        """Human-readable justification; identical to :attr:`reasoning`."""
        return self.reasoning


class RetrievedCandidate(BaseModel):
    """A catalogue-verified standard that retrieval surfaced.

    This is a *candidate for AI reasoning*, never an applicability decision.
    It deliberately carries no ``confidence``, no ``applicability_type`` and no
    generated justification: the deterministic layer only reports that it
    retrieved the record and how strongly it matched, which is exactly the
    supporting-signal role it is allowed to play.
    """

    standard_number: str
    title: str | None = None
    category: str | None = None
    retrieval_rank: int = Field(
        description="1-based position after deterministic semantic retrieval."
    )
    similarity_score: float | None = Field(
        default=None,
        description="Vector-store similarity. A retrieval signal, not applicability.",
    )
    attribute_score: float | None = Field(
        default=None,
        description="Share of stated requirement attributes this record states. A signal.",
    )
    matched_attributes: list[str] = Field(default_factory=list)
    disclaimer: str = Field(
        default=(
            "Retrieved candidate - not an applicability decision. Only AI reasoning "
            "can classify this standard as applicable."
        )
    )


class DetectedAttribute(BaseModel):
    """One technical attribute read verbatim out of the requirement text."""

    kind: str
    label: str
    surface: str = Field(description="The exact text the attribute was read from.")
    value: str | None = Field(
        default=None, description="Normalized value, or null when only a family matched."
    )


class RequirementUnderstanding(BaseModel):
    """Supporting description of the requirement, shown before/with analysis.

    These attributes are *supporting information only*. They never decide
    applicability: the AI reasoning stage remains the sole authority.
    """

    detected: list[DetectedAttribute] = Field(default_factory=list)
    unresolved_families: list[str] = Field(
        default_factory=list,
        description=(
            "Attribute families the requirement asks for that no retrieved record "
            "states, so they could not influence candidate retrieval."
        ),
    )
    note: str = Field(
        default=(
            "Extracted from the requirement text for transparency. These are "
            "supporting signals only; applicability is decided by AI reasoning."
        )
    )


class AIReasoningInfo(BaseModel):
    """Outcome of the applicability-authority stage (diagnostics for the UI)."""

    status: Literal["completed", "unavailable", "not_configured"] = Field(
        description=(
            "'completed' = the model produced the final verdict; "
            "'unavailable' = the stage ran but failed, so no final recommendation "
            "was made; 'not_configured' = no LLM provider is configured."
        )
    )
    provider: str | None = None
    model: str | None = None
    candidates_supplied: int = 0
    candidates_selected: int = 0
    candidates_excluded: list[str] = Field(default_factory=list)
    invented_dropped: list[str] = Field(
        default_factory=list,
        description="Designations the model named that were not verified candidates.",
    )
    truncated: bool = False
    error: str | None = None


class RecommendationRequest(BaseModel):
    """Body of ``POST /api/recommendations/analyze``."""

    requirement: str = Field(
        min_length=3,
        description="Free-text procurement / product requirement.",
    )
    document_id: str | None = Field(
        default=None,
        description="Optional id returned by /api/documents/upload.",
    )
    top_k: int | None = Field(default=None, ge=1, le=50)


class PipelineInfo(BaseModel):
    """Which parts of the AI/RAG stack are available right now.

    The UI uses this to explain *why* no recommendations were produced yet,
    instead of pretending a recommendation exists. ``ready`` means "retrieval and
    ranking can run", which does not depend on an LLM: ``ranking_stage`` states
    which stage actually produced the recommendations.
    """

    ready: bool
    llm_provider: str
    embedding_provider: str
    vector_store: str
    indexed_chunks: int = 0
    standards_available: int = 0
    ranking_stage: str = Field(
        default="ai_reasoning_unavailable",
        description=(
            "Who produced ``recommendations``. 'ai_reasoning_ranking' means the "
            "LLM made the final applicability classification and ordering and is "
            "the ONLY value that may accompany a non-empty recommendations list. "
            "'ai_reasoning_unavailable' means the reasoning stage could not "
            "complete, so no final recommendation was made and "
            "``recommendations`` is empty. 'deterministic_retrieval_ranking' is "
            "retained for compatibility and describes the retrieval/support layer "
            "only - it never yields a final recommendation."
        ),
    )
    reasons: list[str] = Field(default_factory=list)


class ProcurementTargetInfo(BaseModel):
    """What the AI concluded is actually being procured.

    Surfaced so the user can see the same target the candidates were judged
    against. It is the model's own statement, not a classification produced by
    the retrieval layer.
    """

    target: str = Field(description="The product/material/equipment/service being procured.")
    explicit_specifications: list[str] = Field(
        default_factory=list,
        description="Specifications the user explicitly stated.",
    )
    note: str | None = None


class RecommendationResponse(BaseModel):
    """Payload returned by ``POST /api/recommendations/analyze``.

    Invariant enforced by the service: ``recommendations`` is non-empty only
    when ``pipeline.ranking_stage == "ai_reasoning_ranking"``. The deterministic
    retrieval layer supplies ``retrieved_candidates`` and ``retrieved_evidence``
    as supporting material; it never becomes the applicability authority.
    """

    status: RecommendationStatus
    message: str = Field(description="Explicit statement of what was (not) produced.")
    requirement: str
    document_id: str | None = None
    recommendations: list[RecommendedStandard] = Field(default_factory=list)
    retrieved_candidates: list[RetrievedCandidate] = Field(
        default_factory=list,
        description=(
            "Catalogue-verified standards surfaced by semantic retrieval. These are "
            "NOT applicability decisions and are never presented as recommendations."
        ),
    )
    retrieved_evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Retrieved context chunks (real retrieval output, not generated).",
    )
    requirement_understanding: RequirementUnderstanding | None = None
    ai_reasoning: AIReasoningInfo | None = None
    procurement_target: ProcurementTargetInfo | None = Field(
        default=None,
        description=(
            "The AI's own statement of what is actually being procured. Shown so "
            "the applicability decision is inspectable against the same target the "
            "model used."
        ),
    )
    pipeline: PipelineInfo | None = None
    warnings: list[str] = Field(default_factory=list)
    no_result_reason: NoResultReason | None = Field(
        default=None,
        description=(
            "Set only when `recommendations` is empty. States *why* it is empty so "
            "the UI never guesses and never conflates 'the AI reviewed the "
            "candidates and found none applicable' with 'the local verified index "
            "does not cover this requirement'."
        ),
    )
    coverage_note: str | None = Field(
        default=None,
        description=(
            "Honest scope statement about the local verified index, shown when "
            "coverage is the limiting factor. Never implies that no Indian "
            "Standard exists outside the local index."
        ),
    )

