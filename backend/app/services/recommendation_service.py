"""Recommendation service: the single orchestration point of the API.

ARCHITECTURE (the single most important rule of this module)
----------------------------------------------------------
``deterministic retrieval`` -> ``verified evidence`` -> ``AI/LLM reasoning``
-> ``FINAL RECOMMENDATIONS``.

The deterministic layer (:mod:`app.rag.ranking`, :mod:`app.rag.retrieval`) is a
**retrieval / support layer only**. It embeds, searches ChromaDB, verifies
designations against the loaded catalogue, applies a relevance floor, extracts
technical attributes and computes supporting signals. It MUST NOT decide that a
standard is applicable, classify Direct/Related, assign applicability
confidence, generate the final "why it matches" text, or set the final order.

:mod:`app.ai.reasoning` is the **sole applicability authority**. It selects,
excludes, classifies, reorders and explains - strictly from the verified
candidate set it was given, and strictly from verified catalogue fields.

Consequence, enforced below and asserted by the test-suite:
``recommendations`` is non-empty **only** when
``pipeline.ranking_stage == AI_RANKING_STAGE``. If AI reasoning is not
configured, fails, times out, rate-limits, returns malformed JSON, is truncated
or returns nothing verifiable, the response carries **no** final
recommendation at all - the retrieved candidates are still returned, but only
as ``retrieved_candidates`` (``RetrievedCandidate``), which by construction has
no confidence, no applicability class and no generated justification.

No standard is ever invented: a designation only reaches the response when it
is present in the loaded, validated catalogue.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.ai.llm.factory import get_llm_provider
from app.ai.reasoning import (
    CandidateEvidence,
    CandidateStandard,
    ReasonedStandard,
    ReasoningOutcome,
    normalize_designation,
    run_ai_reasoning,
)
from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.rag.attributes import KIND_LABELS, extract_technical_attributes
from app.rag.pipeline import (
    PipelineReadiness,
    RecommendationPipeline,
    get_recommendation_pipeline,
)
from app.rag.ranking import (
    RankedStandard,
    RankingOutcome,
    rank_standards,
)
from app.rag.retrieval import RetrievedChunk
from app.schemas.recommendation import (
    AIReasoningInfo,
    DetectedAttribute,
    EvidenceItem,
    NoResultReason,
    PipelineInfo,
    ProcurementTargetInfo,
    RecommendationRequest,
    RecommendationResponse,
    RecommendationStatus,
    RequirementUnderstanding,
    RecommendedStandard,
    RetrievedCandidate,
)
from app.services.document_service import DocumentService, get_document_service
from app.services.standards_service import (
    StandardsService,
    get_standards_service,
    to_standard_record,
)

logger = get_logger(__name__)

#: Reported in ``pipeline.ranking_stage`` for the retrieval/support layer
#: (similarity + catalogue verification, no language model). **This value never
#: accompanies a non-empty ``recommendations`` list**; it is kept only so older
#: clients keep parsing, and it describes candidates, not decisions.
RANKING_STAGE = "deterministic_retrieval_ranking"

#: Reported in ``pipeline.ranking_stage`` when the configured LLM made the final
#: applicability classification and ranking from the verified candidates. The
#: ONLY stage that may accompany final recommendations.
AI_RANKING_STAGE = "ai_reasoning_ranking"

#: Reported when the applicability-authority stage could not complete. No final
#: recommendation is made; retrieved candidates are reported as candidates only.
AI_UNAVAILABLE_STAGE = "ai_reasoning_unavailable"

#: Upper bound on how many verified candidates are handed to the model.
REASONING_MAX_CANDIDATES = 12

_EVIDENCE_SOURCE = "standards_index"

_NOT_CONFIGURED_MESSAGE = (
    "Semantic retrieval is not available (no embedding provider, no vector store "
    "or an empty index), so no standards could be ranked. Nothing is guessed in "
    "the meantime: build the index and configure the embedding provider, then "
    "retry."
)
_DATASET_MESSAGE = (
    "No Indian Standards dataset has been loaded or indexed yet, so no standards "
    "can be recommended. Import a verified standards dataset and build the index "
    "to enable recommendations."
)
_PLACEHOLDER_MESSAGE = (
    "Your requirement is outside the currently indexed verified knowledge base. "
    "The local index holds a limited, verified subset of Indian Standards for "
    "rotating electrical machines, and no catalogued record matched this "
    "requirement closely enough to be evaluated. This is a limit of this "
    "assistant's local index only - it is NOT a statement that no Indian "
    "Standard exists. The official BIS catalogue is far larger, and for "
    "authoritative confirmation please search the official BIS portal at "
    "standards.bis.gov.in."
)

#: Used when the retrieval *query* failed even though the pipeline reported
#: itself ready. Kept apart from the coverage message on purpose: a broken query
#: says nothing at all about what the index does or does not contain.
_RETRIEVAL_FAILED_MESSAGE = (
    "The verified evidence for this requirement could not be retrieved because "
    "the search index query failed, so no standard could be evaluated. Nothing is "
    "presented as applicable, and this is not a statement about whether an Indian "
    "Standard exists. Please try the analysis again."
)

#: Honest, non-endorsing scope statement. The BIS does not operate or endorse
#: this product, so the wording points at the official source rather than
#: implying local completeness.
_COVERAGE_NOTE = (
    "Local verified index coverage: rotating electrical machines. For the "
    "complete official catalogue, search standards.bis.gov.in."
)


def _ok_message(count: int, *, ai_ranked: bool = False) -> str:
    """Message used when grounded recommendations were produced."""
    if ai_ranked:
        return (
            f"{count} applicable standard(s) identified by AI applicability "
            "reasoning over the verified evidence retrieved for this requirement. "
            "The model selected these from the catalogue-verified candidates, "
            "classified their applicability and set this order; each one cites the "
            "retrieved text it was judged on. No standard outside the loaded "
            "dataset is ever proposed."
        )
    return (
        f"{count} standard(s) were retrieved for this requirement. They are "
        "reported as candidates only: without an AI applicability verdict they "
        "are NOT recommendations, and no applicability has been decided."
    )


_AI_UNAVAILABLE_MESSAGE = (
    "AI applicability analysis could not be completed, so we could not safely "
    "determine which standards apply. Nothing is presented as applicable."
)

_AI_RATE_LIMIT_MESSAGE = (
    "AI applicability analysis is temporarily unavailable because the analysis "
    "service reached its usage limit. No standard is presented as applicable. "
    "Please try again in a few minutes."
)

#: Raw provider errors (model names, quota figures, organisation ids) are
#: developer detail. They must never appear in a user-facing message or warning,
#: or a rate limit leaks infrastructure into the main reading flow. The service
#: exposes them through `ai_reasoning.error` for the "Technical details" panel.
_PROVIDER_ERROR_PREFIX = "Analysis service detail (see Technical details): "

_AI_NOT_CONFIGURED_MESSAGE = (
    "AI applicability analysis is not configured on this deployment, so we could "
    "not determine which standards apply. The potential matches below are search "
    "results only, and nothing is presented as applicable."
)

_AI_EXCLUDED_ALL_MESSAGE = (
    "No applicable standard identified. AI applicability analysis reviewed the "
    "standards found in the verified catalogue and determined that none of them "
    "applies to this requirement."
)

#: Substrings that identify a provider rate limit, so the user-facing message can
#: say so plainly instead of "something went wrong".
_RATE_LIMIT_MARKERS = ("rate limit", "429", "too many requests", "quota")


def _is_rate_limit(error: str | None) -> bool:
    """True when a reasoning error is a provider rate limit rather than a fault."""
    if not error:
        return False
    lowered = error.lower()
    return any(marker in lowered for marker in _RATE_LIMIT_MARKERS)


def _dedupe_warnings(warnings: Iterable[str]) -> list[str]:
    """One clear warning per issue: drop exact repeats, keep first-seen order."""
    seen: set[str] = set()
    unique: list[str] = []
    for warning in warnings:
        text = " ".join(str(warning).split())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(text)
    return unique



class RecommendationService:
    """Coordinates standards lookup, retrieval and (later) LLM reasoning."""

    def __init__(
        self,
        *,
        pipeline: RecommendationPipeline | None = None,
        standards_service: StandardsService | None = None,
        document_service: DocumentService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._pipeline = pipeline or get_recommendation_pipeline()
        self._standards = standards_service or get_standards_service()
        self._documents = document_service or get_document_service()
        self._settings = settings or get_settings()

    # --- helpers ----------------------------------------------------------
    def pipeline_info(
        self,
        readiness: PipelineReadiness | None = None,
        *,
        ranking_stage: str = AI_UNAVAILABLE_STAGE,
    ) -> PipelineInfo:
        """Describe which AI/RAG components are ready.

        ``ready`` is about *retrieval and ranking*, not about the LLM. The LLM is
        the applicability authority, so its absence is reported as
        ``ai_unavailable`` with no recommendations rather than as a blocker.

        ``ranking_stage`` states who decided. ``ai_reasoning_ranking`` is the only
        value that may accompany final recommendations; every other value means the
        applicability authority did not (or could not) decide, so the response
        carries no recommendation at all.
        """
        readiness = readiness or self._pipeline.readiness()
        llm_provider = get_llm_provider()
        reasons = list(readiness.reasons)
        if not llm_provider.is_configured:
            reasons.append(
                "No LLM provider is configured (LLM_PROVIDER / LLM_MODEL / "
                "LLM_API_KEY). Retrieval still works and produces candidates, but "
                "without an AI applicability verdict no standard can be presented "
                "as a recommendation."
            )
        elif ranking_stage == AI_UNAVAILABLE_STAGE:
            reasons.append(
                "An LLM provider is configured, but it did not complete an "
                "applicability verdict for this request, so no final recommendation "
                "was made."
            )
        return PipelineInfo(
            ready=readiness.ready,
            llm_provider=llm_provider.describe(),
            embedding_provider=readiness.embedding_provider,
            vector_store=readiness.vector_store,
            indexed_chunks=readiness.indexed_chunks,
            standards_available=self._standards.repository.count(),
            ranking_stage=ranking_stage,
            reasons=reasons,
        )

    def _to_evidence(self, chunks: Iterable[RetrievedChunk]) -> list[EvidenceItem]:
        """Map retrieved chunks to the verbatim evidence items the API returns."""
        return [
            EvidenceItem(
                source=_EVIDENCE_SOURCE,
                standard_code=str(chunk.metadata.get("standard_code") or "") or None,
                chunk_id=chunk.chunk_id,
                snippet=chunk.text,
                page_number=chunk.page_number,
                score=chunk.score,
            )
            for chunk in chunks
        ]

    def _retrieve(
        self, query: str, *, readiness: PipelineReadiness, top_k: int | None
    ) -> tuple[list[RetrievedChunk], list[str], bool]:
        """Retrieve real chunks; failures become warnings, never fabrications.

        Returns the chunks, the warnings to surface and whether the *query*
        itself failed. Retrieval is skipped entirely when the pipeline is not
        ready (no embedding provider, no vector store or an empty index), which is
        exactly the case the status logic reports as ``not_configured``; that is
        not a failure, so the third element stays ``False``.

        A pipeline that reports itself ready and then throws (a vector-store or
        embedding fault at query time) is a genuinely different outcome: it must
        not be reported as "the index does not cover this requirement".
        """
        if not readiness.ready:
            return [], [], False
        try:
            chunks = self._pipeline.retrieve(query, top_k=top_k)
        except AppError as exc:
            logger.warning("Retrieval unavailable: %s", exc.message)
            return (
                [],
                [f"Evidence retrieval is unavailable: {exc.message}"],
                True,
            )
        return chunks, [], False

    def _catalogue_has(self, code: str) -> bool:
        """True when ``code`` is a designation in the loaded, validated catalogue."""
        return self._standards.repository.get_by_code(code) is not None

    def _build_retrieved_candidates(
        self, candidates: Iterable[RankedStandard]
    ) -> list[RetrievedCandidate]:
        """Project ranked candidates onto the *candidate* schema.

        This is the deterministic layer's only output shape. It deliberately
        carries no applicability confidence, no Direct/Related class and no
        generated justification: it reports *that* a catalogue record was
        retrieved and the supporting retrieval signals, which is exactly the
        support role retrieval is allowed to play.
        """
        retrieved: list[RetrievedCandidate] = []
        for candidate in candidates:
            entity = self._standards.repository.get_by_code(candidate.standard_number)
            if entity is None:
                # Defensive: the ranker already filters on the catalogue, and a
                # standard that is not in it is never recommended.
                logger.warning(
                    "Skipping '%s': retrieved but absent from the catalogue.",
                    candidate.standard_number,
                )
                continue
            record = to_standard_record(entity)
            retrieved.append(
                RetrievedCandidate(
                    standard_number=record.code,
                    title=record.title,
                    category=record.category,
                    retrieval_rank=candidate.rank,
                    similarity_score=candidate.retrieval_score,
                    attribute_score=candidate.attribute_score,
                    matched_attributes=list(candidate.matched_attributes),
                )
            )
        return retrieved

    def _build_reasoning_candidates(
        self, candidates: Iterable[RankedStandard]
    ) -> list[CandidateStandard]:
        """Project ranked candidates onto the payload the reasoning layer needs.

        Catalogue fields come from the verified repository (never from the
        model); the deterministic rank/similarity/attribute matches travel as
        *signals*, explicitly labelled as such for the model.
        """
        reasoning_candidates: list[CandidateStandard] = []
        for candidate in candidates:
            entity = self._standards.repository.get_by_code(candidate.standard_number)
            if entity is None:
                continue
            record = to_standard_record(entity)
            version = record.version
            certification = tuple(
                " ".join(
                    part for part in (item.scheme, item.marking, item.notes) if part
                )
                for item in record.certification
            )
            related = tuple(
                " ".join(
                    part for part in (item.code, item.title, item.relation) if part
                )
                for item in record.references
            )
            # ``references`` are structured cross-references; ``related_standards``
            # are the plain designations the dataset states this standard relates
            # to. Both are catalogue facts, so both are offered to the model.
            related = related + tuple(record.related_standards)
            reasoning_candidates.append(
                CandidateStandard(
                    standard_number=record.code,
                    title=record.title,
                    scope=record.scope,
                    category=record.category,
                    status=version.status if version else None,
                    year=version.year if version else None,
                    revision=version.revision if version else None,
                    amendments=tuple(version.amendments) if version else (),
                    certification=certification,
                    related_standards=related,
                    deterministic_rank=candidate.rank,
                    similarity_score=candidate.retrieval_score,
                    attribute_score=candidate.attribute_score,
                    matched_attributes=tuple(candidate.matched_attributes),
                    unmatched_attributes=tuple(candidate.unmatched_attributes),
                    matched_terms=tuple(candidate.matched_terms),
                    evidence=tuple(
                        CandidateEvidence(
                            chunk_id=chunk.chunk_id,
                            snippet=chunk.text,
                            page_number=chunk.page_number,
                            score=chunk.score,
                        )
                        for chunk in candidate.evidence
                    ),
                )
            )
        return reasoning_candidates

    def _apply_reasoning(
        self,
        verdict: list[ReasonedStandard],
        candidates: list[RankedStandard],
    ) -> list[RecommendedStandard]:
        """Turn the LLM verdict into the final recommendations.

        The split of responsibility is strict:

        * the **model** supplies order, applicability classification, confidence,
          the justification, matched/uncovered aspects and (optionally) which of
          its own evidence chunks it relied on;
        * the **verified catalogue** supplies the designation, title, version,
          certification and references - never the model.

        Only designations in the verified candidate set can appear: the
        reasoning layer already dropped invented ones, and this method re-checks
        against the catalogue so grounding holds even if that layer is bypassed.
        """
        merged: list[RecommendedStandard] = []
        for item in verdict:
            entity = self._standards.repository.get_by_code(item.standard_number)
            if entity is None:
                # Never emit a designation the verified catalogue does not hold.
                logger.warning(
                    "AI verdict named '%s', which is not in the catalogue; dropped.",
                    item.standard_number,
                )
                continue
            record = to_standard_record(entity)
            evidence = self._to_evidence(
                self._evidence_for(item.standard_number, candidates)
            )
            if item.evidence_chunk_ids:
                wanted = set(item.evidence_chunk_ids)
                cited = [chunk for chunk in evidence if chunk.chunk_id in wanted]
                if cited:
                    evidence = cited
            merged.append(
                RecommendedStandard(
                    code=record.code,
                    title=record.title,
                    confidence=item.confidence,
                    category=record.category,
                    reasoning=item.why_it_matches,
                    applicability_type=item.applicability_type,
                    matched_requirements=list(item.matched_requirements),
                    uncovered_requirements=list(item.uncovered_requirements),
                    version=record.version,
                    certification=record.certification,
                    related_standards=record.references,
                    evidence=evidence,
                    # Kept so the deterministic order stays visible after the
                    # reasoning layer reorders the list.
                    deterministic_rank=self._retrieval_rank(
                        item.standard_number, candidates
                    ),
                )
            )
        return merged

    @staticmethod
    def _evidence_for(standard_number, candidates):
        """Retrieved chunks backing one candidate, looked up by designation."""
        key = normalize_designation(standard_number)
        for candidate in candidates:
            if normalize_designation(candidate.standard_number) == key:
                return list(candidate.evidence)
        return []

    @staticmethod
    def _retrieval_rank(standard_number, candidates):
        """Deterministic retrieval position, for the AI-vs-retrieval comparison."""
        key = normalize_designation(standard_number)
        for candidate in candidates:
            if normalize_designation(candidate.standard_number) == key:
                return candidate.rank
        return None

    def _run_ai_reasoning(
        self,
        requirement: str,
        ranked_candidates: list[RankedStandard],
    ) -> ReasoningOutcome | None:
        """Run the LLM reasoning layer, ``None`` when it cannot/should not run.

        ``None`` means the stage was skipped entirely: no LLM configured, or no
        verified candidate to reason about. A provider failure instead returns an
        outcome carrying ``error`` and no recommendations, so the response can say
        *why* the analysis could not be completed (rate limit, provider fault,
        unparseable verdict) instead of silently degrading to retrieval output.
        """
        if not ranked_candidates:
            return None
        provider = get_llm_provider()
        if not provider.is_configured:
            return None
        candidates = self._build_reasoning_candidates(ranked_candidates)
        if not candidates:
            return None
        attributes = [
            {"kind": attribute.kind, "value": attribute.value}
            for attribute in extract_technical_attributes(requirement)
        ]
        try:
            outcome = run_ai_reasoning(
                provider=provider,
                requirement=requirement,
                attributes=attributes,
                candidates=candidates[:REASONING_MAX_CANDIDATES],
                temperature=self._settings.llm_temperature,
                max_tokens=max(self._settings.llm_max_tokens, 2000),
            )
        except AppError as exc:
            # Provider-level failure (rate limit, timeout, HTTP error). The
            # applicability authority did not rule, so NO final recommendation
            # may be made; the candidates stay candidates.
            logger.warning("AI applicability reasoning unavailable: %s", exc)
            return ReasoningOutcome(
                provider=provider.provider_name, error=exc.message
            )
        except Exception as exc:  # noqa: BLE001 - the endpoint must always survive
            # Any unexpected provider-level failure (transport, SDK, payload)
            # must leave the response without a final recommendation, never take
            # down the endpoint and never silently promote retrieval output.
            logger.warning(
                "AI reasoning failed unexpectedly (%s: %s); no recommendation made.",
                type(exc).__name__,
                exc,
            )
            return ReasoningOutcome(
                provider=provider.provider_name,
                error=f"{type(exc).__name__}: {exc}",
            )

        if outcome.truncated:
            # The verdict was cut off by the token limit, so the candidates it
            # never mentioned look "excluded" only because it never reached them.
            # A partial verdict is not a verdict: discard it entirely.
            logger.warning(
                "AI reasoning response was truncated by the token limit "
                "(usage=%s); no final recommendation will be made.",
                outcome.usage,
            )
            return ReasoningOutcome(
                provider=outcome.provider,
                model=outcome.model,
                truncated=True,
                usage=outcome.usage,
                error=(
                    "the model response was cut off by the token limit before it "
                    "finished the verdict, so the partial answer was discarded "
                    "rather than used to exclude standards it never reached"
                ),
            )
        return outcome

    def _advisories(
        self,
        outcome: RankingOutcome,
        candidates: list[RankedStandard],
        *,
        retrieval_available: bool,
        reasoning: ReasoningOutcome | None = None,
        ai_ranked: bool = False,
        llm_configured: bool = True,
    ) -> list[str]:
        """Explain the run, one clear warning per distinct issue."""
        warnings: list[str] = []

        if outcome.unmatchable_kinds:
            names = ", ".join(KIND_LABELS.get(k, k) for k in outcome.unmatchable_kinds)
            warnings.append(
                f"No retrieved record states a value for: {names}. Those parts of the "
                "requirement could not be matched against this catalogue and therefore "
                "did not influence candidate retrieval."
            )

        if reasoning is not None and reasoning.dropped:
            dropped = ", ".join(sorted(set(reasoning.dropped))[:5])
            warnings.append(
                f"The AI response named {len(reasoning.dropped)} designation(s) outside "
                f"the verified candidate set; they were discarded: {dropped}."
            )

        if ai_ranked and reasoning is not None and reasoning.excluded:
            excluded = ", ".join(sorted(set(reasoning.excluded))[:5])
            warnings.append(
                f"AI applicability reasoning excluded {len(reasoning.excluded)} verified "
                f"candidate(s) as out of scope for this requirement: {excluded}."
            )

        if ai_ranked:
            warnings.append(
                "The final applicability classification, ordering and justifications were "
                "made by AI reasoning over the retrieved evidence. Retrieval ranks and "
                "similarity scores are supporting signals only. Nothing outside the "
                "verified catalogue was proposed."
            )
            return _dedupe_warnings(warnings)

        if not retrieval_available:
            return _dedupe_warnings(warnings)

        if outcome.groups_seen == 0:
            warnings.append(
                "The retrieved chunks carried no catalogue designation, so no verified "
                "candidate could be produced."
            )
            return _dedupe_warnings(warnings)

        if outcome.unverified:
            listed = ", ".join(sorted(outcome.unverified)[:5])
            warnings.append(
                f"{len(outcome.unverified)} retrieved designation(s) are not in the loaded "
                f"catalogue and were discarded: {listed}."
            )
        if outcome.below_floor:
            warnings.append(
                f"{len(outcome.below_floor)} retrieved standard(s) scored below the "
                f"relevance floor of {outcome.min_score:.2f} and were not passed to AI "
                "reasoning as candidates."
            )
        if not candidates and not warnings:
            warnings.append(
                "No catalogue-verified candidate was retrieved for this requirement, so "
                "AI applicability reasoning had nothing to assess."
            )
        return _dedupe_warnings(warnings)

    def _determine_status(
        self,
        readiness: PipelineReadiness,
        info: PipelineInfo,
        recommendations: list[RecommendedStandard],
        *,
        candidates: list[RankedStandard],
        reasoning: ReasoningOutcome | None,
        llm_configured: bool,
        retrieval_failed: bool = False,
    ) -> tuple[RecommendationStatus, str, NoResultReason | None]:
        """Decide the response status from what actually happened.

        Also returns *why* the list is empty, so the genuinely different
        "nothing to show" outcomes are never collapsed into one message:

        * ``dataset_unavailable`` - nothing is indexed *and* no catalogue is loaded.
        * ``not_configured``     - retrieval itself cannot run (provider/store/index).
        * ``retrieval_unavailable`` - retrieval *is* configured but the query
          failed, so nothing could be retrieved or evaluated at all.
        * ``ai_unavailable``     - retrieval worked but the applicability authority
          did not rule, so no final recommendation is made.
        * ``ok``                 - AI reasoning produced grounded recommendations.
        * ``placeholder``        - either AI ruled that nothing applies, or no
          verified candidate existed.

        A configured-but-failing LLM is deliberately *not* reported as
        ``not_configured`` (retrieval genuinely worked); it is reported as
        ``ai_unavailable`` so the UI says "reasoning unavailable" instead of
        quietly presenting retrieval output as recommendations.
        """
        if info.standards_available == 0 and readiness.indexed_chunks == 0:
            return (
                RecommendationStatus.DATASET_UNAVAILABLE,
                _DATASET_MESSAGE,
                NoResultReason.RETRIEVAL_UNAVAILABLE,
            )
        if not readiness.ready:
            return (
                RecommendationStatus.NOT_CONFIGURED,
                _NOT_CONFIGURED_MESSAGE,
                NoResultReason.RETRIEVAL_UNAVAILABLE,
            )
        if retrieval_failed:
            # The pipeline is ready and the catalogue is loaded, but the query
            # itself failed. That is neither a coverage statement nor an AI
            # verdict, so it gets its own reason and its own wording.
            return (
                RecommendationStatus.PLACEHOLDER,
                _RETRIEVAL_FAILED_MESSAGE,
                NoResultReason.RETRIEVAL_UNAVAILABLE,
            )
        if recommendations:
            return (
                RecommendationStatus.OK,
                _ok_message(len(recommendations), ai_ranked=True),
                None,
            )
        if not candidates:
            # Retrieval produced nothing verifiable for this requirement. This is
            # a statement about the coverage of the indexed knowledge base, not
            # about the model, and emphatically not about whether an Indian
            # Standard exists.
            return (
                RecommendationStatus.PLACEHOLDER,
                _PLACEHOLDER_MESSAGE,
                NoResultReason.INSUFFICIENT_COVERAGE,
            )
        if not llm_configured:
            return (
                RecommendationStatus.AI_UNAVAILABLE,
                _AI_NOT_CONFIGURED_MESSAGE,
                NoResultReason.AI_UNAVAILABLE,
            )
        if reasoning is not None and reasoning.error:
            if _is_rate_limit(reasoning.error):
                return (
                    RecommendationStatus.AI_UNAVAILABLE,
                    _AI_RATE_LIMIT_MESSAGE,
                    NoResultReason.AI_UNAVAILABLE,
                )
            return (
                RecommendationStatus.AI_UNAVAILABLE,
                _AI_UNAVAILABLE_MESSAGE,
                NoResultReason.AI_UNAVAILABLE,
            )
        if reasoning is not None and not reasoning.recommendations:
            # The model ruled, and excluded every candidate it was shown. This is
            # a considered answer, not a coverage failure.
            return (
                RecommendationStatus.PLACEHOLDER,
                _AI_EXCLUDED_ALL_MESSAGE,
                NoResultReason.AI_RULED_NONE,
            )
        return (
            RecommendationStatus.AI_UNAVAILABLE,
            _AI_UNAVAILABLE_MESSAGE,
            NoResultReason.AI_UNAVAILABLE,
        )

    # --- entry point -------------------------------------------------------
    def analyze(self, request: RecommendationRequest) -> RecommendationResponse:
        """Analyse a requirement and return a clearly marked response.

        The deterministic ranking stage always runs first as the candidate and
        signal layer. When an LLM provider is configured, the verified
        candidates and their evidence are handed to the reasoning layer, whose
        verdict becomes the final applicability classification and ranking.
        """
        requirement = request.requirement.strip()
        warnings: list[str] = []
        document_text = ""

        if request.document_id:
            try:
                document_text = self._documents.get_processed_text(request.document_id)
            except AppError as exc:
                warnings.append(exc.message)

        readiness = self._pipeline.readiness()

        # The requirement plus (when supplied) the uploaded document is what gets
        # retrieved and term-matched against.
        query = "\n\n".join(part for part in (requirement, document_text[:4000]) if part)
        chunks, retrieval_warnings, retrieval_failed = self._retrieve(
            query, readiness=readiness, top_k=request.top_k
        )
        warnings.extend(retrieval_warnings)

        outcome = rank_standards(
            chunks,
            requirement=query,
            catalogue_contains=self._catalogue_has,
        )
        candidates = list(outcome.candidates)
        retrieved_candidates = self._build_retrieved_candidates(candidates)

        # ---- AI/LLM reasoning: the ONLY applicability authority ---------------
        reasoning = self._run_ai_reasoning(query, candidates)
        llm_configured = get_llm_provider().is_configured

        recommendations: list[RecommendedStandard] = []
        if reasoning is not None and reasoning.recommendations and not reasoning.error:
            recommendations = self._apply_reasoning(
                reasoning.recommendations, candidates
            )

        if not recommendations:
            # The single invariant of this architecture: a non-empty
            # recommendations list requires ranking_stage == ai_reasoning_ranking.
            recommendations = []

        if recommendations:
            ranking_stage = AI_RANKING_STAGE
        else:
            ranking_stage = AI_UNAVAILABLE_STAGE

        info = self.pipeline_info(readiness, ranking_stage=ranking_stage)

        understanding = self._requirement_understanding(requirement, outcome)
        ai_info = self._ai_reasoning_info(reasoning, candidates, llm_configured)

        status, message, no_result_reason = self._determine_status(
            readiness,
            info,
            recommendations,
            candidates=candidates,
            reasoning=reasoning,
            llm_configured=llm_configured,
            retrieval_failed=retrieval_failed,
        )
        warnings.extend(
            self._advisories(
                outcome,
                candidates,
                # A failed query is not an available retrieval: the "no catalogue
                # designation was retrieved" story belongs to a *successful*
                # retrieval that returned unusable chunks.
                retrieval_available=readiness.ready and not retrieval_failed,
                reasoning=reasoning,
                ai_ranked=bool(recommendations),
                llm_configured=llm_configured,
            )
        )
        warnings.extend(self._ai_warnings(reasoning, candidates, llm_configured))

        return RecommendationResponse(
            status=status,
            message=message,
            requirement=requirement,
            document_id=request.document_id,
            recommendations=recommendations,
            retrieved_candidates=retrieved_candidates,
            retrieved_evidence=self._to_evidence(chunks),
            requirement_understanding=understanding,
            ai_reasoning=ai_info,
            # The model's own statement of what is being procured, surfaced so the
            # applicability decision can be inspected against the same target.
            procurement_target=(
                ProcurementTargetInfo(
                    target=reasoning.procurement_target.target,
                    explicit_specifications=list(
                        reasoning.procurement_target.explicit_specifications
                    ),
                    note=reasoning.procurement_target.note or None,
                )
                if reasoning is not None and reasoning.procurement_target is not None
                else None
            ),
            pipeline=info,
            warnings=_dedupe_warnings(warnings),
            no_result_reason=no_result_reason,
            # Coverage is the limiting factor exactly when there were no
            # candidates to evaluate; stating the local scope is the honest
            # thing to do and never implies BIS incompleteness.
            coverage_note=(
                _COVERAGE_NOTE
                if no_result_reason is NoResultReason.INSUFFICIENT_COVERAGE
                else None
            ),
        )

    # --- requirement understanding & AI reporting ---------------------------
    @staticmethod
    def _requirement_understanding(
        requirement: str, outcome: RankingOutcome
    ) -> RequirementUnderstanding:
        """Describe what was read out of the requirement text.

        Supporting information only: these attributes are shown for transparency
        and are handed to the model as context. They never decide applicability -
        that stays exclusively with the AI reasoning stage.
        """
        detected = [
            DetectedAttribute(
                kind=attribute.kind,
                label=attribute.label,
                surface=attribute.surface,
                value=attribute.value,
            )
            for attribute in extract_technical_attributes(requirement)
        ]
        return RequirementUnderstanding(
            detected=detected,
            unresolved_families=[
                KIND_LABELS.get(kind, kind) for kind in outcome.unmatchable_kinds
            ],
        )

    @staticmethod
    def _ai_reasoning_info(
        reasoning: ReasoningOutcome | None,
        candidates: list[RankedStandard],
        llm_configured: bool,
    ) -> AIReasoningInfo:
        """Diagnostics for the reasoning stage (surfaced under "Technical details")."""
        supplied = len(candidates[:REASONING_MAX_CANDIDATES])
        if reasoning is None:
            return AIReasoningInfo(
                status="not_configured" if not llm_configured else "unavailable",
                candidates_supplied=supplied,
            )
        # A successful run that selects nothing is still a completed verdict: the
        # model ruled that none of the verified candidates applies. It is only
        # "unavailable" when the stage genuinely failed or produced nothing
        # readable.
        completed = reasoning.error is None and not reasoning.truncated
        status = "completed" if completed else "unavailable"
        return AIReasoningInfo(
            status=status,
            provider=reasoning.provider or None,
            model=reasoning.model,
            candidates_supplied=supplied,
            candidates_selected=len(reasoning.recommendations),
            candidates_excluded=list(reasoning.excluded),
            invented_dropped=list(reasoning.dropped),
            truncated=reasoning.truncated,
            error=reasoning.error,
        )

    @staticmethod
    def _ai_warnings(
        reasoning: ReasoningOutcome | None,
        candidates: list[RankedStandard],
        llm_configured: bool,
    ) -> list[str]:
        """One clear warning per AI-stage issue - never a silent fallback."""
        if not candidates:
            return []
        if not llm_configured:
            return [
                "No AI analysis service is configured, so we could not determine "
                "which standards apply. The potential matches below are search "
                "results only, and nothing is presented as applicable."
            ]
        if reasoning is None:
            return [
                "AI applicability analysis could not be completed, so we could not "
                "safely determine which standards apply. Nothing is presented as "
                "applicable."
            ]
        if reasoning.truncated:
            return [
                "AI applicability analysis could not be completed because the "
                "response was cut short, so no standard is presented as applicable."
            ]
        if reasoning.error:
            if _is_rate_limit(reasoning.error):
                # The raw provider error is deliberately not repeated here: the
                # service already exposes it via `ai_reasoning.error` for the
                # "Technical details" panel.
                return [
                    "AI applicability analysis could not be completed because the "
                    "analysis service reached its usage limit. No standard is "
                    "presented as applicable — please try again shortly."
                ]
            return [
                "AI applicability analysis could not be completed. The potential "
                "matches below are search results only, not a decision, and "
                "nothing is presented as applicable."
            ]
        if not reasoning.recommendations:
            return [
                "AI applicability analysis reviewed the standards found in the "
                "catalogue and identified no applicable standard for this requirement."
            ]
        return []


def get_recommendation_service() -> RecommendationService:
    """Return a recommendation service wired to the configured components."""
    return RecommendationService()
