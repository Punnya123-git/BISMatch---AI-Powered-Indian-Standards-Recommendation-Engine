"""Tests for the AI reasoning stage - the final applicability + ranking layer.

The deterministic ranker (:mod:`app.rag.ranking`) keeps generating and filtering
candidates; these tests pin down that the LLM verdict - and only the LLM verdict -
decides the final order, classification and justification, that the model can
only ever return designations from the verified candidate set, and that
everything degrades to the clearly labelled deterministic fallback when no
provider is configured or when the call fails.

No network is involved: a clearly-labelled stub provider is wired in through the
real ``register_llm_provider`` factory hook, exactly the way a production
provider is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pytest

from app.ai.llm.base import LLMProvider, LLMRequest, LLMResponse
from app.ai.llm.factory import register_llm_provider, reset_llm_provider_cache
from app.ai.prompts.templates import STANDARD_RECOMMENDATION_SYSTEM_PROMPT
from app.ai.prompts.templates import (
    STANDARD_RECOMMENDATION_PROMPT,
    STANDARD_RECOMMENDATION_SYSTEM_PROMPT,
)
from app.ai.reasoning import (
    CandidateEvidence,
    CandidateStandard,
    normalize_designation,
    parse_reasoning_response,
    run_ai_reasoning,
)
from app.core.config import get_settings
from app.core.exceptions import LLMProviderError, LLMRateLimitError, VectorStoreError
from app.database.repositories import InMemoryStandardRepository
from app.rag.pipeline import PipelineReadiness
from app.rag.retrieval import RetrievedChunk
from app.schemas.recommendation import (
    NoResultReason,
    RecommendationRequest,
    RecommendationStatus,
)
from app.services.recommendation_service import (
    AI_RANKING_STAGE,
    AI_UNAVAILABLE_STAGE,
    RecommendationService,
)
from app.services.standards_service import StandardsService
from app.standards.dataset_loader import StandardsDatasetLoader
from app.standards.documents import build_standard_document

_ENRICHED_DATASET = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "standards"
    / "standards_enriched_bis_verified.json"
)

#: Real designations from the shipped catalogue.
_EFFICIENCY_MOTORS = "IS 12615:2018"  # IE efficiency classes
_PUMP_MOTORS = "IS 7538:1996"  # squirrel cage motors for centrifugal pumps
_NOISE = "IS 12065:2025"

_MOTOR_REQUIREMENT = (
    "Supply of 15 kW IE3 three-phase squirrel cage induction motor, "
    "foot mounted, IP55 enclosure, 415 V, 50 Hz"
)

_INVENTED = "IS 99999:2030"


# --- test double ------------------------------------------------------------
class StubReasoningProvider(LLMProvider):
    """Deterministic, test-only provider: a scripted verdict, no network.

    Class-level state keeps the scripted response and the received requests
    inspectable from the tests without threading fixtures through the provider
    constructor (which the factory calls with ``Settings``).
    """

    provider_name = "stub-reasoning"

    #: Pristine state, restored before *and* after every test that uses the stub
    #: so one test's response metadata can never leak into the next one.
    DEFAULTS = {
        "response_text": "{}",
        "raises": None,
        "finish_reason": "stop",
    }

    #: Verbatim text handed back by :meth:`generate`.
    response_text: str = "{}"
    #: When set, :meth:`generate` raises it instead of answering.
    raises: BaseException | None = None
    #: ``finish_reason`` reported back, e.g. ``"length"`` for a truncated answer.
    finish_reason: str | None = "stop"
    #: Every request the service sent, for prompt/payload assertions.
    requests: list[LLMRequest] = []

    @classmethod
    def reset(cls) -> None:
        """Restore every scripted attribute to its default value."""
        cls.response_text = cls.DEFAULTS["response_text"]
        cls.raises = cls.DEFAULTS["raises"]
        cls.finish_reason = cls.DEFAULTS["finish_reason"]
        cls.requests = []

    def __init__(self, settings: Any = None) -> None:
        self.settings = settings

    @property
    def is_configured(self) -> bool:
        return True

    @property
    def model(self) -> str:
        return "stub-reasoning-model"

    def generate(self, request: LLMRequest) -> LLMResponse:
        cls = type(self)
        cls.requests.append(request)
        if cls.raises is not None:
            raise cls.raises
        return LLMResponse(
            text=cls.response_text,
            provider=self.provider_name,
            model=self.model,
            finish_reason=cls.finish_reason,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        )


@pytest.fixture
def stub_llm(monkeypatch: pytest.MonkeyPatch):
    """Make the stub *the* configured provider for the duration of one test.

    The scripted state is reset on entry *and* on exit, so ``finish_reason``,
    ``raises`` and ``response_text`` set by one test can never influence another,
    whatever order the tests run in.
    """
    register_llm_provider(StubReasoningProvider.provider_name, StubReasoningProvider)
    StubReasoningProvider.reset()
    monkeypatch.setenv("LLM_PROVIDER", StubReasoningProvider.provider_name)
    get_settings.cache_clear()
    reset_llm_provider_cache()
    yield StubReasoningProvider
    StubReasoningProvider.reset()
    get_settings.cache_clear()
    reset_llm_provider_cache()


# --- catalogue / retrieval fixtures -----------------------------------------
@pytest.fixture(scope="module")
def dataset_records() -> dict[str, Any]:
    """The shipped (enriched BIS) catalogue records, keyed by designation."""
    dataset = StandardsDatasetLoader(path=_ENRICHED_DATASET).load()
    return {record.standard_number: record for record in dataset.records}


@pytest.fixture(scope="module")
def catalogue() -> StandardsService:
    """Standards service backed by the shipped catalogue and a fresh repository."""
    service = StandardsService(
        repository=InMemoryStandardRepository(),
        loader=StandardsDatasetLoader(path=_ENRICHED_DATASET),
    )
    service.ensure_loaded()
    return service


class _StubPipeline:
    """Stand-in for ``RecommendationPipeline`` returning prepared chunks.

    ``raises`` models a *runtime* retrieval failure: the pipeline reports itself
    as ready (provider, store and index all present) and then the query itself
    fails, which is a different fact from "the index does not cover this".
    """

    def __init__(
        self,
        *,
        ready: bool,
        chunks: Sequence[RetrievedChunk] = (),
        indexed_chunks: int | None = None,
        reasons: Iterable[str] = (),
        raises: BaseException | None = None,
    ) -> None:
        self._ready = ready
        self._chunks = list(chunks)
        self._indexed = len(self._chunks) if indexed_chunks is None else indexed_chunks
        self._reasons = list(reasons)
        self._raises = raises

    def readiness(self) -> PipelineReadiness:
        return PipelineReadiness(
            ready=self._ready,
            embedding_provider="stub (configured, model=test)",
            vector_store="chroma",
            indexed_chunks=self._indexed,
            reasons=self._reasons,
        )

    def retrieve(self, query: str, *, top_k: int | None = None, where=None):
        if self._raises is not None:
            raise self._raises
        return list(self._chunks)


def _retrieved(record: Any, score: float, *, chunk_index: int = 0) -> RetrievedChunk:
    """A retrieved chunk carrying the real indexed text of ``record``."""
    document = build_standard_document(record)
    metadata = dict(document.metadata)
    metadata.update({"source_id": document.source_id, "chunk_index": chunk_index})
    return RetrievedChunk(
        chunk_id=f"{document.source_id}::{chunk_index}",
        text=document.text,
        score=score,
        source_id=document.source_id,
        metadata=metadata,
    )


def _service(
    catalogue: StandardsService,
    *,
    ready: bool = True,
    chunks: Sequence[RetrievedChunk] = (),
    indexed_chunks: int | None = None,
    raises: BaseException | None = None,
) -> RecommendationService:
    return RecommendationService(
        pipeline=_StubPipeline(
            ready=ready,
            chunks=chunks,
            indexed_chunks=indexed_chunks,
            raises=raises,
        ),
        standards_service=catalogue,
    )


def _motor_service(catalogue: StandardsService, dataset_records: dict[str, Any]):
    """The IE3 motor scenario: deterministic order is 12615, then 7538."""
    return _service(
        catalogue,
        chunks=[
            _retrieved(dataset_records[_EFFICIENCY_MOTORS], 0.75),
            _retrieved(dataset_records[_PUMP_MOTORS], 0.45),
        ],
    )


# --- candidate / verdict builders -------------------------------------------
def _candidate(
    number: str,
    *,
    rank: int = 1,
    chunk_ids: Sequence[str] = ("chunk-0",),
) -> CandidateStandard:
    return CandidateStandard(
        standard_number=number,
        title=f"{number} catalogue title",
        scope=f"{number} catalogue scope",
        category="rotating electrical machines",
        deterministic_rank=rank,
        similarity_score=round(0.9 - rank / 10, 4),
        matched_attributes=("IE3",),
        evidence=tuple(
            CandidateEvidence(chunk_id=cid, snippet=f"indexed text of {cid}")
            for cid in chunk_ids
        ),
    )


def _item(
    number: str,
    *,
    applicability: str = "direct",
    confidence: float = 0.9,
    why: str = "Because the scope covers the requested product.",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "standard_number": number,
        "applicability_type": applicability,
        "confidence": confidence,
        "why_it_matches": why,
        "matched_requirements": ["efficiency class IE3"],
        "uncovered_requirements": [],
        **extra,
    }


def _verdict(*items: dict[str, Any], target: str | None = None) -> str:
    """Build a model verdict. ``target`` sets the procurement-target block."""
    body: dict[str, Any] = {"recommendations": list(items)}
    if target is not None:
        body["procurement_target"] = {
            "target": target,
            "explicit_specifications": [],
            "note": "",
        }
    return json.dumps(body)


# --- parsing / grounding rules ---------------------------------------------
def test_normalize_designation_folds_separators_and_case() -> None:
    assert normalize_designation("IS-12615:2018") == normalize_designation(
        "is 12615 2018"
    )


def test_parse_keeps_only_designations_from_the_candidate_set() -> None:
    candidates = [_candidate(_EFFICIENCY_MOTORS), _candidate(_PUMP_MOTORS, rank=2)]

    outcome = parse_reasoning_response(
        _verdict(_item(_PUMP_MOTORS), _item(_INVENTED), _item("IS 12345:2020")),
        candidates,
    )

    assert [r.standard_number for r in outcome.recommendations] == [_PUMP_MOTORS]
    # Invented designations are reported, never silently swallowed.
    assert sorted(outcome.dropped) == sorted([_INVENTED, "IS 12345:2020"])
    assert outcome.error is None


def test_parse_accepts_a_reformatted_but_real_designation() -> None:
    candidates = [_candidate(_EFFICIENCY_MOTORS)]

    outcome = parse_reasoning_response(_verdict(_item("is-12615-2018")), candidates)

    assert [r.standard_number for r in outcome.recommendations] == [
        _EFFICIENCY_MOTORS
    ]
    assert outcome.dropped == []


def test_parse_reports_an_error_when_the_verdict_is_not_json() -> None:
    candidates = [_candidate(_EFFICIENCY_MOTORS)]

    outcome = parse_reasoning_response(
        "I think IS 12615:2018 is probably the right one.", candidates
    )

    assert outcome.recommendations == []
    assert outcome.error is not None
    assert "JSON" in outcome.error


def test_parse_reports_an_explicit_empty_selection_as_a_verdict_not_an_error() -> None:
    """An empty list is the model saying "none of these apply", not a failure."""
    candidates = [_candidate(_EFFICIENCY_MOTORS)]

    outcome = parse_reasoning_response(json.dumps({"recommendations": []}), candidates)

    assert outcome.recommendations == []
    assert outcome.error is None
    # Every candidate is reported as considered-and-excluded.
    assert outcome.excluded == [_EFFICIENCY_MOTORS]


def test_parse_reports_an_error_when_the_verdict_has_no_recommendations_key() -> None:
    candidates = [_candidate(_EFFICIENCY_MOTORS)]

    outcome = parse_reasoning_response(json.dumps({"notes": "nothing found"}), candidates)

    assert outcome.recommendations == []
    assert outcome.error is not None
    assert "recommendations" in outcome.error


def test_unknown_applicability_degrades_to_needs_verification() -> None:
    candidates = [_candidate(_EFFICIENCY_MOTORS)]

    outcome = parse_reasoning_response(
        _verdict(_item(_EFFICIENCY_MOTORS, applicability="probably-fine",
                       confidence=0.99)),
        candidates,
    )

    verdict = outcome.recommendations[0]
    assert verdict.applicability_type == "needs_verification"
    # needs_verification is never allowed to look confident.
    assert verdict.confidence <= 0.6


def test_needs_verification_confidence_is_capped() -> None:
    candidates = [_candidate(_EFFICIENCY_MOTORS)]

    outcome = parse_reasoning_response(
        _verdict(
            _item(
                _EFFICIENCY_MOTORS,
                applicability="needs_verification",
                confidence=0.97,
                why="Insufficient evidence to determine applicability; "
                "verification is required.",
            )
        ),
        candidates,
    )

    verdict = outcome.recommendations[0]
    assert verdict.confidence <= 0.6
    assert "verification is required" in verdict.why_it_matches


def test_evidence_ids_are_restricted_to_the_candidate_own_chunks() -> None:
    candidates = [
        _candidate(_EFFICIENCY_MOTORS, chunk_ids=("own-1", "own-2")),
        _candidate(_PUMP_MOTORS, rank=2, chunk_ids=("other-1",)),
    ]

    outcome = parse_reasoning_response(
        _verdict(
            _item(
                _EFFICIENCY_MOTORS,
                evidence_chunk_ids=["own-1", "other-1", "not-a-real-chunk"],
            )
        ),
        candidates,
    )

    assert outcome.recommendations[0].evidence_chunk_ids == ["own-1"]


def test_parse_records_candidates_the_model_left_out_as_excluded() -> None:
    candidates = [_candidate(_EFFICIENCY_MOTORS), _candidate(_NOISE, rank=2)]

    outcome = parse_reasoning_response(_verdict(_item(_EFFICIENCY_MOTORS)), candidates)

    assert outcome.excluded == [_NOISE]


def test_run_ai_reasoning_sends_the_verified_candidates_and_parses_the_verdict(
    stub_llm: type[StubReasoningProvider],
) -> None:
    stub_llm.response_text = _verdict(_item(_EFFICIENCY_MOTORS, confidence=0.77))
    candidates = [_candidate(_EFFICIENCY_MOTORS), _candidate(_PUMP_MOTORS, rank=2)]

    outcome = run_ai_reasoning(
        provider=StubReasoningProvider(),
        requirement=_MOTOR_REQUIREMENT,
        attributes=[{"kind": "efficiency_class", "value": "IE3"}],
        candidates=candidates,
    )

    request = stub_llm.requests[0]
    assert request.system_prompt == STANDARD_RECOMMENDATION_SYSTEM_PROMPT
    assert _EFFICIENCY_MOTORS in request.prompt
    assert _PUMP_MOTORS in request.prompt
    assert _MOTOR_REQUIREMENT in request.prompt
    assert "efficiency_class: IE3" in request.prompt
    # The deterministic score travels as a labelled signal, not as the verdict.
    assert "deterministic_signals" in request.prompt
    assert "not decisions" in request.prompt

    assert outcome.error is None
    assert [r.standard_number for r in outcome.recommendations] == [
        _EFFICIENCY_MOTORS
    ]
    assert outcome.recommendations[0].confidence == pytest.approx(0.77)
    assert outcome.provider == "stub-reasoning"

# --- service: the LLM verdict is the final layer ----------------------------
def test_ai_verdict_reorders_the_deterministic_candidates(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    # The deterministic stage ranks the efficiency standard first; the model
    # judges the pump-set standard the better match for this specification.
    stub_llm.response_text = _verdict(
        _item(_PUMP_MOTORS, why="Scope directly governs the driven equipment."),
        _item(
            _EFFICIENCY_MOTORS,
            applicability="related",
            why="Normative reference for the efficiency class only.",
        ),
    )
    service = _motor_service(catalogue, dataset_records)

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))

    assert response.status is RecommendationStatus.OK
    # Final order is the model's, not the deterministic one.
    assert [r.code for r in response.recommendations] == [
        _PUMP_MOTORS,
        _EFFICIENCY_MOTORS,
    ]
    # The deterministic position stays attached so both orders stay comparable.
    assert [r.deterministic_rank for r in response.recommendations] == [2, 1]


def test_ai_verdict_replaces_the_deterministic_explanation_and_confidence(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    stub_llm.response_text = _verdict(
        _item(_EFFICIENCY_MOTORS, confidence=0.31, why="Model-authored rationale."),
        _item(
            _PUMP_MOTORS,
            confidence=0.22,
            applicability="needs_verification",
            why="Insufficient evidence to determine applicability; "
            "verification is required.",
        ),
    )
    service = _motor_service(catalogue, dataset_records)

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))
    by_code = {item.code: item for item in response.recommendations}

    assert by_code[_EFFICIENCY_MOTORS].reasoning == "Model-authored rationale."
    assert by_code[_EFFICIENCY_MOTORS].confidence == pytest.approx(0.31)
    assert by_code[_PUMP_MOTORS].applicability_type == "needs_verification"
    assert by_code[_PUMP_MOTORS].confidence <= 0.6
    # Catalogue fields are never taken from the model.
    assert (
        by_code[_EFFICIENCY_MOTORS].title == dataset_records[_EFFICIENCY_MOTORS].title
    )


def test_ai_verdict_can_exclude_an_irrelevant_candidate(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    stub_llm.response_text = _verdict(_item(_EFFICIENCY_MOTORS))
    service = _motor_service(catalogue, dataset_records)

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))

    assert [r.code for r in response.recommendations] == [_EFFICIENCY_MOTORS]
    assert any(_PUMP_MOTORS in w and "excluded" in w for w in response.warnings)


def test_ai_verdict_can_never_introduce_a_standard(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    stub_llm.response_text = _verdict(
        _item(_EFFICIENCY_MOTORS), _item(_INVENTED), _item("IS 4458:2019")
    )
    service = _motor_service(catalogue, dataset_records)

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))
    codes = {r.code for r in response.recommendations}

    assert codes <= set(dataset_records)
    assert _INVENTED not in codes
    assert "IS 4458:2019" not in codes
    assert any("outside the verified candidate set" in w for w in response.warnings)

def test_response_reports_the_ai_ranking_stage_when_the_verdict_is_used(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    stub_llm.response_text = _verdict(_item(_EFFICIENCY_MOTORS))
    service = _motor_service(catalogue, dataset_records)

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))

    assert response.pipeline is not None
    assert response.pipeline.ranking_stage == AI_RANKING_STAGE
    assert "AI applicability reasoning" in response.message
    assert any(
        "AI reasoning over the retrieved evidence" in w for w in response.warnings
    )


def test_needs_verification_is_the_answer_when_evidence_is_insufficient(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    stub_llm.response_text = _verdict(
        _item(
            _EFFICIENCY_MOTORS,
            applicability="needs_verification",
            confidence=0.95,
            why="Insufficient evidence to determine applicability; "
            "verification is required.",
        )
    )
    service = _motor_service(catalogue, dataset_records)

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))
    top = response.recommendations[0]

    assert top.applicability_type == "needs_verification"
    assert top.confidence <= 0.6


# --- service: the AI is the ONLY applicability authority ----------------------
def test_without_an_llm_no_final_recommendation_is_made(
    catalogue: StandardsService, dataset_records: dict[str, Any]
) -> None:
    """No provider configured: candidates are surfaced, nothing is recommended.

    The deterministic retrieval layer is a support layer. It may not become the
    applicability authority, so the response carries zero recommendations.
    """
    service = _motor_service(catalogue, dataset_records)

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))

    assert response.recommendations == []
    assert response.status is RecommendationStatus.AI_UNAVAILABLE
    assert response.pipeline is not None
    assert response.pipeline.ranking_stage == AI_UNAVAILABLE_STAGE
    assert response.pipeline.ranking_stage != AI_RANKING_STAGE
    # Retrieval still happened, and is reported as candidates only.
    assert [c.standard_number for c in response.retrieved_candidates] == [
        _EFFICIENCY_MOTORS,
        _PUMP_MOTORS,
    ]
    # A candidate must not smuggle an applicability decision out.
    for candidate in response.retrieved_candidates:
        payload = candidate.model_dump()
        assert "confidence" not in payload
        assert "applicability_type" not in payload
        assert "reasoning" not in payload
    assert "not configured on this deployment" in response.message
    assert any("No AI analysis service is configured" in w for w in response.warnings)


def test_a_failing_provider_makes_no_final_recommendation(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    stub_llm.raises = RuntimeError("upstream model exploded")
    service = _motor_service(catalogue, dataset_records)

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))

    assert response.recommendations == []
    assert response.status is RecommendationStatus.AI_UNAVAILABLE
    assert response.pipeline is not None
    assert response.pipeline.ranking_stage == AI_UNAVAILABLE_STAGE
    assert response.retrieved_candidates


def test_an_unparseable_verdict_makes_no_final_recommendation(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    stub_llm.response_text = "Sorry, I cannot help with that."
    service = _motor_service(catalogue, dataset_records)

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))

    assert response.recommendations == []
    assert response.pipeline is not None
    assert response.pipeline.ranking_stage == AI_UNAVAILABLE_STAGE
    assert "could not be completed" in response.message


def test_an_empty_verdict_list_makes_no_final_recommendation(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """A well-formed verdict that selects nothing is a decision, not a failure."""
    stub_llm.response_text = _verdict()
    service = _motor_service(catalogue, dataset_records)

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))

    assert response.recommendations == []
    assert response.status is RecommendationStatus.PLACEHOLDER
    assert response.pipeline is not None
    assert response.pipeline.ranking_stage == AI_UNAVAILABLE_STAGE
    assert "No applicable standard identified" in response.message


# --- THE core architecture invariant ----------------------------------------
#: Queries whose correct answer is "nothing in our catalogue applies". Each of
#: these previously produced a false motor recommendation.
_OUT_OF_CATALOGUE_QUERIES = (
    "We need BIS standards applicable to packaged drinking water",
    "Ordinary Portland cement for structural construction",
    "Concrete M40 with compressive strength testing",
    "Galvanised mild steel pipes, 50 mm nominal bore, for water supply",
    "Fire extinguisher for a data centre server room",
    "Yoga mat for a corporate wellness programme",
)


def test_no_deterministic_recommendation_is_ever_produced(
    catalogue: StandardsService, dataset_records: dict[str, Any]
) -> None:
    """The load-bearing rule, independent of which standard is retrieved.

    Retrieval will happily return motor standards for a drinking-water query
    (that is what embeddings do). The deterministic layer must not turn that into
    an applicability decision, however confident the retrieval score.
    """
    service = _motor_service(catalogue, dataset_records)

    for query in _OUT_OF_CATALOGUE_QUERIES:
        response = service.analyze(RecommendationRequest(requirement=query))

        assert response.recommendations == [], (
            f"deterministic output leaked a recommendation for {query!r}"
        )
        assert response.pipeline is not None
        assert response.pipeline.ranking_stage != AI_RANKING_STAGE


def test_packaged_drinking_water_never_returns_a_motor_standard(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """Regression: "packaged drinking water" returned IS 9283:2024 at 36%.

    IS 9283:2024 is "Line Operated A.C. Motors for Submersible Pump Sets" - it
    has nothing to do with drinking water. It appeared because a failed AI call
    exposed the deterministic ranking as if it were a verdict. A provider
    failure now yields no recommendation at all.
    """
    submersible_motors = "IS 9283:2024"
    assert submersible_motors in dataset_records, "the regression record must be real"
    service = _motor_service(catalogue, dataset_records)

    stub_llm.raises = LLMRateLimitError("LLM provider rate limit reached (HTTP 429)")
    response = service.analyze(
        RecommendationRequest(
            requirement="We need BIS standards applicable to packaged drinking water"
        )
    )

    assert response.recommendations == []
    assert submersible_motors not in [r.code for r in response.recommendations]
    assert response.status is RecommendationStatus.AI_UNAVAILABLE
    assert response.pipeline is not None
    assert response.pipeline.ranking_stage == AI_UNAVAILABLE_STAGE
    # Part 1/13: the user-facing message must stay free of infrastructure detail.
    assert "usage limit" in response.message
    for banned in ("gpt-oss", "onnx", "chroma", "TPM", "organization"):
        assert banned not in response.message
    assert all(banned not in w for w in response.warnings for banned in ("gpt-oss", "TPM"))
    # The raw detail is still available for the "Technical details" panel.
    assert response.ai_reasoning is not None
    assert response.ai_reasoning.error


def test_an_ai_verdict_may_say_nothing_applies_for_drinking_water(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """When the AI *does* rule and finds nothing applicable, that is the answer."""
    stub_llm.response_text = _verdict()  # explicit empty selection
    service = _motor_service(catalogue, dataset_records)

    response = service.analyze(
        RecommendationRequest(
            requirement="We need BIS standards applicable to packaged drinking water"
        )
    )

    assert response.recommendations == []
    assert "IS 9283:2024" not in [r.code for r in response.recommendations]
    # An AI decision ("none apply"), reported distinctly from a provider failure.
    assert response.status is RecommendationStatus.PLACEHOLDER
    assert "No applicable standard identified" in response.message
    assert response.ai_reasoning is not None
    assert response.ai_reasoning.status == "completed"
    assert response.ai_reasoning.candidates_selected == 0


def test_no_verified_candidate_yields_a_coverage_message(
    catalogue: StandardsService, dataset_records: dict[str, Any]
) -> None:
    """Nothing retrieved above the floor: a coverage statement, not a guess.

    The wording must never imply that no Indian Standard exists - only that the
    *local* verified index does not cover this requirement.
    """
    service = _service(
        catalogue,
        chunks=[_retrieved(dataset_records[_EFFICIENCY_MOTORS], 0.04)],
        indexed_chunks=1,
    )

    response = service.analyze(
        RecommendationRequest(requirement="Yoga mat for a corporate wellness programme")
    )

    assert response.recommendations == []
    assert response.status is RecommendationStatus.PLACEHOLDER
    assert response.no_result_reason is NoResultReason.INSUFFICIENT_COVERAGE
    assert "outside the currently indexed verified knowledge base" in response.message
    # Honest about the local limit, and redirects to the official catalogue.
    assert "NOT a statement that no Indian Standard exists" in response.message
    assert "standards.bis.gov.in" in response.message
    assert response.coverage_note is not None
    assert "rotating electrical machines" in response.coverage_note
    # A coverage gap is *not* a claim about BIS.
    assert "No Indian Standard exists" not in response.message


def test_coverage_and_ai_ruled_none_are_distinguishable(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """Part 5: these two "empty" outcomes are different facts, not one message."""
    # --- AI ran and deliberately selected nothing ----------------------------
    stub_llm.response_text = _verdict()
    ruled_none = _motor_service(catalogue, dataset_records).analyze(
        RecommendationRequest(requirement="Requirements for a submersible pump set")
    )

    # --- retrieval found nothing at all -------------------------------------
    no_candidates = _service(
        catalogue,
        chunks=[_retrieved(dataset_records[_EFFICIENCY_MOTORS], 0.04)],
        indexed_chunks=1,
    ).analyze(RecommendationRequest(requirement="Ordinary Portland cement"))

    # --- the AI provider failed ----------------------------------------------
    stub_llm.raises = LLMRateLimitError("rate limit")
    ai_down = _motor_service(catalogue, dataset_records).analyze(
        RecommendationRequest(requirement="Requirements for a submersible pump set")
    )

    reasons = {r.no_result_reason for r in (ruled_none, no_candidates, ai_down)}
    assert reasons == {
        NoResultReason.AI_RULED_NONE,
        NoResultReason.INSUFFICIENT_COVERAGE,
        NoResultReason.AI_UNAVAILABLE,
    }
    # Only the coverage case carries a coverage note.
    assert ruled_none.coverage_note is None
    assert no_candidates.coverage_note is not None
    assert ai_down.coverage_note is None
    # "AI ruled none" and "AI unavailable" differ in status, not just wording.
    assert ruled_none.status is RecommendationStatus.PLACEHOLDER
    assert ai_down.status is RecommendationStatus.AI_UNAVAILABLE


def test_a_runtime_retrieval_failure_is_not_reported_as_a_coverage_gap(
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """A broken query is not "your requirement is outside the index".

    The pipeline is ready and the catalogue is loaded - the vector-store query
    itself failed. That is a retrieval failure, and must never be presented as a
    statement about the coverage of the verified knowledge base (nor as an AI
    verdict). The reasons stay distinct, as they are for every other case.
    """
    service = _service(
        catalogue,
        chunks=[_retrieved(dataset_records[_EFFICIENCY_MOTORS], 0.75)],
        indexed_chunks=1,
        raises=VectorStoreError("chroma query failed: index is corrupt"),
    )

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))

    assert response.recommendations == []
    assert response.retrieved_candidates == []
    assert response.no_result_reason is NoResultReason.RETRIEVAL_UNAVAILABLE
    # Never the coverage wording: the local index was never consulted.
    assert response.coverage_note is None
    assert "outside the currently indexed verified knowledge base" not in response.message
    assert "search index query failed" in response.message
    # The failure is stated plainly, with the technical detail kept to the warning.
    assert any(
        "retrieval is unavailable" in w.lower() and "index is corrupt" in w
        for w in response.warnings
    )


def test_a_silently_truncated_verdict_is_not_used_to_exclude_candidates(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """An unfinished verdict must not claim it excluded candidates.

    ``finish_reason`` is not always reported (a proxy or a dropped stream can end
    the body without it). The JSON envelope is then repaired so the *finished*
    entries survive validation - but the model never reached the remaining
    candidates, so the verdict is incomplete and must be discarded exactly like a
    reported token-limit cut-off. Otherwise the response tells the user that
    standards were "excluded as out of scope" when they were never judged.
    """
    stub_llm.finish_reason = "stop"  # the provider did NOT report the cut-off
    stub_llm.response_text = (
        '{"recommendations": ['
        '{"standard_number": "' + _EFFICIENCY_MOTORS + '", '
        '"applicability_type": "direct", "confidence": 0.9, '
        '"why_it_matches": "Covers the IE3 efficiency classes."},'
        '{"standard_number": "' + _PUMP_MOTORS + '", "applic'
    )
    service = _motor_service(catalogue, dataset_records)

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))

    assert response.recommendations == []
    assert response.pipeline is not None
    assert response.pipeline.ranking_stage == AI_UNAVAILABLE_STAGE
    assert response.status is RecommendationStatus.AI_UNAVAILABLE
    assert response.ai_reasoning is not None
    assert response.ai_reasoning.truncated is True
    # Nothing was excluded, because nothing was decided.
    assert response.ai_reasoning.candidates_excluded == []
    assert not any("excluded" in w for w in response.warnings)
    assert any("cut short" in w for w in response.warnings)


def test_ai_failure_never_assigns_a_fake_confidence(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """On failure there is no confidence to show - not even a retrieval score."""
    stub_llm.raises = LLMRateLimitError("rate limit")
    service = _motor_service(catalogue, dataset_records)

    payload = service.analyze(
        RecommendationRequest(requirement=_MOTOR_REQUIREMENT)
    ).model_dump()

    assert payload["recommendations"] == []
    # Retrieval signals survive on the *candidate* objects, explicitly disclaimed.
    assert payload["retrieved_candidates"]
    for candidate in payload["retrieved_candidates"]:
        assert "confidence" not in candidate
        assert "applicability_type" not in candidate
        assert "not an applicability decision" in candidate["disclaimer"].lower()


def test_a_successful_verdict_produces_final_recommendations(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """The positive case: AI success -> final, grounded recommendations."""
    stub_llm.response_text = _verdict(
        _item(_EFFICIENCY_MOTORS, confidence=0.91),
        _item(_PUMP_MOTORS, applicability="related", confidence=0.62),
    )
    service = _motor_service(catalogue, dataset_records)

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))

    assert response.status is RecommendationStatus.OK
    assert [r.code for r in response.recommendations] == [
        _EFFICIENCY_MOTORS,
        _PUMP_MOTORS,
    ]
    assert response.pipeline is not None
    assert response.pipeline.ranking_stage == AI_RANKING_STAGE
    # Confidence and applicability come from the model, in the model's order.
    assert response.recommendations[0].confidence == 0.91
    assert response.recommendations[0].applicability_type == "direct"
    assert response.recommendations[1].applicability_type == "related"
    # Grounding: every final recommendation exists in the verified catalogue.
    assert {r.code for r in response.recommendations} <= set(dataset_records)
    assert response.ai_reasoning is not None
    assert response.ai_reasoning.status == "completed"
    assert response.ai_reasoning.candidates_selected == 2


def test_the_ai_can_reorder_and_exclude_candidates(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """AI order wins over retrieval order, and it may drop candidates entirely."""
    # Deterministic order is 12615, 7538. The model inverts it.
    stub_llm.response_text = _verdict(
        _item(_PUMP_MOTORS, confidence=0.7),
        _item(_EFFICIENCY_MOTORS, confidence=0.9),
    )
    service = _motor_service(catalogue, dataset_records)

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))

    assert [r.code for r in response.recommendations] == [
        _PUMP_MOTORS,
        _EFFICIENCY_MOTORS,
    ]
    # The retrieval rank stays attached so the UI can show the movement.
    assert response.recommendations[0].deterministic_rank == 2
    assert response.recommendations[1].deterministic_rank == 1
    assert response.ai_reasoning is not None
    assert response.ai_reasoning.candidates_excluded == []


def test_the_ai_can_exclude_a_verified_candidate(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """A verified candidate the model rules out is dropped, not smuggled through."""
    stub_llm.response_text = _verdict(_item(_EFFICIENCY_MOTORS, confidence=0.9))
    service = _motor_service(catalogue, dataset_records)

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))

    assert [r.code for r in response.recommendations] == [_EFFICIENCY_MOTORS]
    assert response.ai_reasoning is not None
    assert response.ai_reasoning.candidates_excluded == [_PUMP_MOTORS]
    assert any("excluded" in w and _PUMP_MOTORS in w for w in response.warnings)


def test_an_invented_standard_never_reaches_the_recommendations(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """Grounding: the model cannot introduce a designation we never offered."""
    stub_llm.response_text = _verdict(
        _item(_EFFICIENCY_MOTORS, confidence=0.8),
        _item(_INVENTED, confidence=0.99),
    )
    service = _motor_service(catalogue, dataset_records)

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))

    codes = [r.code for r in response.recommendations]
    assert _INVENTED not in codes
    assert set(codes) <= set(dataset_records)
    assert response.ai_reasoning is not None
    assert _INVENTED in response.ai_reasoning.invented_dropped
    assert any("outside the verified candidate set" in w for w in response.warnings)


def test_warnings_are_deduplicated(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """One clear warning per issue, never the same text twice."""
    stub_llm.raises = LLMRateLimitError("rate limit")
    service = _motor_service(catalogue, dataset_records)

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))

    assert response.warnings == list(dict.fromkeys(response.warnings))


# --- procurement-target grounding -------------------------------------------
# The defect: for "packaged drinking water manufacturing and bottling plant" the
# model recommended MOTOR standards (IS 9283, IS 12615, IS 996, IS 9320) by
# reasoning that motors/pumps could exist inside such a plant.
#
# The fix is general prompt-level reasoning, NOT a product rule, so these tests
# assert on the PROMPT CONTRACT and on the model remaining the authority.
# Nothing here encodes "water rejects motors".


def test_prompt_forbids_facility_to_component_inference() -> None:
    """The anti-inference rule must be present and stated as a general rule."""
    system = " ".join(STANDARD_RECOMMENDATION_SYSTEM_PROMPT.lower().split())
    prompt = " ".join(STANDARD_RECOMMENDATION_PROMPT.lower().split())

    assert "could be used within the facility" in system
    assert "explicitly stated it as part of the procurement" in system
    assert "do not infer that a component is being procured" in prompt
    assert "manufacturing process, plant, or system" in prompt


def test_prompt_makes_applicability_evidence_bound_and_domain_agnostic() -> None:
    """Applicability must hinge on verified scope covering the target."""
    # The prompt is hard-wrapped, so compare against a whitespace-normalised copy.
    prompt = " ".join(STANDARD_RECOMMENDATION_PROMPT.lower().split())
    assert "procurement target" in prompt
    assert "verified scope/evidence covers" in prompt
    assert "never sufficient" in prompt
    assert "packaged drinking water" in prompt
    assert "submersible pump sets" in prompt
    assert "must not mirror retrieval similarity" in prompt
    assert "convert high embedding similarity into" in prompt


def test_prompt_separates_a_facility_target_from_an_article_target() -> None:
    """The rule must not over-exclude when the target is itself a product.

    Excluding equipment that merely happens to sit inside a described plant is
    correct (the original defect). Applying the same exclusion when the target
    IS an article - e.g. "supply of submersible pump sets" against a standard for
    the motors integral to those sets - would be a false negative. The prompt has
    to tell the two cases apart, in general terms, for every domain.
    """
    prompt = " ".join(STANDARD_RECOMMENDATION_PROMPT.lower().split())
    # The facility case: equipment inside a plant is not the target.
    assert "facility, plant, process, site or industry" in prompt
    assert "not the equipment installed in it" in prompt
    # The article case: the item itself, and components integral to it, count.
    assert "article, product, equipment or material" in prompt
    assert "integral to and supplied with it" in prompt
    # The exclusion must be scoped to both, not to the facility case alone.
    assert "neither the procurement target nor a component integral to it" in prompt


def test_prompt_does_not_promote_marginal_candidates() -> None:
    """Regression guard for the defect's most direct cause.

    The previous prompt told the model that a lower-similarity standard covering
    the requested scope "must be ranked ABOVE" a better-scoring candidate, which
    actively rewarded reaching for marginal matches. That instruction is gone.
    """
    prompt = STANDARD_RECOMMENDATION_PROMPT
    assert "must be ranked ABOVE" not in prompt
    assert "Do not just echo the deterministic similarity scores" not in prompt


def test_prompt_contains_no_product_to_standard_mapping() -> None:
    """The prompt must stay domain-agnostic: no keyword->standard table."""
    prompt = " ".join(STANDARD_RECOMMENDATION_PROMPT.lower().split())
    for forbidden in (
        "if the requirement mentions",
        "if the product is",
        "when the product is",
        "for water",
        "for cement",
        "for motors",
    ):
        assert forbidden not in prompt


# --- A-F: the scenarios the brief calls out --------------------------------


def test_a_motor_procurement_can_recommend_motor_standards(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """A: verified motor scope supports the motor requirement."""
    stub_llm.response_text = _verdict(
        {
            "standard_number": _EFFICIENCY_MOTORS,
            "applicability_type": "direct",
            "confidence": 0.93,
            "why_it_matches": "Scope covers IE efficiency classes for induction motors.",
        },
        {
            "standard_number": _PUMP_MOTORS,
            "applicability_type": "related",
            "confidence": 0.55,
            "why_it_matches": "Normative reference for squirrel cage motor selection.",
        },
        target="A 15 kW IE3 squirrel cage induction motor, foot mounted, IP55, 415 V.",
    )

    response = _motor_service(catalogue, dataset_records).analyze(
        RecommendationRequest(requirement=_MOTOR_REQUIREMENT)
    )

    by_code = {r.code: r for r in response.recommendations}
    assert set(by_code) == {_EFFICIENCY_MOTORS, _PUMP_MOTORS}
    assert by_code[_EFFICIENCY_MOTORS].applicability_type == "direct"
    assert by_code[_PUMP_MOTORS].applicability_type == "related"
    assert response.no_result_reason is None


def test_b_pump_procurement_can_recommend_pump_standards(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """B: a submersible pump set may be matched when its scope supports it."""
    stub_llm.response_text = _verdict(
        {
            "standard_number": _PUMP_MOTORS,
            "applicability_type": "direct",
            "confidence": 0.88,
            "why_it_matches": "Scope covers squirrel cage motors for centrifugal pumps.",
        },
        target="Submersible pump sets.",
    )

    response = _motor_service(catalogue, dataset_records).analyze(
        RecommendationRequest(requirement="Supply of submersible pump sets")
    )

    assert [r.code for r in response.recommendations] == [_PUMP_MOTORS]


def test_c_packaged_water_never_yields_motor_standards(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """C: the reported defect, at the level the fix operates on.

    Motor candidates ARE still retrieved for a water query - retrieval is
    unchanged and domain-agnostic. What must happen is that the *model* excludes
    them, yielding zero recommendations. The test asserts the outcome and the
    reason; it encodes no rule that would make it happen without the model.
    """
    stub_llm.response_text = _verdict(
        target=(
            "Packaged drinking water for a bottling plant. The user did not state "
            "that pumps or motors are being procured."
        )
    )

    response = _motor_service(catalogue, dataset_records).analyze(
        RecommendationRequest(
            requirement=(
                "BIS standards applicable to packaged drinking water manufacturing "
                "and bottling plant"
            )
        )
    )

    assert response.retrieved_candidates  # retrieval unchanged
    assert response.recommendations == []
    assert response.no_result_reason is NoResultReason.AI_RULED_NONE
    # Specifically none of the standards named in the defect report.
    codes = {r.code for r in response.recommendations}
    assert codes.isdisjoint(
        {"IS 9283:2024", "IS 12615:2018", "IS 996:2009", "IS 9320:2025"}
    )


def test_d_explicitly_named_component_may_be_considered(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """D: naming the pump set puts pump/motor standards back in play."""
    stub_llm.response_text = _verdict(
        {
            "standard_number": _PUMP_MOTORS,
            "applicability_type": "direct",
            "confidence": 0.86,
            "why_it_matches": "The user explicitly stated submersible pump sets.",
        },
        target=(
            "A packaged drinking water plant that explicitly includes procurement "
            "of submersible pump sets."
        ),
    )

    response = _motor_service(catalogue, dataset_records).analyze(
        RecommendationRequest(
            requirement=(
                "Packaged drinking water plant including procurement of "
                "submersible pump sets"
            )
        )
    )

    assert [r.code for r in response.recommendations] == [_PUMP_MOTORS]
    assert "pump sets" in response.procurement_target.target.lower()


def test_e_noise_requirements_can_recommend_retrieved_standards(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """E: verified scope, not similarity, drives the outcome."""
    stub_llm.response_text = _verdict(
        {
            "standard_number": _NOISE,
            "applicability_type": "direct",
            "confidence": 0.9,
            "why_it_matches": "Scope covers noise emission for rotating machines.",
        },
        target="Noise and vibration limits for rotating electrical machines.",
    )
    service = _service(catalogue, chunks=[_retrieved(dataset_records[_NOISE], 0.62)])

    response = service.analyze(
        RecommendationRequest(
            requirement=(
                "Noise and vibration requirements for rotating electrical machines"
            )
        )
    )

    assert [r.code for r in response.recommendations] == [_NOISE]


def test_f_unrelated_query_never_fabricates_a_standard(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """F: an unrelated requirement yields nothing invented, either way."""
    stub_llm.response_text = _verdict(
        {
            "standard_number": _INVENTED,
            "applicability_type": "direct",
            "confidence": 0.9,
            "why_it_matches": "Fabricated.",
        },
        target="Handwoven silk Banarasi saree.",
    )

    response = _motor_service(catalogue, dataset_records).analyze(
        RecommendationRequest(
            requirement="Handwoven silk Banarasi saree for export to the European Union"
        )
    )

    assert response.recommendations == []
    assert _INVENTED not in {r.code for r in response.recommendations}
    # The invented designation is reported, never silently dropped.
    assert _INVENTED in response.ai_reasoning.invented_dropped


# --- transparency + domain-agnostic guarantees -----------------------------


def test_procurement_target_is_surfaced_without_driving_the_decision(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """The model's own target statement is captured and exposed verbatim."""
    stub_llm.response_text = _verdict(
        {
            "standard_number": _EFFICIENCY_MOTORS,
            "applicability_type": "direct",
            "confidence": 0.9,
            "why_it_matches": "Covers IE efficiency classes.",
        },
        target="An IE3 three-phase squirrel cage induction motor, 15 kW, foot mounted, IP55.",
    )

    response = _motor_service(catalogue, dataset_records).analyze(
        RecommendationRequest(requirement=_MOTOR_REQUIREMENT)
    )

    assert response.procurement_target is not None
    assert "IE3" in response.procurement_target.target
    assert response.recommendations[0].code == _EFFICIENCY_MOTORS


def test_a_missing_procurement_target_never_blocks_a_valid_verdict(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """The target block is transparency, not a gate.

    A model that answers the schema without it still yields its decision. The
    field must never become a hidden precondition that silently empties results.
    """
    stub_llm.response_text = _verdict(
        {
            "standard_number": _EFFICIENCY_MOTORS,
            "applicability_type": "direct",
            "confidence": 0.8,
            "why_it_matches": "Covers IE efficiency classes.",
        }
    )

    response = _motor_service(catalogue, dataset_records).analyze(
        RecommendationRequest(requirement=_MOTOR_REQUIREMENT)
    )

    assert response.procurement_target is None
    assert [r.code for r in response.recommendations] == [_EFFICIENCY_MOTORS]


def test_arbitrary_domains_are_never_special_cased(
    stub_llm: type[StubReasoningProvider],
    catalogue: StandardsService,
    dataset_records: dict[str, Any],
) -> None:
    """Adversarial: no domain is special-cased anywhere in the service.

    Each requirement is judged purely by what the model returns. No domain
    knowledge is encoded in this test or in the service - that is the point.
    """
    for requirement in (
        "Supply of 2.5 kg fire extinguishers with ISI marking",
        "500 kVA 11/0.415 kV distribution transformer",
        "XLPE armoured 4 core 70 sq mm electrical cable",
        "Ordinary Portland Cement 43 Grade",
        "M40 ready mix concrete",
        "8 inch submersible pump set",
    ):
        stub_llm.response_text = _verdict(
            {
                "standard_number": _EFFICIENCY_MOTORS,
                "applicability_type": "direct",
                "confidence": 0.9,
                "why_it_matches": "Model-authored rationale.",
            },
            target=f"The model read this as: {requirement}",
        )
        response = _motor_service(catalogue, dataset_records).analyze(
            RecommendationRequest(requirement=requirement)
        )
        # The model chose it, so it is returned. No domain filter overrode it,
        # which is exactly the guarantee: the decision is never made in code.
        assert [r.code for r in response.recommendations] == [_EFFICIENCY_MOTORS]
        assert requirement in response.procurement_target.target


def test_retrieval_scores_are_never_reported_as_applicability_confidence(
    catalogue: StandardsService, dataset_records: dict[str, Any]
) -> None:
    """A high-similarity candidate must not surface as a confident answer.

    With no LLM configured the only output is candidates, and candidates carry
    no confidence or applicability fields at all.
    """
    response = _motor_service(catalogue, dataset_records).analyze(
        RecommendationRequest(requirement=_MOTOR_REQUIREMENT)
    )

    assert response.recommendations == []
    for candidate in response.model_dump()["retrieved_candidates"]:
        assert "confidence" not in candidate
        assert "applicability_type" not in candidate

