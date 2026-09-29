"""Tests for the deterministic ranking stage and the recommendation contract.

Nothing here touches the network or an embedding model: the ranker and the
service are fed *prepared* retrievals built from the real, shipped catalogue
records. That keeps the tests fast while still exercising the grounding rules
(verified catalogue data only, verbatim evidence, no invented fields).

The load-bearing architectural rule, asserted throughout: the deterministic
layer retrieves and scores candidates, and **never** decides applicability.
Final recommendations require an AI verdict.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pytest

from app.ai.llm.base import LLMProvider, LLMRequest, LLMResponse
from app.ai.llm.factory import register_llm_provider, reset_llm_provider_cache
from app.core.config import get_settings
from app.database.repositories import InMemoryStandardRepository
from app.rag.attributes import (
    FAMILY_MATCH_QUALITY,
    extract_technical_attributes,
    match_attributes,
)
from app.rag.pipeline import PipelineReadiness
from app.rag.ranking import (
    attribute_rarity,
    attribute_weights,
    explain_match,
    extract_terms,
    group_chunks_by_standard,
    rank_standards,
    score_confidence,
    split_clauses,
)
from app.rag.retrieval import RetrievedChunk
from app.schemas.recommendation import RecommendationRequest, RecommendationStatus
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
_EFFICIENCY_MOTORS = "IS 12615:2018"  # efficiency classes / IE code
_PUMP_MOTORS = "IS 7538:1996"  # squirrel cage motors for centrifugal pumps
_NOISE = "IS 12065:2025"

_MOTOR_REQUIREMENT = (
    "Supply of 15 kW IE3 three-phase squirrel cage induction motor, "
    "foot mounted, IP55 enclosure, 415 V, 50 Hz"
)


# --- fixtures ---------------------------------------------------------------
class _StubProvider(LLMProvider):
    """Scripted, network-free provider used to exercise the AI-decides path."""

    provider_name = "stub-ranking"
    response_text: str = "{}"
    raises: BaseException | None = None
    requests: list[LLMRequest] = []

    @classmethod
    def reset(cls) -> None:
        cls.response_text = "{}"
        cls.raises = None
        cls.requests = []

    def __init__(self, settings: Any = None) -> None:
        self.settings = settings

    @property
    def is_configured(self) -> bool:
        return True

    @property
    def model(self) -> str:
        return "stub-ranking-model"

    def generate(self, request: LLMRequest) -> LLMResponse:
        cls = type(self)
        cls.requests.append(request)
        if cls.raises is not None:
            raise cls.raises
        return LLMResponse(
            text=cls.response_text,
            provider=self.provider_name,
            model=self.model,
            finish_reason="stop",
            usage={},
        )


@pytest.fixture
def stub_llm(monkeypatch: pytest.MonkeyPatch):
    """Make the stub the configured provider for a single test."""
    register_llm_provider(_StubProvider.provider_name, _StubProvider)
    _StubProvider.reset()
    monkeypatch.setenv("LLM_PROVIDER", _StubProvider.provider_name)
    get_settings.cache_clear()
    reset_llm_provider_cache()
    yield _StubProvider
    _StubProvider.reset()
    get_settings.cache_clear()
    reset_llm_provider_cache()


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
    """Stand-in for ``RecommendationPipeline`` returning prepared chunks."""

    def __init__(
        self,
        *,
        ready: bool,
        chunks: Sequence[RetrievedChunk] = (),
        indexed_chunks: int | None = None,
        reasons: Iterable[str] = (),
    ) -> None:
        self._ready = ready
        self._chunks = list(chunks)
        self._indexed = len(self._chunks) if indexed_chunks is None else indexed_chunks
        self._reasons = list(reasons)

    def readiness(self) -> PipelineReadiness:
        return PipelineReadiness(
            ready=self._ready,
            embedding_provider="onnx_minilm (configured, model=all-MiniLM-L6-v2)",
            vector_store="chroma",
            indexed_chunks=self._indexed,
            reasons=self._reasons,
        )

    def retrieve(self, query: str, *, top_k: int | None = None, where=None):
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
) -> RecommendationService:
    return RecommendationService(
        pipeline=_StubPipeline(
            ready=ready, chunks=chunks, indexed_chunks=indexed_chunks
        ),
        standards_service=catalogue,
    )


# --- term / clause helpers --------------------------------------------------
def test_extract_terms_ignores_filler_and_folds_plurals() -> None:
    terms = extract_terms("Supply of the three phase induction motors for pumps")

    assert {"three", "phase", "induction", "motor", "motors", "pump", "pumps"} <= terms
    for filler in ("supply", "the", "of", "for"):
        assert filler not in terms


def test_split_clauses_keeps_verbatim_fragments() -> None:
    clauses = split_clauses("Supply of 15 kW IE3 motors, foot mounted, IP55")

    assert "Supply of 15 kW IE3 motors" in clauses
    assert "foot mounted" in clauses
    assert "IP55" in clauses


def test_confidence_is_monotone_and_bounded() -> None:
    assert score_confidence(0.0, 0) == 0.0
    assert score_confidence(2.0, 20) <= 1.0
    assert score_confidence(0.8, 3) > score_confidence(0.3, 3)
    assert score_confidence(0.5, 10) > score_confidence(0.5, 1)


def test_group_chunks_ignores_chunks_without_a_designation(
    dataset_records: dict[str, Any],
) -> None:
    chunk = _retrieved(dataset_records[_EFFICIENCY_MOTORS], 0.6)
    anonymous = RetrievedChunk(
        chunk_id="upload::0", text="tender text", score=0.9, metadata={}
    )

    groups = group_chunks_by_standard([chunk, anonymous])

    assert list(groups) == [_EFFICIENCY_MOTORS]


# --- ranking stage ----------------------------------------------------------
def test_rank_standards_groups_chunks_and_orders_by_confidence(
    dataset_records: dict[str, Any],
) -> None:
    chunks = [
        _retrieved(dataset_records[_EFFICIENCY_MOTORS], 0.30, chunk_index=0),
        _retrieved(dataset_records[_EFFICIENCY_MOTORS], 0.61, chunk_index=1),
        _retrieved(dataset_records[_NOISE], 0.28, chunk_index=0),
    ]

    outcome = rank_standards(
        chunks,
        requirement="efficiency classes for three phase induction motors",
    )

    assert [c.standard_number for c in outcome.candidates] == [
        _EFFICIENCY_MOTORS,
        _NOISE,
    ]
    best = outcome.candidates[0]
    assert best.retrieval_score == 0.61  # best chunk wins, not the first one
    assert best.chunk_count == 2
    assert best.rank == 1
    assert outcome.candidates[1].rank == 2
    assert best.evidence[0].score == 0.61  # evidence ordered strongest first
    assert outcome.groups_seen == 2
    assert outcome.rejected_count == 0


def test_rank_standards_rejects_records_below_the_relevance_floor(
    dataset_records: dict[str, Any],
) -> None:
    outcome = rank_standards(
        [_retrieved(dataset_records[_EFFICIENCY_MOTORS], 0.04)],
        requirement="Supply of 43 grade ordinary Portland cement in 50 kg bags",
    )

    assert outcome.candidates == []
    assert outcome.below_floor == [_EFFICIENCY_MOTORS]
    assert outcome.unverified == []


def test_rank_standards_discards_designations_absent_from_the_catalogue(
    dataset_records: dict[str, Any],
) -> None:
    verified = _retrieved(dataset_records[_EFFICIENCY_MOTORS], 0.55)
    invented = RetrievedChunk(
        chunk_id="ghost::0",
        text="Standard Number: IS 99999:2030\nTitle: Not a real record",
        score=0.95,
        metadata={"standard_number": "IS 99999:2030"},
    )

    outcome = rank_standards(
        [verified, invented],
        requirement="three phase induction motor efficiency classes",
        catalogue_contains=lambda code: code == _EFFICIENCY_MOTORS,
    )

    assert [c.standard_number for c in outcome.candidates] == [_EFFICIENCY_MOTORS]
    assert outcome.unverified == ["IS 99999:2030"]


def test_rank_standards_is_deterministic_and_respects_max_results(
    dataset_records: dict[str, Any],
) -> None:
    chunks = [
        _retrieved(dataset_records[code], 0.5)
        for code in (_EFFICIENCY_MOTORS, _PUMP_MOTORS, _NOISE)
    ]
    requirement = "three phase squirrel cage induction motors for pumps"

    first = rank_standards(chunks, requirement=requirement, max_results=2)
    second = rank_standards(chunks, requirement=requirement, max_results=2)

    assert [c.standard_number for c in first.candidates] == [
        c.standard_number for c in second.candidates
    ]
    assert len(first.candidates) == 2
    assert first.groups_seen == 3


def test_matched_requirements_and_terms_come_from_the_retrieved_text(
    dataset_records: dict[str, Any],
) -> None:
    requirement = (
        "Supply of squirrel cage induction motors for centrifugal pumps, "
        "IE3 efficiency class, IP55 enclosure"
    )

    outcome = rank_standards(
        [_retrieved(dataset_records[_PUMP_MOTORS], 0.7)], requirement=requirement
    )
    candidate = outcome.candidates[0]

    assert "squirrel" in candidate.matched_terms
    assert candidate.matched_requirements
    # Every quoted clause is the caller's own wording, never paraphrased.
    for clause in candidate.matched_requirements:
        assert clause in requirement


def test_explanation_states_its_basis_without_claiming_applicability(
    dataset_records: dict[str, Any],
) -> None:
    record = dataset_records[_EFFICIENCY_MOTORS]

    text = explain_match(
        rank=1,
        standard_number=_EFFICIENCY_MOTORS,
        retrieval_score=0.6118,
        matched_terms=("motor", "phase"),
        scope=record.scope,
    )

    assert "Ranked #1" in text
    assert "0.612" in text
    assert _EFFICIENCY_MOTORS in text
    assert "motor" in text and "phase" in text
    assert record.scope[:40] in text
    assert "not a determination that the standard applies" in text


def test_explanation_says_when_a_match_is_purely_semantic() -> None:
    text = explain_match(
        rank=2,
        standard_number=_NOISE,
        retrieval_score=0.4,
        matched_terms=(),
        semantic_only=True,
    )

    assert "semantic similarity alone" in text


# --- technical attributes ---------------------------------------------------
def test_extract_technical_attributes_reads_the_requirement() -> None:
    attributes = extract_technical_attributes(_MOTOR_REQUIREMENT)

    values = {attribute.value for attribute in attributes}
    assert {
        "ie3",
        "kw:15",
        "v:415",
        "hz:50",
        "ip55",
        "phase:3",
        "squirrel-cage",
        "induction",
        "foot",
    } <= values
    # Verbatim surfaces are kept so an explanation can quote them.
    surfaces = {attribute.surface for attribute in attributes}
    assert {"IE3", "15 kW", "415 V", "50 Hz", "IP55"} <= surfaces


def test_power_voltage_and_frequency_values_are_normalised() -> None:
    attributes = extract_technical_attributes("Motor: 11 kV, 0.75 kW, 60 Hz")

    values = {attribute.value for attribute in attributes}
    assert {"v:11000", "kw:0.75", "hz:60"} <= values


def test_stated_rating_ranges_cover_a_required_value() -> None:
    available = extract_technical_attributes("Output range: 0.37 kW to 375 kW")

    match = match_attributes(extract_technical_attributes("15 kW motor"), available)

    assert match.quality_by_kind()["power"] == 1.0


def test_attribute_matching_distinguishes_exact_family_and_missing() -> None:
    required = extract_technical_attributes("IE3 motor, IP55 enclosure, 415 V")
    available = extract_technical_attributes(
        "Line-Operated Three-Phase a.c. Motors - Efficiency Classes and "
        "Performance Specification (IE Code)"
    )

    match = match_attributes(required, available)
    detail = {item.kind: item for item in match.details}

    # The record covers the IE efficiency classes but does not state IE3.
    assert detail["efficiency_class"].quality == FAMILY_MATCH_QUALITY
    assert detail["efficiency_class"].family_values == ("IE3",)
    assert detail["efficiency_class"].missing_values == ()
    # Nothing in that record is about a voltage.
    assert detail["voltage"].quality == 0.0
    assert detail["voltage"].missing_values == ("415 V",)
    assert "415 V" in detail["voltage"].values


def test_a_single_phase_or_dc_record_earns_no_phase_or_type_credit() -> None:
    required = extract_technical_attributes("three phase induction motor")

    single_phase = match_attributes(
        required,
        extract_technical_attributes("Single Phase a.c. Induction Motors"),
    )
    dc_machine = match_attributes(
        required,
        extract_technical_attributes("Guide for Testing Direct Current (DC) Machines"),
    )

    # Phase mismatch earns nothing; the induction/DC type mismatch earns nothing.
    assert single_phase.quality_by_kind()["phase"] == 0.0
    assert single_phase.quality_by_kind()["motor_type"] == 1.0
    assert dc_machine.quality_by_kind()["motor_type"] == 0.0
    assert dc_machine.quality_by_kind()["phase"] == 0.0


def test_attribute_weights_favour_stated_values_and_rare_kinds() -> None:
    required = ("efficiency_class", "phase", "motor_type")
    weights = attribute_weights(required, {"efficiency_class": 1, "phase": 9, "motor_type": 9}, 10)

    # A stated-value kind outweighs product-classification kinds...
    assert weights["efficiency_class"] > weights["phase"]
    # ...and a rare kind outweighs a kind every candidate mentions.
    assert weights["motor_type"] > 0.0
    assert attribute_rarity(1, 10) > attribute_rarity(9, 10) > 0.0


def test_confidence_lets_attributes_lead_and_similarity_follow() -> None:
    # Same plain-text support, different attribute coverage: attributes decide.
    weak_attributes = score_confidence(0.60, 0, 0.10)
    strong_attributes = score_confidence(0.46, 0, 0.42)

    assert strong_attributes > weak_attributes
    # Without attributes the blend falls back to similarity/text overlap only.
    assert score_confidence(0.50, 0) == round(0.85 * 0.50, 4)



# --- recommendation service contract ----------------------------------------
def test_analyze_returns_candidates_but_no_recommendation_without_ai(
    catalogue: StandardsService, dataset_records: dict[str, Any]
) -> None:
    """Retrieval works, but the deterministic layer is not the decision maker.

    This is the invariant that the packaged-drinking-water regression turned on:
    retrieval output must never be presented as applicable standards.
    """
    service = _service(
        catalogue,
        chunks=[
            _retrieved(dataset_records[_EFFICIENCY_MOTORS], 0.75),
            _retrieved(dataset_records[_PUMP_MOTORS], 0.45),
        ],
    )

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))

    # The deterministic layer may not decide applicability.
    assert response.recommendations == []
    assert response.status is RecommendationStatus.AI_UNAVAILABLE
    assert response.pipeline is not None
    assert response.pipeline.ranking_stage != AI_RANKING_STAGE

    # What it *may* do: retrieve verified candidates with supporting signals.
    assert [c.standard_number for c in response.retrieved_candidates] == [
        _EFFICIENCY_MOTORS,
        _PUMP_MOTORS,
    ]
    assert [c.retrieval_rank for c in response.retrieved_candidates] == [1, 2]
    assert response.retrieved_candidates[0].title == dataset_records[_EFFICIENCY_MOTORS].title
    # Grounding: every candidate is a designation from the loaded catalogue.
    assert {c.standard_number for c in response.retrieved_candidates} <= set(dataset_records)

    # Evidence is the indexed catalogue text itself, never generated prose.
    assert response.retrieved_evidence
    assert response.retrieved_evidence[0].snippet.startswith("Standard Number:")


def test_retrieved_candidates_carry_no_applicability_decision(
    catalogue: StandardsService, dataset_records: dict[str, Any]
) -> None:
    """A candidate must not carry confidence, a class, or generated reasoning."""
    service = _service(
        catalogue, chunks=[_retrieved(dataset_records[_EFFICIENCY_MOTORS], 0.64)]
    )

    payload = service.analyze(
        RecommendationRequest(requirement=_MOTOR_REQUIREMENT)
    ).model_dump()

    assert payload["recommendations"] == []
    candidate = payload["retrieved_candidates"][0]
    for forbidden in ("confidence", "applicability_type", "reasoning", "why_it_matches"):
        assert forbidden not in candidate
    # The signal is labelled as a retrieval signal, and disclaimed as such.
    assert candidate["retrieval_rank"] == 1
    assert "not an applicability decision" in candidate["disclaimer"].lower()
    # The UI is told the candidate is not a decision.
    assert "recommendations" not in payload or payload["recommendations"] == []


def test_response_exposes_the_requirement_understanding_fields(
    catalogue: StandardsService, dataset_records: dict[str, Any]
) -> None:
    """Detected attributes are surfaced, and clearly marked as supporting only."""
    service = _service(
        catalogue, chunks=[_retrieved(dataset_records[_EFFICIENCY_MOTORS], 0.64)]
    )

    payload = service.analyze(
        RecommendationRequest(requirement=_MOTOR_REQUIREMENT)
    ).model_dump()

    understanding = payload["requirement_understanding"]
    assert understanding is not None
    detected = understanding["detected"]
    assert detected, "the IE3 motor requirement has detectable technical attributes"
    surfaces = {item["surface"].lower() for item in detected}
    # Values are read verbatim from the requirement, never invented.
    assert any("15 kw" in surface for surface in surfaces)
    assert any("415 v" in surface for surface in surfaces)
    assert any("ie3" in surface for surface in surfaces)
    for item in detected:
        assert item["label"] and item["kind"]
    # Supporting information only: never an applicability rule.
    assert "supporting" in understanding["note"].lower()


def test_analyze_reports_not_configured_when_retrieval_cannot_run(
    catalogue: StandardsService,
) -> None:
    service = _service(catalogue, ready=False, chunks=[], indexed_chunks=0)

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))

    assert response.status is RecommendationStatus.NOT_CONFIGURED
    assert response.recommendations == []
    assert response.retrieved_evidence == []
    assert response.pipeline is not None
    assert response.pipeline.ready is False
    # The missing LLM is reported as a reason, not as the cause of the status:
    # retrieval itself is what is unconfigured here.
    assert any("LLM" in reason for reason in response.pipeline.reasons)
    assert all("not configured" not in warning for warning in response.warnings)


def test_analyze_reports_placeholder_when_nothing_can_be_verified(
    catalogue: StandardsService,
) -> None:
    ghost = RetrievedChunk(
        chunk_id="ghost::0",
        text="Standard Number: IS 99999:2030\nTitle: Not a real record",
        score=0.92,
        metadata={"standard_number": "IS 99999:2030"},
    )
    service = _service(catalogue, chunks=[ghost], indexed_chunks=1)

    response = service.analyze(
        RecommendationRequest(requirement="IS 99999:2030 rotating machines")
    )

    assert response.status is RecommendationStatus.PLACEHOLDER
    assert response.recommendations == []
    assert any("not in the loaded catalogue" in w for w in response.warnings)
    assert response.retrieved_evidence  # retrieval output is still reported


def test_analyze_reports_placeholder_when_matches_are_below_the_floor(
    catalogue: StandardsService, dataset_records: dict[str, Any]
) -> None:
    service = _service(
        catalogue,
        chunks=[_retrieved(dataset_records[_EFFICIENCY_MOTORS], 0.04)],
        indexed_chunks=1,
    )

    response = service.analyze(
        RecommendationRequest(
            requirement="Supply of 43 grade ordinary Portland cement in 50 kg bags"
        )
    )

    assert response.status is RecommendationStatus.PLACEHOLDER
    assert response.recommendations == []
    assert any("relevance floor" in w for w in response.warnings)


def test_recommendations_never_invent_certification_or_titles(
    stub_llm, catalogue: StandardsService, dataset_records: dict[str, Any]
) -> None:
    """Certification and titles come from the catalogue, never from the model."""
    certified = "IS 9283:2024"
    service = _service(
        catalogue,
        chunks=[
            _retrieved(dataset_records[certified], 0.62),
            _retrieved(dataset_records[_EFFICIENCY_MOTORS], 0.55),
        ],
    )
    # The model asserts a certification and a title that the catalogue does not hold.
    stub_llm.response_text = json.dumps(
        {
            "recommendations": [
                {
                    "standard_number": certified,
                    "applicability_type": "direct",
                    "confidence": 0.8,
                    "why_it_matches": "Scope covers submersible pump sets.",
                },
                {
                    "standard_number": _EFFICIENCY_MOTORS,
                    "applicability_type": "related",
                    "confidence": 0.5,
                    "why_it_matches": "Related efficiency standard.",
                },
            ]
        }
    )

    response = service.analyze(
        RecommendationRequest(requirement="line operated a.c. motors for submersible pump sets")
    )
    by_code = {record.code: record for record in response.recommendations}

    assert set(by_code) == {certified, _EFFICIENCY_MOTORS}
    # Only the record whose dataset entry states a certification reports one.
    assert by_code[certified].certification
    assert by_code[certified].certification[0].notes == (
        dataset_records[certified].certification.notes
    )
    assert by_code[_EFFICIENCY_MOTORS].certification == []
    # Titles come from the catalogue, not from the model.
    assert by_code[_EFFICIENCY_MOTORS].title == dataset_records[_EFFICIENCY_MOTORS].title
    assert all(record.title for record in response.recommendations)

def test_latest_revision_from_status():
    """Latest-revision detection is evidence-bound, never inferred."""
    from app.services.standards_service import _latest_revision_from_status as f

    newer = (
        "A newer BIS revision, IS 12615:2026, is listed in the current BIS "
        "archive; the 2018 edition should not be treated as the latest version."
    )
    assert f("IS 12615:2018", newer) == "IS 12615:2026"
    # Nothing stated -> not "latest", just unknown.
    assert f("IS 7538:1996", "Published; First Revision; reviewed in 2026.") is None
    assert f("IS 12615:2018", None) is None
    # Another standard's year must not be borrowed.
    assert f("IS 12615:2018", "See IS 7538:1996 for details.") is None
    # An older year is not a "latest revision".
    assert f("IS 12615:2018", "Superseded by IS 12615:2010.") is None


def test_latest_revision_reaches_the_recommendation(
    stub_llm, catalogue: StandardsService, dataset_records: dict[str, Any]
) -> None:
    """The newer edition is surfaced on the card without hiding the record."""
    stub_llm.response_text = json.dumps(
        {
            "recommendations": [
                {
                    "standard_number": _EFFICIENCY_MOTORS,
                    "applicability_type": "direct",
                    "confidence": 0.9,
                    "why_it_matches": "Covers IE efficiency classes.",
                }
            ]
        }
    )
    service = _service(
        catalogue, chunks=[_retrieved(dataset_records[_EFFICIENCY_MOTORS], 0.75)]
    )

    response = service.analyze(RecommendationRequest(requirement=_MOTOR_REQUIREMENT))
    top = response.recommendations[0]

    assert top.code == _EFFICIENCY_MOTORS
    assert top.version is not None
    assert top.version.latest_revision == "IS 12615:2026"


# --- query scenarios (similarities measured against the live index) ----------
#: Similarities measured with the shipped 20-record index and the onnx_minilm
#: embedding provider (RUN_LIVE_RAG_TESTS / verify_search) - kept here so these
#: tests stay fast and deterministic while ranking real catalogue records.
_IE3_SIMILARITIES = (
    ("IS 7538:1996", 0.6070),
    ("IS 13529:2021", 0.5860),
    ("IS 14578:1999", 0.4884),
    ("IS 8151:2024", 0.4882),
    ("IS 1231:2019", 0.4723),
    ("IS 12615:2018", 0.4642),
    ("IS 4029:2010", 0.4301),
    ("IS 18073:2023", 0.4294),
    ("IS/IEC 60034-1:2022", 0.4119),
    ("IS 14582:2021", 0.4090),
)
_IP55_SIMILARITIES = (
    ("IS 7538:1996", 0.5365),
    ("IS 13529:2021", 0.4895),
    ("IS 14578:1999", 0.4123),
    ("IS 8151:2024", 0.4099),
    ("IS/IEC 60034-5:2020", 0.3596),
)
_NOISE_SIMILARITIES = (
    ("IS 12065:2025", 0.7501),
    ("IS 12075:2024", 0.7468),
    ("IS/IEC 60034-1:2022", 0.5082),
    ("IS/IEC 60034-2-1:2024", 0.4934),
    ("IS/IEC 60034-5:2020", 0.4795),
)


def _chunks_from(
    dataset_records: dict[str, Any], scores: tuple[tuple[str, float], ...]
) -> list[RetrievedChunk]:
    return [
        _retrieved(dataset_records[code], score, chunk_index=index)
        for index, (code, score) in enumerate(scores)
    ]


def test_ie3_query_ranks_the_efficiency_classes_standard_first(
    dataset_records: dict[str, Any],
) -> None:
    """Regression test for the reported issue: IE3 must beat raw title similarity."""
    outcome = rank_standards(
        _chunks_from(dataset_records, _IE3_SIMILARITIES),
        requirement=_MOTOR_REQUIREMENT,
        catalogue_contains=lambda code: code in dataset_records,
    )

    order = [c.standard_number for c in outcome.candidates]
    assert order[0] == _EFFICIENCY_MOTORS
    assert order.index(_EFFICIENCY_MOTORS) < order.index("IS 7538:1996")

    top = outcome.candidates[0]
    pump = next(c for c in outcome.candidates if c.standard_number == "IS 7538:1996")
    assert pump.retrieval_score > top.retrieval_score
    assert top.attribute_score > pump.attribute_score
    assert any("IE efficiency class" in item for item in top.matched_attributes)

    # Missing attributes across the retrieved candidate pool are reported, never guessed.
    assert set(outcome.unmatchable_kinds) >= {"power", "voltage", "frequency"}
    assert any("power rating" in item for item in top.unmatched_attributes)


def test_ip55_query_ranks_the_ip_code_standard_first(
    dataset_records: dict[str, Any],
) -> None:
    requirement = (
        "Totally enclosed fan cooled squirrel cage induction motor, "
        "IP55 protection, three phase"
    )
    outcome = rank_standards(
        _chunks_from(dataset_records, _IP55_SIMILARITIES),
        requirement=requirement,
        catalogue_contains=lambda code: code in dataset_records,
    )

    order = [c.standard_number for c in outcome.candidates]
    assert order[0] == "IS/IEC 60034-5:2020"

    top = outcome.candidates[0]
    detail = next(
        item for item in top.matched_attributes if "IP enclosure rating" in item
    )
    assert "IP55" in detail
    assert "does not state IP55" in detail


def test_noise_and_vibration_query_keeps_the_relevant_standards_on_top(
    dataset_records: dict[str, Any],
) -> None:
    requirement = (
        "Permissible noise level limits and vibration measurement for rotating "
        "electrical machines with shaft height 56 mm and above"
    )
    outcome = rank_standards(
        _chunks_from(dataset_records, _NOISE_SIMILARITIES),
        requirement=requirement,
        catalogue_contains=lambda code: code in dataset_records,
    )

    order = [c.standard_number for c in outcome.candidates]
    assert set(order[:2]) == {"IS 12065:2025", "IS 12075:2024"}
    assert order.index("IS/IEC 60034-5:2020") > 1
    # When the requirement states no parseable attribute, similarity leads.
    assert all(c.attribute_score is None for c in outcome.candidates)


def test_unrelated_query_produces_no_recommendations(
    dataset_records: dict[str, Any],
) -> None:
    chunks = [
        _retrieved(dataset_records[_EFFICIENCY_MOTORS], 0.081),
        _retrieved(dataset_records["IS 12065:2025"], 0.080),
    ]
    outcome = rank_standards(
        chunks,
        requirement="Supply of 43 grade ordinary Portland cement in 50 kg bags",
        catalogue_contains=lambda code: code in dataset_records,
    )

    assert outcome.candidates == []
    assert outcome.below_floor == [_EFFICIENCY_MOTORS, "IS 12065:2025"]
    assert outcome.attribute_kinds == ()


def test_broad_motor_query_prefers_three_phase_induction_standards(
    dataset_records: dict[str, Any],
) -> None:
    chunks = [
        _retrieved(dataset_records["IS 996:2009"], 0.70),  # single phase
        _retrieved(dataset_records["IS 9320:2025"], 0.68),  # DC machines
        _retrieved(dataset_records["IS 14578:1999"], 0.55),  # three-phase induction
    ]
    outcome = rank_standards(
        chunks,
        requirement="Three phase induction motor",
        catalogue_contains=lambda code: code in dataset_records,
    )

    # Even though IS 14578 had lower similarity, matching both the phase and
    # induction motor attributes puts it on top over single-phase or DC machines.
    assert outcome.candidates[0].standard_number == "IS 14578:1999"
    assert [c.standard_number for c in outcome.candidates] == [
        "IS 14578:1999",
        "IS 996:2009",
        "IS 9320:2025",
    ]


