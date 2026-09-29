"""Deterministic ranking of retrieved chunks into standard recommendations.

This module turns *real* retrieval output (chunks returned by the vector store)
into an ordered list of standard candidates. It is deliberately LLM-free so the
recommendation feature works with no API key, and it never invents anything:

* a candidate is produced only from chunks that carry a catalogue identifier, so
  the text behind every recommendation is the indexed catalogue text itself;
* ``catalogue_contains`` is an optional gate: when the caller passes the loaded
  catalogue's lookup, any retrieved designation that is *not* part of the
  validated dataset is dropped instead of being recommended;
* the confidence score is a documented blend of the vector-store similarity and
  how many distinctive requirement terms actually appear in the record, so it can
  be recomputed by hand rather than trusted blindly.

Ordering is fully deterministic (confidence, then similarity, then
designation), which keeps the API reproducible for the same query and index.

This module is the *candidate generation / signal* stage of the pipeline: it
never decides applicability. When an LLM provider is configured,
:mod:`app.ai.reasoning` consumes these verified candidates plus their evidence
and returns the final applicability classification and ranking, and it may
reorder the list. The score here is therefore reported as
``deterministic_rank`` / a supporting signal, not as the verdict.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

from app.rag.attributes import (
    CATEGORY_KINDS,
    KIND_LABELS,
    VALUE_KINDS,
    AttributeMatch,
    TechnicalAttribute,
    describe_matched,
    describe_unmatched,
    extract_technical_attributes,
    match_attributes,
    required_kinds,
)
from app.rag.retrieval import RetrievedChunk

#: Minimum vector-store similarity (cosine, 0..1) for a record to be ranked.
#: Below this the "match" is noise: measured out-of-catalogue queries score
#: ~0.05-0.15 against this 20-record catalogue while genuine ones score ~0.4-0.8.
DEFAULT_MIN_SCORE = 0.25

#: Upper bound on returned recommendations.
DEFAULT_MAX_RECOMMENDATIONS = 10

#: Number of matched requirement terms that saturates the lexical support signal.
TERM_SUPPORT_TARGET = 5

#: Confidence weights used when the requirement states technical attributes.
#: The embedding model cannot distinguish "IE3" from "IP55" from a frame number,
#: so explicit attribute satisfaction leads and free-text similarity - which a
#: near-literal title match can inflate - is only a secondary signal. Text-overlap
#: of words the attributes already account for is a weak tie-breaker only.
SIMILARITY_WEIGHT = 0.25
ATTRIBUTE_WEIGHT = 0.70
TERM_SUPPORT_WEIGHT = 0.05

#: Confidence weights used when no technical attribute could be read from the
#: requirement (nothing to match, so similarity decides).
SIMILARITY_WEIGHT_NO_ATTRIBUTES = 0.85
TERM_SUPPORT_WEIGHT_NO_ATTRIBUTES = 0.15

#: A requirement that states a concrete value (IE3, IP55, 15 kW, 415 V, 50 Hz,
#: IC411, duty, insulation class) outweighs one that only classifies the product
#: (three-phase, squirrel-cage, foot-mounted). Rarity is applied on top.
STATED_VALUE_TIER = 3.5
CATEGORY_TIER = 1.0

#: Longest catalogue scope quoted inside an explanation.
_MAX_SCOPE_IN_EXPLANATION = 240


_TERM_PATTERN = re.compile(r"[a-z0-9]+")
_CLAUSE_SPLIT = re.compile(r"[.;:!?\n]+|,")
_WHITESPACE = re.compile(r"\s+")

_MIN_TERM_LENGTH = 3

#: Procurement filler that carries no discriminating meaning. Kept small and
#: explicit: every entry here is a word an engineer would not use to describe a
#: *product*, so removing it cannot hide a real requirement term.
_STOPWORDS = frozenset(
    """
    a an and are as at be been being by for from in into is it its of on or per
    shall should such than that the their them then there these this those to
    up was were will with within without would
    any all each every other same
    supply supplies supplying procurement procure provide provides provision
    purchase purchasing deliver delivery item items material materials
    requirement requirements specification specifications work works
    applicable relevant
    """.split()
)


def _fold_plural(term: str) -> str:
    """Naive, deterministic plural folding (``motors`` -> ``motor``)."""
    if len(term) > 3 and term.endswith("s") and not term.endswith("ss"):
        return term[:-1]
    return term


def extract_terms(text: str) -> set[str]:
    """Return the distinctive lowercase terms of ``text``.

    Both the raw token and its plural-folded form are returned, so ``machines``
    matches ``machine`` without a stemming dependency. Words shorter than
    :data:`_MIN_TERM_LENGTH` and :data:`_STOPWORDS` are ignored.
    """
    terms: set[str] = set()
    for raw in _TERM_PATTERN.findall((text or "").lower()):
        if len(raw) < _MIN_TERM_LENGTH or raw in _STOPWORDS:
            continue
        terms.add(raw)
        folded = _fold_plural(raw)
        if folded not in _STOPWORDS and len(folded) >= _MIN_TERM_LENGTH:
            terms.add(folded)
    return terms


def split_clauses(text: str) -> list[str]:
    """Split a requirement into its individual clauses (verbatim fragments)."""
    clauses: list[str] = []
    for raw in _CLAUSE_SPLIT.split(text or ""):
        clause = _WHITESPACE.sub(" ", raw).strip(" -\u2013\u2014")
        if len(clause) >= _MIN_TERM_LENGTH:
            clauses.append(clause)
    return clauses



def group_chunks_by_standard(
    chunks: Iterable[RetrievedChunk],
) -> dict[str, list[RetrievedChunk]]:
    """Group retrieved chunks by the catalogue designation they belong to.

    Chunks without a designation (for example document chunks) are ignored: a
    recommendation can only ever be about a real catalogue record.
    """
    groups: dict[str, list[RetrievedChunk]] = {}
    for chunk in chunks:
        metadata = chunk.metadata or {}
        code = str(
            metadata.get("standard_number") or metadata.get("standard_code") or ""
        ).strip()
        if not code:
            continue
        groups.setdefault(code, []).append(chunk)
    return groups


@dataclass(slots=True)
class RankedStandard:
    """One catalogued standard ranked for a requirement."""

    standard_number: str
    title: str | None
    confidence: float
    retrieval_score: float
    matched_terms: tuple[str, ...] = ()
    matched_requirements: tuple[str, ...] = ()
    evidence: tuple[RetrievedChunk, ...] = ()
    chunk_count: int = 0
    rank: int = 0
    #: Rarity-weighted share of the requirement's technical attributes this record
    #: satisfies (``None`` when the requirement states no attributes at all).
    attribute_score: float | None = None
    #: Attribute matches, described without overstating them.
    matched_attributes: tuple[str, ...] = ()
    #: Requirement attributes this record's indexed text does not state.
    unmatched_attributes: tuple[str, ...] = ()


@dataclass(slots=True)
class RankingOutcome:
    """What a ranking run produced, including what it rejected and why."""

    candidates: list[RankedStandard] = field(default_factory=list)
    groups_seen: int = 0
    unverified: list[str] = field(default_factory=list)
    below_floor: list[str] = field(default_factory=list)
    min_score: float = DEFAULT_MIN_SCORE
    #: Attribute kinds the requirement asked for.
    attribute_kinds: tuple[str, ...] = ()
    #: Requirement kinds that *no* retrieved record mentions, so they could not
    #: contribute to any score (reported so the caller can say so explicitly).
    unmatchable_kinds: tuple[str, ...] = ()

    @property
    def rejected_count(self) -> int:
        """Number of retrieved records that were deliberately not recommended."""
        return len(self.unverified) + len(self.below_floor)


def score_confidence(
    retrieval_score: float,
    matched_term_count: int,
    attribute_score: float | None = None,
) -> float:
    """Blend the ranking signals into a 0..1 confidence value.

    With technical attributes in the requirement::

        confidence = 0.25 * similarity + 0.70 * attributes + 0.05 * text overlap

    Without any parseable attribute the attribute term is dropped and the weights
    become ``0.85 / 0.15`` (similarity / text overlap), so non-technical queries
    behave exactly as before.

    ``attribute_score`` is the rarity-weighted share of the requirement's stated
    technical attributes a record satisfies (see :func:`rank_standards`). The
    result is a *relevance* indicator for ranking, not a probability.
    """
    similarity = max(0.0, min(1.0, float(retrieval_score)))
    term_support = min(1.0, max(0, matched_term_count) / TERM_SUPPORT_TARGET)

    if attribute_score is None:
        blended = (
            SIMILARITY_WEIGHT_NO_ATTRIBUTES * similarity
            + TERM_SUPPORT_WEIGHT_NO_ATTRIBUTES * term_support
        )
    else:
        attributes = max(0.0, min(1.0, float(attribute_score)))
        blended = (
            SIMILARITY_WEIGHT * similarity
            + ATTRIBUTE_WEIGHT * attributes
            + TERM_SUPPORT_WEIGHT * term_support
        )
    return round(max(0.0, min(1.0, blended)), 4)


def attribute_rarity(mentioning: int, total: int) -> float:
    """How rare an attribute kind is inside the retrieved candidate pool.

    ``log(1 + N / (1 + df)) / log(1 + N)`` - 1.0 when no candidate mentions the
    kind (so a *matching* kind is highly discriminative) down to a small value
    when every candidate mentions it (a kind that cannot discriminate at all).
    """
    if total <= 0:
        return 1.0
    return math.log(1.0 + total / (1.0 + mentioning)) / math.log(1.0 + total)


def attribute_weights(
    required: Sequence[str], mentioning: dict[str, int], total: int
) -> dict[str, float]:
    """Weight each required attribute kind by tier (value vs category) and rarity."""
    return {
        kind: (STATED_VALUE_TIER if kind in VALUE_KINDS else CATEGORY_TIER)
        * attribute_rarity(mentioning.get(kind, 0), total)
        for kind in required
    }


def _attribute_terms(attributes: Iterable[TechnicalAttribute]) -> set[str]:
    """Tokens a (matched) attribute already accounts for (no double counting)."""
    terms: set[str] = set()
    for attribute in attributes:
        terms |= extract_terms(attribute.surface)
        if attribute.value:
            terms |= extract_terms(attribute.value)
    return terms



def _truncate(text: str, limit: int) -> str:
    text = _WHITESPACE.sub(" ", text).strip()
    return text if len(text) <= limit else f"{text[:limit].rstrip()}..."


def rank_standards(
    chunks: Sequence[RetrievedChunk],
    *,
    requirement: str,
    min_score: float = DEFAULT_MIN_SCORE,
    max_results: int = DEFAULT_MAX_RECOMMENDATIONS,
    catalogue_contains: Callable[[str], bool] | None = None,
) -> RankingOutcome:
    """Rank retrieved chunks into catalogued standard candidates.

    Ranking combines three signals (see :func:`score_confidence`):

    1. technical-attribute satisfaction - the requirement's IE class, power,
       voltage, frequency, IP rating, phase, motor type, mounting, cooling, duty
       and insulation values are matched against what the record's indexed text
       states, weighted by how rare that kind is in the retrieved pool;
    2. vector-store similarity, which a near-literal title match can inflate, so
       it is only secondary when attributes are present;
    3. text overlap of the remaining distinctive words.

    Args:
        chunks: retrieved chunks (real vector-store output).
        requirement: the text whose attributes, terms and clauses are matched.
        min_score: similarity floor; weaker records are rejected, not ranked.
        max_results: maximum number of candidates to return.
        catalogue_contains: optional lookup used to reject designations that are
            not part of the loaded, validated catalogue.
    """
    requirement_terms = extract_terms(requirement)
    clauses = split_clauses(requirement)
    requirement_attributes = extract_technical_attributes(requirement)
    requirement_attr_terms = _attribute_terms(requirement_attributes)
    groups = group_chunks_by_standard(chunks)

    group_texts = {
        code: "\n".join(chunk.text for chunk in group) for code, group in groups.items()
    }
    group_attributes = {
        code: extract_technical_attributes(text) for code, text in group_texts.items()
    }

    required = required_kinds(requirement_attributes)
    mentioning = {
        kind: sum(
            1
            for attributes in group_attributes.values()
            if any(attribute.kind == kind for attribute in attributes)
        )
        for kind in required
    }
    weights = attribute_weights(required, mentioning, len(groups))
    matchable = tuple(kind for kind in required if mentioning[kind] > 0)
    denominator = sum(weights[kind] for kind in matchable)

    outcome = RankingOutcome(
        groups_seen=len(groups),
        min_score=min_score,
        attribute_kinds=required,
        unmatchable_kinds=tuple(kind for kind in required if mentioning[kind] == 0),
    )

    def attribute_score_for(match: AttributeMatch) -> float | None:
        """Rarity-weighted share of the requirement's attributes this record meets."""
        if not denominator:
            return None
        total = sum(
            weights[detail.kind] * detail.quality
            for detail in match.details
            if detail.kind in matchable
        )
        return round(total / denominator, 4)

    def clause_matches(
        clause: str,
        available: Sequence[TechnicalAttribute],
        standard_terms: set[str],
    ) -> bool:
        """A clause counts as matched on text overlap or on a matched attribute."""
        if extract_terms(clause) & standard_terms:
            return True
        clause_attributes = extract_technical_attributes(clause)
        if not clause_attributes:
            return False
        return bool(match_attributes(clause_attributes, available).matched)


    for code, group in groups.items():
        if catalogue_contains is not None and not catalogue_contains(code):
            outcome.unverified.append(code)
            continue

        best = max(group, key=lambda chunk: chunk.score)
        retrieval_score = float(best.score)
        if retrieval_score < min_score:
            outcome.below_floor.append(code)
            continue

        standard_text = group_texts[code]
        standard_terms = extract_terms(standard_text)
        available = group_attributes[code]

        attribute_match = match_attributes(requirement_attributes, available)
        attribute_score = attribute_score_for(attribute_match)

        # Words the attributes already account for are not counted twice: the
        # text-overlap signal only sees what is left.
        attribute_terms = requirement_attr_terms | _attribute_terms(available)
        matched_terms = tuple(sorted(requirement_terms & standard_terms))
        generic_matched = tuple(
            term for term in matched_terms if term not in attribute_terms
        )
        matched_clauses = tuple(
            clause
            for clause in clauses
            if clause_matches(clause, available, standard_terms)
        )
        ordered_evidence = tuple(sorted(group, key=lambda chunk: -chunk.score))

        outcome.candidates.append(
            RankedStandard(
                standard_number=code,
                title=(best.metadata or {}).get("title"),
                confidence=score_confidence(
                    retrieval_score, len(generic_matched), attribute_score
                ),
                retrieval_score=retrieval_score,
                matched_terms=matched_terms,
                matched_requirements=matched_clauses,
                evidence=ordered_evidence,
                chunk_count=len(group),
                attribute_score=attribute_score,
                matched_attributes=tuple(
                    describe_matched(detail) for detail in attribute_match.matched
                ),
                unmatched_attributes=tuple(
                    describe_unmatched(detail) for detail in attribute_match.unmatched
                ),
            )
        )

    outcome.candidates.sort(
        key=lambda candidate: (
            -candidate.confidence,
            -candidate.retrieval_score,
            candidate.standard_number,
        )
    )
    del outcome.candidates[max_results:]
    for position, candidate in enumerate(outcome.candidates, start=1):
        candidate.rank = position
    return outcome


def explain_match(
    *,
    rank: int,
    standard_number: str,
    retrieval_score: float,
    matched_terms: Sequence[str] = (),
    scope: str | None = None,
    semantic_only: bool = False,
    attribute_score: float | None = None,
    matched_attributes: Sequence[str] = (),
    unmatched_attributes: Sequence[str] = (),
) -> str:
    """Deterministic, evidence-based explanation of why a record was ranked.

    Only facts that exist in the catalogue/retrieval output are stated, and the
    text says explicitly that it is a retrieval-based match rather than an expert
    (or LLM) determination of applicability. Attribute matches are reported with
    their strength (an exact value versus family coverage) and the requirement
    attributes the record does not state are named as well.
    """
    basis = f"semantic similarity ({retrieval_score:.3f})"
    if attribute_score is not None:
        basis += f" and technical-attribute coverage ({attribute_score:.2f})"
    parts = [
        f"Ranked #{rank} by {basis} against the verified BIS catalogue record "
        f"{standard_number}."
    ]
    if matched_attributes:
        parts.append(
            "Technical attributes matched: " + "; ".join(matched_attributes) + "."
        )
    if unmatched_attributes:
        parts.append(
            "Requirement attributes not stated in this record's indexed text: "
            + "; ".join(unmatched_attributes)
            + "."
        )
    if matched_terms:
        parts.append(
            "Distinctive terms shared with the requirement: "
            + ", ".join(matched_terms[:10])
            + "."
        )
    if scope and scope.strip():
        parts.append("Catalogue scope: " + _truncate(scope, _MAX_SCOPE_IN_EXPLANATION))
    if semantic_only:
        parts.append(
            "No distinctive requirement term or technical attribute appears in the "
            "indexed record, so this candidate rests on semantic similarity alone."
        )
    parts.append(
        "This is a retrieval-based match on the verified catalogue text shown as "
        "evidence; it is not a determination that the standard applies."
    )
    return " ".join(parts)


__all__ = [
    "ATTRIBUTE_WEIGHT",
    "DEFAULT_MAX_RECOMMENDATIONS",
    "DEFAULT_MIN_SCORE",
    "SIMILARITY_WEIGHT",
    "STATED_VALUE_TIER",
    "TERM_SUPPORT_TARGET",
    "RankedStandard",
    "RankingOutcome",
    "attribute_rarity",
    "attribute_weights",
    "explain_match",
    "extract_terms",
    "group_chunks_by_standard",
    "rank_standards",
    "score_confidence",
    "split_clauses",
]
