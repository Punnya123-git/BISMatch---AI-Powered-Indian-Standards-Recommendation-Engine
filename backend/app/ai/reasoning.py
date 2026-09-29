"""LLM reasoning stage: final applicability classification and ranking.

Architecture reminder: the deterministic retrieval/ranking layer
(:mod:`app.rag.ranking`) is the *candidate-generation / supporting-signal*
layer. It retrieves, catalogue-verifies, floors and scores candidates, but its
scores never decide applicability. The LLM consumes the verified candidates
plus their evidence and produces the final applicability classification and
ranking, grounded strictly in the supplied BIS-based material.

Grounding rules (enforced here, never left to the model):

* only designations present in the supplied candidate set can survive;
* any invented ``standard_number`` is dropped and reported;
* evidence chunk ids are restricted to the candidate's own chunks;
* a verdict whose JSON envelope had to be *repaired* (the model never closed it,
  whether or not the provider said so) is flagged as truncated, because the
  candidates it never reached must not be reported as excluded;
* when the model call fails or yields nothing usable the outcome carries an
  error, and the caller makes **no** final recommendation. Retrieval output is
  never promoted to a recommendation as a fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from app.ai.llm.base import LLMProvider, LLMRequest
from app.ai.prompts.templates import (
    STANDARD_RECOMMENDATION_PROMPT,
    STANDARD_RECOMMENDATION_SYSTEM_PROMPT,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Applicability classes the model may assign.
APPLICABILITY_TYPES = ("direct", "related", "needs_verification")

_MAX_LIST_ITEMS = 12

#: Longest catalogue scope quoted per candidate. Scope is the field that actually
#: drives applicability, so it is kept generously.
_SCOPE_CHARS = 900

#: Verbatim index-text anchor kept per evidence chunk.
#:
#: The chunk text is the indexed catalogue document, which already contains the
#: title, the scope, the certification statement and the references - all of which
#: are sent as their *own* fields below. Re-sending them inside every snippet was
#: by far the largest source of duplication in the prompt, so the snippet is kept
#: only as a short verbatim anchor proving where the record came from.
_SNIPPET_CHARS = 140

#: Per-item cap and count for the deterministic signal strings. These are hints,
#: not decisions (the prompt says so), and an unmatched-attribute sentence is
#: near-identical for every candidate, so a short clip keeps the signal without
#: repeating the same sentence ten times.
_SIGNAL_CHARS = 44
_MAX_SIGNAL_ITEMS = 3

_INSUFFICIENT_EVIDENCE_NOTE = (
    "Insufficient evidence to determine applicability; verification is required."
)
_NEEDS_VERIFICATION_CONFIDENCE_CAP = 0.6


def normalize_designation(value: str | None) -> str:
    """Fold a designation for comparison (``IS-12615:2018`` == ``is 12615 2018``)."""
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _clip(text: str | None, limit: int) -> str | None:
    if not text:
        return None
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _clean_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for raw in value:
        text = " ".join(str(raw).split())
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        items.append(text)
        if len(items) >= _MAX_LIST_ITEMS:
            break
    return items


def _clip_items(values: Iterable[str], limit: int = _SIGNAL_CHARS) -> list[str]:
    """Clip signal strings, dropping empties, keeping at most :data:`_MAX_SIGNAL_ITEMS`."""
    items: list[str] = []
    for raw in values:
        clipped = _clip(str(raw), limit)
        if clipped:
            items.append(clipped)
        if len(items) >= _MAX_SIGNAL_ITEMS:
            break
    return items


@dataclass(slots=True)
class CandidateEvidence:
    """One retrieved chunk backing a candidate (verbatim index text)."""

    chunk_id: str
    snippet: str
    page_number: int | None = None
    score: float | None = None

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "snippet": _clip(self.snippet, _SNIPPET_CHARS),
        }
        if self.page_number is not None:
            payload["page_number"] = self.page_number
        if self.score is not None:
            payload["similarity"] = round(float(self.score), 4)
        return payload


@dataclass(slots=True)
class CandidateStandard:
    """Verified candidate handed to the model (catalogue data + signals)."""

    standard_number: str
    title: str | None = None
    scope: str | None = None
    category: str | None = None
    status: str | None = None
    year: int | None = None
    revision: str | None = None
    amendments: tuple[str, ...] = ()
    certification: tuple[str, ...] = ()
    related_standards: tuple[str, ...] = ()
    deterministic_rank: int = 0
    similarity_score: float | None = None
    attribute_score: float | None = None
    matched_attributes: tuple[str, ...] = ()
    unmatched_attributes: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = ()
    evidence: tuple[CandidateEvidence, ...] = ()

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "standard_number": self.standard_number,
            "title": self.title,
            "scope": _clip(self.scope, _SCOPE_CHARS),
            "category": self.category,
            "status": self.status,
            "year": self.year,
            "revision": self.revision,
            "amendments": list(self.amendments),
            "certification": list(self.certification),
            "related_standards": list(self.related_standards),
            # Signals are hints, not decisions. Keys are deliberately short and the
            # descriptive strings are clipped: they are repeated for every candidate
            # and the requirement's attributes are already listed once, separately,
            # in the prompt.
            "deterministic_signals": {
                "det_rank": self.deterministic_rank,
                "cos": (
                    round(float(self.similarity_score), 4)
                    if self.similarity_score is not None
                    else None
                ),
                "attr": (
                    round(float(self.attribute_score), 4)
                    if self.attribute_score is not None
                    else None
                ),
                "matched_attributes": _clip_items(self.matched_attributes),
                "unmatched_attributes": _clip_items(self.unmatched_attributes),
                "matched_terms": _clip_items(self.matched_terms),
            },
            "evidence": [item.to_payload() for item in self.evidence],
        }
        return {k: v for k, v in payload.items() if v not in (None, [], ())}

@dataclass(slots=True)
class ReasonedStandard:
    """One model verdict for a verified candidate (order = final ranking)."""

    standard_number: str
    applicability_type: str
    confidence: float
    why_it_matches: str
    matched_requirements: list[str] = field(default_factory=list)
    uncovered_requirements: list[str] = field(default_factory=list)
    evidence_chunk_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ProcurementTarget:
    """What the model concluded is actually being procured.

    Captured from the model rather than derived, so the applicability judgement
    stays inspectable: the response can show the user the same target the
    standards were judged against. This is never used to make an applicability
    decision in code - it is the model's own statement, surfaced for transparency.
    """

    target: str
    explicit_specifications: list[str] = field(default_factory=list)
    note: str = ""


@dataclass(slots=True)
class ReasoningOutcome:
    """What the reasoning stage produced (possibly nothing usable)."""

    recommendations: list[ReasonedStandard] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    provider: str = ""
    model: str | None = None
    error: str | None = None
    #: The model's own statement of what is actually being procured. Recorded for
    #: transparency so the user sees the same target the standards were judged
    #: against. It is never used to make an applicability decision in code - that
    #: would make the deterministic layer the authority instead of the model.
    procurement_target: "ProcurementTarget | None" = None
    #: True when the provider stopped generating because the token limit was hit.
    #: The verdict is then *incomplete* - the model simply never got to mention
    #: the remaining candidates - so it must not be used to exclude anything.
    truncated: bool = False
    #: Token accounting reported by the provider, for diagnostics.
    usage: dict[str, int] = field(default_factory=dict)


#: Matching closers for the two JSON container characters.
_CLOSERS = {"{": "}", "[": "]"}


def _strip_fences(text: str) -> str:
    """Remove a surrounding markdown code fence (`` ```json ... ``` ``)."""
    cleaned = (text or "").strip()
    if not cleaned.startswith("```"):
        return cleaned
    newline = cleaned.find("\n")
    cleaned = cleaned[newline + 1 :] if newline != -1 else cleaned[3:]
    cleaned = cleaned.strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned


def _end_of_object(text: str, start: int) -> int | None:
    """Index just past the ``}`` that closes the object opened at ``start``.

    String-aware, so a brace inside a quoted value is not mistaken for the end
    of the object (the previous first-brace/last-brace slice could not do this).
    Returns ``None`` when the object is never closed.
    """
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in _CLOSERS:
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                return index + 1
    return None


def _repair_truncated(fragment: str) -> str | None:
    """Close an object the model stopped writing (typically ``finish_reason=length``).

    The half-written tail is *discarded*, never guessed: the fragment is cut back
    to the last element that was fully terminated, and the containers left open
    are closed in reverse order. Nothing is invented, so the result still has to
    survive the caller's strict validation.

    Returns ``None`` when the fragment is not truncated (nothing to repair).
    """
    # One frame per open container; ``cut`` is the index of the last separator
    # seen directly inside it, i.e. the point after the last *complete* element.
    stack: list[tuple[str, int, int]] = []
    in_string = False
    escaped = False
    for index, char in enumerate(fragment):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in _CLOSERS:
            stack.append((char, index, index + 1))
        elif char in "}]" and stack and _CLOSERS[stack[-1][0]] == char:
            stack.pop()
        elif char == "," and stack:
            opener, at, cut = stack[-1]
            stack[-1] = (opener, at, index)

    if not stack and not in_string:
        return None

    # Deepest container that already holds at least one complete element.
    deepest = None
    for position in range(len(stack) - 1, -1, -1):
        opener, at, cut = stack[position]
        if cut > at + 1:
            deepest = position
            break

    if deepest is None:
        # Nothing was completed anywhere: keep only the outermost container.
        opener, at, _ = stack[0]
        return fragment[at : at + 1] + _CLOSERS[opener]

    _, _, cut = stack[deepest]
    # Every frame still on the stack is an open container: close them innermost
    # first, then keep the fragment up to (but excluding) the dangling comma.
    closers = "".join(
        _CLOSERS[stack[i][0]] for i in range(len(stack) - 1, -1, -1)
    )
    return fragment[:cut] + closers


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the verdict JSON object out of raw model text.

    Tolerates what real models actually emit - surrounding whitespace, markdown
    code fences, a sentence before or after the object, braces inside strings -
    and repairs a response cut off mid-object by the token limit.

    This only makes the *envelope* recoverable. Every field inside is still
    validated by :func:`parse_reasoning_response`, so nothing unsafe is accepted
    merely because it happened to parse.
    """
    parsed, _ = extract_json_object_with_status(text)
    return parsed


def extract_json_object_with_status(text: str) -> tuple[dict[str, Any], bool]:
    """Extract the verdict object and report whether it had to be *repaired*.

    The second element is ``True`` only when the object was never closed and
    :func:`_repair_truncated` had to close it. That means the model's output is
    **incomplete**: anything it did not get to write is simply missing, so the
    caller must not read the absence of a candidate as a decision about it.
    """
    cleaned = _strip_fences(text)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("LLM response contained no JSON object.")

    end = _end_of_object(cleaned, start)
    if end is not None:
        parsed = json.loads(cleaned[start:end])
        repaired = False
    else:
        repaired_source = _repair_truncated(cleaned[start:])
        if repaired_source is None:
            raise ValueError("LLM response contained no complete JSON object.")
        parsed = json.loads(repaired_source)
        repaired = True

    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON is not an object.")
    return parsed, repaired


def _clamp_confidence(raw: Any, fallback: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return fallback
    if value != value:
        return fallback
    return max(0.0, min(1.0, value))



def parse_reasoning_response(
    text: str,
    candidates: Sequence[CandidateStandard],
    *,
    provider: str = "",
    model: str | None = None,
) -> ReasoningOutcome:
    """Validate a raw model response against the verified candidate set."""
    outcome = ReasoningOutcome(provider=provider, model=model)
    allowed = {normalize_designation(c.standard_number): c for c in candidates}
    try:
        data, repaired = extract_json_object_with_status(text)
    except (ValueError, json.JSONDecodeError) as exc:
        outcome.error = f"Could not parse the AI verdict as JSON: {exc}"
        return outcome

    # A repaired envelope means the model's output was cut off before it closed
    # the JSON object, whether or not the provider reported it via
    # ``finish_reason``. The verdict is therefore *incomplete*: candidates the
    # model never reached are absent for that reason alone, so they must never be
    # reported as "excluded". Flagging it here makes the caller treat a silently
    # truncated verdict exactly like a reported token-limit cut-off.
    outcome.truncated = repaired

    # Record the model's own statement of the procurement target. Read for
    # transparency only: it is never used to accept or reject a candidate in
    # code, because the model must remain the applicability authority.
    raw_target = data.get("procurement_target")
    if isinstance(raw_target, dict):
        target_text = " ".join(str(raw_target.get("target") or "").split())
        if target_text:
            outcome.procurement_target = ProcurementTarget(
                target=target_text,
                explicit_specifications=_clean_items(
                    raw_target.get("explicit_specifications")
                ),
                note=" ".join(str(raw_target.get("note") or "").split()),
            )

    raw_items = data.get("recommendations")
    if not isinstance(raw_items, list):
        outcome.error = (
            "The AI verdict did not contain a 'recommendations' list, so no "
            "applicability decision can be read from it."
        )
        return outcome
    if not raw_items:
        # An explicit empty selection is a *decision*: the model was shown the
        # verified candidates and concluded that none of them applies. It is
        # reported as "no applicable standard among the verified candidates",
        # never as a provider failure and never backfilled with retrieval output.
        outcome.excluded = [c.standard_number for c in candidates]
        return outcome
    seen: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        designation = str(raw.get("standard_number") or "").strip()
        key = normalize_designation(designation)
        candidate = allowed.get(key)
        if candidate is None:
            if designation:
                outcome.dropped.append(designation)
            continue
        if key in seen:
            continue
        seen.add(key)
        applicability = str(raw.get("applicability_type") or "").strip().lower()
        if applicability not in APPLICABILITY_TYPES:
            applicability = "needs_verification"
        conf = _clamp_confidence(raw.get("confidence"), fallback=0.5)
        if applicability == "needs_verification":
            conf = min(conf, _NEEDS_VERIFICATION_CONFIDENCE_CAP)
        reason = " ".join(str(raw.get("why_it_matches") or "").split())
        matched = _clean_items(raw.get("matched_requirements"))
        uncovered = _clean_items(raw.get("uncovered_requirements"))
        if not reason:
            reason = (
                _INSUFFICIENT_EVIDENCE_NOTE
                if applicability == "needs_verification"
                else f"Classified as '{applicability}' from retrieved evidence."
            )
        known_chunks = {item.chunk_id for item in candidate.evidence}
        evidence_ids = [
            str(cid)
            for cid in _clean_items(raw.get("evidence_chunk_ids"))
            if str(cid) in known_chunks
        ]
        outcome.recommendations.append(
            ReasonedStandard(
                standard_number=candidate.standard_number,
                applicability_type=applicability,
                confidence=conf,
                why_it_matches=reason,
                matched_requirements=matched,
                uncovered_requirements=uncovered,
                evidence_chunk_ids=evidence_ids,
            )
        )
    if not outcome.recommendations:
        outcome.error = (
            "The AI verdict named no verifiable candidate standard, so no "
            "applicability recommendation can be made."
        )
        return outcome
    outcome.excluded = [
        c.standard_number
        for c in candidates
        if normalize_designation(c.standard_number) not in seen
    ]
    return outcome


def run_ai_reasoning(
    *,
    provider: LLMProvider,
    requirement: str,
    attributes: Iterable[Mapping[str, Any]] | None,
    candidates: Sequence[CandidateStandard],
    temperature: float = 0.0,
    max_tokens: int = 2000,
) -> ReasoningOutcome:
    """Run the configured LLM over verified candidates and parse its verdict."""
    # Compact separators: the candidates are machine-read JSON, and pretty-printing
    # ten nested records added a large amount of whitespace to the prompt for no
    # gain in how well the model reads it.
    context = json.dumps(
        [c.to_payload() for c in candidates],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    lines = [
        " - "
        + (
            f"{item.get('kind')}: {item.get('value')}"
            if item.get("value")
            else f"{item.get('kind')} (family in scope)"
        )
        for item in (attributes or [])
    ]
    prompt = STANDARD_RECOMMENDATION_PROMPT.format(
        requirement=requirement.strip(),
        attributes=("\n".join(lines) if lines else "(none extracted)"),
        context=context,
        allowed_standards="\n".join(
            f" - {c.standard_number}" for c in candidates
        ),
    )
    response = provider.generate(
        LLMRequest(
            prompt=prompt,
            system_prompt=STANDARD_RECOMMENDATION_SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )
    outcome = parse_reasoning_response(
        response.text, candidates, provider=response.provider, model=response.model
    )
    # Truncation has two possible tell-tales and either is decisive:
    # ``finish_reason == "length"`` (the provider says the token limit stopped
    # generation) or a JSON envelope that had to be repaired because the model
    # never closed it. Both mean the verdict is a *fragment*, so it is labelled
    # as such and the caller decides whether a partial verdict may be trusted (it
    # may not be used to exclude candidates the model never actually reached).
    outcome.usage = dict(response.usage)
    outcome.truncated = outcome.truncated or response.finish_reason == "length"
    logger.info(
        "AI reasoning via %s classified %d/%d candidates (%d invented dropped)"
        "%s.",
        response.provider,
        len(outcome.recommendations),
        len(candidates),
        len(outcome.dropped),
        "; response TRUNCATED at the token limit" if outcome.truncated else "",
    )
    return outcome
