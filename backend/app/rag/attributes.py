"""Structured extraction and matching of technical procurement attributes.

The embedding model cannot tell "IE3" from "IP55" from a frame number, so the
ranking stage needs an explicit, auditable view of the technical attributes a
requirement asks for and a catalogue record actually states. This module provides
exactly that, and nothing more:

* extraction is pure regex over the *text that exists* (the requirement, and the
  indexed catalogue record text) - no value is ever inferred;
* every attribute keeps the verbatim ``surface`` it was read from, so a
  recommendation can quote what it matched;
* matching distinguishes an *exact value* match (``IE3`` stated in the record), a
  *family* match (the record covers the IP code classification but does not state
  IP55) and *no match* (the record states some other explicit value);
* the vocabulary is a generic engineering vocabulary (IE classes, power, voltage,
  frequency, enclosure/IP, phase, motor type, mounting, cooling, duty,
  insulation) - it is not tailored to one query or one standard.

Attribute *weights* are deliberately not defined here: rarity is measured by the
ranking stage against the retrieved candidate pool (see :mod:`app.rag.ranking`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

# --- attribute kinds ---------------------------------------------------------
EFFICIENCY_CLASS = "efficiency_class"
POWER = "power"
VOLTAGE = "voltage"
FREQUENCY = "frequency"
ENCLOSURE = "enclosure"
PHASE = "phase"
MOTOR_TYPE = "motor_type"
MOUNTING = "mounting"
COOLING = "cooling"
DUTY = "duty"
INSULATION = "insulation"

#: Human readable kind names used in explanations and warnings.
KIND_LABELS: dict[str, str] = {
    EFFICIENCY_CLASS: "IE efficiency class",
    POWER: "power rating",
    VOLTAGE: "voltage",
    FREQUENCY: "frequency",
    ENCLOSURE: "IP enclosure rating",
    PHASE: "phase",
    MOTOR_TYPE: "motor type",
    MOUNTING: "mounting",
    COOLING: "cooling method",
    DUTY: "duty",
    INSULATION: "insulation class",
}

#: Kinds that state a *specified value* (a performance/protection figure or code).
#: These outweigh product-classification kinds when the requirement states them.
VALUE_KINDS = frozenset(
    {EFFICIENCY_CLASS, POWER, VOLTAGE, FREQUENCY, ENCLOSURE, COOLING, DUTY, INSULATION}
)

#: Kinds that classify the product itself (phase, motor type, mounting style).
CATEGORY_KINDS = frozenset({PHASE, MOTOR_TYPE, MOUNTING})

#: Kinds for which "the record covers this family of attributes without stating
#: the value" is meaningful, checkable evidence: the family *is* the subject of
#: the record ("Efficiency Classes ... (IE Code)", "Degrees of Protection ...
#: (IP Code)", "Methods of Cooling", "insulation class"). Engineering *quantities*
#: (power, voltage, frequency) are deliberately excluded: a passing phrase such as
#: "voltage range" in a scope sentence is not evidence of a rating, so only a
#: stated figure (or a range containing it) counts for those kinds.
FAMILY_COVERAGE_KINDS = frozenset({EFFICIENCY_CLASS, ENCLOSURE, COOLING, INSULATION})

#: How much credit a family-only match earns (an exact value earns 1.0).
FAMILY_MATCH_QUALITY = 0.5

#: Tolerance when comparing engineering quantities (power, voltage).
QUANTITY_TOLERANCE = 0.01

# --- value patterns ---------------------------------------------------------
_IE_VALUE = re.compile(r"\bIE\s?-?\s?([1-4])\b", re.IGNORECASE)
_IE_FAMILY = re.compile(
    r"\bIE\s?Code\b|\bEfficiency\s+Classes?\b|\bIEC\s*60034-30\b", re.IGNORECASE
)
_POWER_VALUE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(kilowatt|megawatt|horsepower|kw|mw|hp|w)\b", re.IGNORECASE
)
_POWER_RANGE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:kw|w|hp)\s*(?:to|up to|and above|-|\u2013)\s*"
    r"(\d+(?:\.\d+)?)?\s*(?:kw|w|hp)\b",
    re.IGNORECASE,
)
_VOLTAGE_VALUE = re.compile(r"(\d+(?:\.\d+)?)\s*(kilovolt|volts?|kv|v)\b", re.IGNORECASE)
_FREQUENCY_VALUE = re.compile(r"(\d+(?:\.\d+)?)\s*(hertz|hz)\b", re.IGNORECASE)
_ENCLOSURE_VALUE = re.compile(r"\bIP\s?(\d{2})(?:\s?([A-Z]))?\b", re.IGNORECASE)
_ENCLOSURE_FAMILY = re.compile(
    r"\bIP\s?Code\b|\bDegrees?\s+of\s+Protection\b|\bIngress\s+Protection\b", re.IGNORECASE
)
_PHASE_ONE = re.compile(r"\b(?:single|one|mono)\s*-?\s*phase\b|\b1\s?ph\b", re.IGNORECASE)
_PHASE_TWO = re.compile(r"\btwo\s*-?\s*phase\b|\b2\s?ph\b", re.IGNORECASE)
_PHASE_THREE = re.compile(r"\b(?:three|3)\s*-?\s*phase\b|\b3\s?ph\b", re.IGNORECASE)
_COOLING_VALUE = re.compile(r"\bIC\s?(\d{2,3}[A-Z]?)\b", re.IGNORECASE)
_COOLING_FAMILY = re.compile(r"\bmethods?\s+of\s+cooling\b|\bcooling\b", re.IGNORECASE)
_DUTY_VALUE = re.compile(r"\bS([1-9])\b")
_INSULATION_VALUE = re.compile(r"\bClass\s+([FBH])\b", re.IGNORECASE)
_INSULATION_FAMILY = re.compile(r"\binsulation\s+class(?:es)?\b", re.IGNORECASE)

_WHITESPACE = re.compile(r"\s+")

#: Motor-type vocabulary (canonical value, pattern).
_MOTOR_TYPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("squirrel-cage", re.compile(r"\bsquirrel[\s-]*cage\b", re.IGNORECASE)),
    ("slip-ring", re.compile(r"\bslip[\s-]*ring\b|\bwound[\s-]*rotor\b", re.IGNORECASE)),
    ("induction", re.compile(r"\binduction\b|\basynchronous\b", re.IGNORECASE)),
    ("synchronous", re.compile(r"\bsynchronous\b", re.IGNORECASE)),
    ("dc", re.compile(r"\bdirect[\s-]*current\b|\bDC\b", re.IGNORECASE)),
    ("universal", re.compile(r"\buniversal\b", re.IGNORECASE)),
    ("submersible", re.compile(r"\bsubmersible\b", re.IGNORECASE)),
    ("traction", re.compile(r"\btraction\b", re.IGNORECASE)),
    ("brake-motor", re.compile(r"\bbrake[\s-]*motor\b", re.IGNORECASE)),
    ("geared", re.compile(r"\bgear(?:ed|[\s-]*motor)\b", re.IGNORECASE)),
    ("permanent-magnet", re.compile(r"\bpermanent[\s-]*magnet\b", re.IGNORECASE)),
    ("capacitor", re.compile(r"\bcapacitor\b", re.IGNORECASE)),
    ("shaded-pole", re.compile(r"\bshaded[\s-]*pole\b", re.IGNORECASE)),
)

#: Mounting vocabulary (canonical value, pattern).
_MOUNTINGS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("foot", re.compile(r"\bfoot\b|\bB3\b", re.IGNORECASE)),
    ("flange", re.compile(r"\bflange\b|\bB5\b", re.IGNORECASE)),
    ("face", re.compile(r"\bface[\s-]*mount\w*\b|\bB14\b", re.IGNORECASE)),
)

_POWER_TO_KW = {"w": 0.001, "kw": 1.0, "mw": 1000.0, "hp": 0.7457}
_VOLTAGE_TO_V = {"v": 1.0, "kv": 1000.0}


@dataclass(frozen=True, slots=True)
class TechnicalAttribute:
    """One technical attribute read out of a text.

    ``value`` is a canonical, comparable form (``ie3``, ``kw:15``, ``ip55``,
    ``phase:3``, ``squirrel-cage``, ``foot``...). It is ``None`` when the text
    only shows that a *family* of attributes is in scope (for example a record
    whose title says "Efficiency Classes ... (IE Code)" without naming a class).
    ``surface`` is the verbatim fragment the attribute was read from.
    """

    kind: str
    value: str | None
    surface: str

    @property
    def label(self) -> str:
        """Human readable description, e.g. ``IP enclosure rating IP55``."""
        kind = KIND_LABELS.get(self.kind, self.kind)
        return f"{kind} {self.surface}" if self.surface else kind


def _found(pattern: re.Pattern[str], text: str) -> str | None:
    """Return the verbatim match of ``pattern`` in ``text``, if any."""
    match = pattern.search(text)
    return _WHITESPACE.sub(" ", match.group(0)).strip() if match else None


def _attribute(kind: str, value: str | None, surface: str) -> TechnicalAttribute:
    return TechnicalAttribute(kind=kind, value=value, surface=surface)


def _efficiency_attributes(text: str) -> list[TechnicalAttribute]:
    attributes = [
        _attribute(EFFICIENCY_CLASS, f"ie{match.group(1)}", _WHITESPACE.sub(" ", match.group(0)))
        for match in _IE_VALUE.finditer(text)
    ]
    family = _found(_IE_FAMILY, text)
    if family:
        attributes.append(_attribute(EFFICIENCY_CLASS, None, family))
    return attributes


def _power_attributes(text: str) -> list[TechnicalAttribute]:
    attributes: list[TechnicalAttribute] = []
    for raw, unit in _POWER_VALUE.findall(text):
        factor = _POWER_TO_KW.get(unit.lower())
        if factor is None:
            continue
        surface = _WHITESPACE.sub(" ", f"{raw} {unit}")
        attributes.append(_attribute(POWER, f"kw:{float(raw) * factor:g}", surface))
    for match in _POWER_RANGE.finditer(text):
        low, high = match.group(1), match.group(2)
        span = f"{float(low) * 1.0:g}-{float(high) * 1.0:g}" if high else f"{float(low):g}+"
        attributes.append(
            _attribute(POWER, f"kw-range:{span}", _WHITESPACE.sub(" ", match.group(0)))
        )
    return attributes


def _voltage_attributes(text: str) -> list[TechnicalAttribute]:
    attributes: list[TechnicalAttribute] = []
    for raw, unit in _VOLTAGE_VALUE.findall(text):
        factor = _VOLTAGE_TO_V.get(unit.lower())
        if factor is None:
            continue
        surface = _WHITESPACE.sub(" ", f"{raw} {unit}")
        attributes.append(_attribute(VOLTAGE, f"v:{float(raw) * factor:g}", surface))
    return attributes


def _frequency_attributes(text: str) -> list[TechnicalAttribute]:
    return [
        _attribute(FREQUENCY, f"hz:{float(raw):g}", _WHITESPACE.sub(" ", f"{raw} {unit}"))
        for raw, unit in _FREQUENCY_VALUE.findall(text)
    ]


def _enclosure_attributes(text: str) -> list[TechnicalAttribute]:
    attributes: list[TechnicalAttribute] = []
    for match in _ENCLOSURE_VALUE.finditer(text):
        digits, suffix = match.group(1), (match.group(2) or "")
        surface = _WHITESPACE.sub(" ", match.group(0))
        attributes.append(_attribute(ENCLOSURE, f"ip{digits}{suffix.lower()}", surface))
    family = _found(_ENCLOSURE_FAMILY, text)
    if family:
        attributes.append(_attribute(ENCLOSURE, None, family))
    return attributes


def _phase_attributes(text: str) -> list[TechnicalAttribute]:
    attributes: list[TechnicalAttribute] = []
    for pattern, value in ((_PHASE_ONE, "1"), (_PHASE_TWO, "2"), (_PHASE_THREE, "3")):
        for match in pattern.finditer(text):
            attributes.append(
                _attribute(PHASE, f"phase:{value}", _WHITESPACE.sub(" ", match.group(0)))
            )
    return attributes


def _vocabulary_attributes(
    text: str, kind: str, vocabulary: Iterable[tuple[str, re.Pattern[str]]]
) -> list[TechnicalAttribute]:
    attributes: list[TechnicalAttribute] = []
    for value, pattern in vocabulary:
        for match in pattern.finditer(text):
            attributes.append(_attribute(kind, value, _WHITESPACE.sub(" ", match.group(0))))
    return attributes


def _cooling_attributes(text: str) -> list[TechnicalAttribute]:
    attributes = [
        _attribute(COOLING, f"ic{match.group(1).upper()}", _WHITESPACE.sub(" ", match.group(0)))
        for match in _COOLING_VALUE.finditer(text)
    ]
    family = _found(_COOLING_FAMILY, text)
    if family:
        attributes.append(_attribute(COOLING, None, family))
    return attributes


def _duty_attributes(text: str) -> list[TechnicalAttribute]:
    return [
        _attribute(DUTY, f"s{match.group(1)}", _WHITESPACE.sub(" ", match.group(0)))
        for match in _DUTY_VALUE.finditer(text)
    ]


def _insulation_attributes(text: str) -> list[TechnicalAttribute]:
    attributes = [
        _attribute(
            INSULATION, f"class-{match.group(1).upper()}", _WHITESPACE.sub(" ", match.group(0))
        )
        for match in _INSULATION_VALUE.finditer(text)
    ]
    family = _found(_INSULATION_FAMILY, text)
    if family:
        attributes.append(_attribute(INSULATION, None, family))
    return attributes


def extract_technical_attributes(text: str) -> tuple[TechnicalAttribute, ...]:
    """Read every technical attribute present in ``text`` (grouped by kind).

    Duplicates (same kind and canonical value, or the same family surface) are
    collapsed, and the result is deterministic for a given input.
    """
    text = text or ""
    found: list[TechnicalAttribute] = []
    found.extend(_efficiency_attributes(text))
    found.extend(_power_attributes(text))
    found.extend(_voltage_attributes(text))
    found.extend(_frequency_attributes(text))
    found.extend(_enclosure_attributes(text))
    found.extend(_phase_attributes(text))
    found.extend(_vocabulary_attributes(text, MOTOR_TYPE, _MOTOR_TYPES))
    found.extend(_vocabulary_attributes(text, MOUNTING, _MOUNTINGS))
    found.extend(_cooling_attributes(text))
    found.extend(_duty_attributes(text))
    found.extend(_insulation_attributes(text))

    deduped: dict[tuple[str, str], TechnicalAttribute] = {}
    for attribute in found:
        key = (attribute.kind, attribute.value or f"family:{attribute.surface.lower()}")
        deduped.setdefault(key, attribute)
    return tuple(deduped.values())


# --- matching ---------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AttributeMatchDetail:
    """How one *kind* of requirement attribute fared against a record."""

    kind: str
    required: tuple[str, ...]
    quality: float
    exact_values: tuple[str, ...] = ()
    family_values: tuple[str, ...] = ()
    missing_values: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)

    @property
    def values(self) -> str:
        return ", ".join(self.required)

    @property
    def exact(self) -> bool:
        """True when every required value in this kind matched exactly."""
        return bool(self.exact_values) and not (
            self.family_values or self.missing_values
        )

    @property
    def family_only(self) -> bool:
        """True when the record covers the family but states no required value."""
        return not self.exact_values and bool(self.family_values)


@dataclass(frozen=True, slots=True)
class AttributeMatch:
    """The full attribute comparison for one record (deterministic order)."""

    details: tuple[AttributeMatchDetail, ...] = ()

    @property
    def matched(self) -> tuple[AttributeMatchDetail, ...]:
        return tuple(detail for detail in self.details if detail.quality > 0)

    @property
    def unmatched(self) -> tuple[AttributeMatchDetail, ...]:
        return tuple(detail for detail in self.details if detail.quality <= 0)

    def quality_by_kind(self) -> dict[str, float]:
        return {detail.kind: detail.quality for detail in self.details}


def _parse_quantity(value: str) -> tuple[float | None, float | None, float | None] | None:
    """Parse ``kw:15`` / ``v:415`` / ``kw-range:0.37-375`` / ``kw-range:0.37+``."""
    prefix, separator, body = value.partition(":")
    if not separator:
        return None
    try:
        if prefix.endswith("-range"):
            if body.endswith("+"):
                return (None, float(body[:-1]), None)
            low, _, high = body.partition("-")
            return (None, float(low), float(high) if high else None)
        return (float(body), None, None)
    except ValueError:
        return None


def _quantities_overlap(
    required: tuple[float | None, float | None, float | None],
    available: tuple[float | None, float | None, float | None],
) -> bool:
    """True when a required figure equals a stated figure or falls in a stated range."""
    req_value, req_low, req_high = required
    avail_value, avail_low, avail_high = available
    if req_value is not None and avail_value is not None:
        return abs(req_value - avail_value) <= QUANTITY_TOLERANCE * max(
            abs(req_value), 1e-9
        )
    if req_value is not None and avail_low is not None:
        upper = avail_high if avail_high is not None else float("inf")
        return avail_low <= req_value <= upper
    if req_low is not None and avail_low is not None:
        upper = avail_high if avail_high is not None else float("inf")
        return req_low <= upper and avail_low <= (
            req_high if req_high is not None else float("inf")
        )
    return False


def _values_equivalent(kind: str, required: str, available: str) -> bool:
    """True when two canonical values of the same kind mean the same thing."""
    if required == available:
        return True
    if kind in (POWER, VOLTAGE):
        req = _parse_quantity(required)
        avail = _parse_quantity(available)
        if req and avail:
            return _quantities_overlap(req, avail)
    return False


def _outcome_for(
    required: TechnicalAttribute, available: Sequence[TechnicalAttribute]
) -> str:
    """Return ``"exact"``, ``"family"`` or ``"none"`` for one required attribute."""
    if not available:
        return "none"

    stated = [item for item in available if item.value]
    family = [item for item in available if item.value is None]

    if required.value is None:
        # The requirement asks for the family itself ("efficiency classes") and
        # the record covers that family.
        return "family"
    if any(
        item.value and _values_equivalent(required.kind, required.value, item.value)
        for item in stated
    ):
        return "exact"
    if family and required.kind in FAMILY_COVERAGE_KINDS:
        # The record covers this classification but states no value.
        return "family"
    return "none"


#: Quality earned by each per-value outcome.
_OUTCOME_QUALITY = {"exact": 1.0, "family": FAMILY_MATCH_QUALITY, "none": 0.0}


def match_attributes(
    required: Iterable[TechnicalAttribute],
    available: Iterable[TechnicalAttribute],
) -> AttributeMatch:
    """Compare the attributes a requirement asks for with those a record states.

    Per kind the quality is the mean over the requirement's values, so asking for
    "IP55 and IP65" against a record that covers only one of them earns 0.5. The
    detail records which values matched exactly, which were covered only as a
    family, and which the record does not state at all.
    """
    required_by_kind: dict[str, list[TechnicalAttribute]] = {}
    for attribute in required:
        required_by_kind.setdefault(attribute.kind, []).append(attribute)

    available_by_kind: dict[str, list[TechnicalAttribute]] = {}
    for attribute in available:
        available_by_kind.setdefault(attribute.kind, []).append(attribute)

    details: list[AttributeMatchDetail] = []
    for kind in sorted(required_by_kind):
        requirements = required_by_kind[kind]
        candidates = available_by_kind.get(kind, [])
        buckets: dict[str, list[str]] = {"exact": [], "family": [], "none": []}
        qualities: list[float] = []
        for requirement in requirements:
            outcome = _outcome_for(requirement, candidates)
            buckets[outcome].append(requirement.surface)
            qualities.append(_OUTCOME_QUALITY[outcome])
        details.append(
            AttributeMatchDetail(
                kind=kind,
                required=tuple(item.surface for item in requirements),
                quality=sum(qualities) / len(qualities),
                exact_values=tuple(buckets["exact"]),
                family_values=tuple(buckets["family"]),
                missing_values=tuple(buckets["none"]),
            )
        )
    return AttributeMatch(details=tuple(details))


def describe_matched(detail: AttributeMatchDetail) -> str:
    """Explain an attribute match without overstating it."""
    notes: list[str] = []
    if detail.exact_values:
        notes.append("exact value match for " + ", ".join(detail.exact_values))
    if detail.family_values:
        notes.append(
            "the record covers this family of attributes but does not state "
            + ", ".join(detail.family_values)
        )
    if detail.missing_values:
        notes.append("not stated in the record: " + ", ".join(detail.missing_values))
    return f"{detail.label}: {detail.values} (" + "; ".join(notes) + ")"


def describe_unmatched(detail: AttributeMatchDetail) -> str:
    """Explain that a required attribute was not found in the record text."""
    return f"{detail.label}: {detail.values} (not stated in the retrieved record text)"



def required_kinds(attributes: Iterable[TechnicalAttribute]) -> tuple[str, ...]:
    """Kinds present in a requirement, in a deterministic order."""
    return tuple(sorted({attribute.kind for attribute in attributes}))


__all__ = [
    "CATEGORY_KINDS",
    "COOLING",
    "DUTY",
    "EFFICIENCY_CLASS",
    "ENCLOSURE",
    "FAMILY_COVERAGE_KINDS",
    "FAMILY_MATCH_QUALITY",
    "FREQUENCY",
    "INSULATION",
    "KIND_LABELS",
    "MOTOR_TYPE",
    "MOUNTING",
    "PHASE",
    "POWER",
    "VALUE_KINDS",
    "VOLTAGE",
    "AttributeMatch",
    "AttributeMatchDetail",
    "TechnicalAttribute",
    "describe_matched",
    "describe_unmatched",
    "extract_technical_attributes",
    "match_attributes",
    "required_kinds",
]


