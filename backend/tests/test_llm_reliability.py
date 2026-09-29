"""Reliability tests for the LLM boundary: parsing, provider errors, prompt budget.

Three real production problems motivated this module:

* the verdict JSON extractor sliced from the first ``{`` to the last ``}``, so a
  response cut off by the token limit (or carrying prose/extra braces) failed
  with "Expecting ',' delimiter";
* every provider error path constructed ``AppError(message, status_code=...)``
  although ``AppError.__init__`` accepts only ``message``/``details``, so the
  fallback itself died with ``TypeError`` instead of falling back;
* the candidate payload repeated the whole indexed document inside every
  evidence snippet, which pushed the prompt past the provider's tokens-per-minute
  budget and truncated the verdict.

Nothing here touches the network: the provider is driven through its own
``client=`` seam with scripted responses.
"""

from __future__ import annotations

import json

import httpx
import pytest

from app.ai.llm.base import LLMProvider, LLMRequest, LLMResponse
from app.ai.llm.providers.openai_compatible import OpenAICompatibleLLMProvider
from app.ai.reasoning import (
    _SCOPE_CHARS,
    _SIGNAL_CHARS,
    _SNIPPET_CHARS,
    CandidateEvidence,
    CandidateStandard,
    extract_json_object,
    extract_json_object_with_status,
    parse_reasoning_response,
    run_ai_reasoning,
)
from app.core.config import Settings
from app.core.exceptions import AppError, LLMProviderError, LLMRateLimitError

_FENCE = chr(96) * 3
_REAL = "IS 12615:2018"
_PUMP = "IS 7538:1996"
_INVENTED = "IS 99999:2030"


def _item(number: str, **overrides) -> dict:
    payload = {
        "standard_number": number,
        "applicability_type": "direct",
        "confidence": 0.8,
        "why_it_matches": "Covers the requested scope.",
        "matched_requirements": ["efficiency class"],
        "uncovered_requirements": [],
    }
    payload.update(overrides)
    return payload


# --- JSON extraction --------------------------------------------------------
def test_extract_reads_plain_json() -> None:
    assert extract_json_object(json.dumps({"recommendations": []})) == {
        "recommendations": []
    }


@pytest.mark.parametrize(
    "wrapper",
    [
        "{body}",
        "  \n{body}\n  ",
        _FENCE + "json\n{body}\n" + _FENCE,
        _FENCE + "\n{body}\n" + _FENCE,
        "Here is the result:\n{body}\nHope that helps.",
    ],
    ids=["bare", "whitespace", "fenced-json", "fenced-plain", "prose-around"],
)
def test_extract_tolerates_normal_model_formatting(wrapper: str) -> None:
    body = json.dumps({"recommendations": [_item(_REAL)]})
    parsed = extract_json_object(wrapper.replace("{body}", body))
    assert [r["standard_number"] for r in parsed["recommendations"]] == [_REAL]


def test_extract_is_not_fooled_by_braces_inside_strings() -> None:
    parsed = extract_json_object(
        json.dumps({"why_it_matches": "states a } brace and a { one", "n": 1})
    )
    assert parsed["why_it_matches"] == "states a } brace and a { one"


def test_extract_ignores_braces_after_the_object() -> None:
    parsed = extract_json_object(
        '{"recommendations": []}\nNote: see {docs} for details.'
    )
    assert parsed == {"recommendations": []}


def test_extract_repairs_a_response_truncated_by_the_token_limit() -> None:
    """The real Groq failure: the verdict was cut off mid-key."""
    truncated = (
        "{\n"
        ' "recommendations": [\n'
        f'  {{"standard_number": "{_REAL}", "applicability_type": "direct", '
        '"confidence": 0.9},\n'
        f'  {{"standard_number": "{_PUMP}", "applicability_ty'
    )
    parsed = extract_json_object(truncated)
    # Nothing is guessed: the second entry keeps only the field the model had
    # finished writing, and the half-written key is dropped.
    assert [r["standard_number"] for r in parsed["recommendations"]] == [_REAL, _PUMP]
    assert "applicability_type" not in parsed["recommendations"][1]
    assert parsed["recommendations"][0]["confidence"] == 0.9


def test_extract_repairs_truncation_in_the_middle_of_a_string() -> None:
    truncated = (
        '{"recommendations": ['
        f'{{"standard_number": "{_REAL}", "why_it_matches": "covers the eff'
    )
    parsed = extract_json_object(truncated)
    assert [r["standard_number"] for r in parsed["recommendations"]] == [_REAL]


def test_extract_repairs_truncation_right_after_a_separator() -> None:
    parsed = extract_json_object('{"recommendations": [{"a": 1},')
    assert parsed == {"recommendations": [{"a": 1}]}


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "I cannot help with that.",
        "[1, 2, 3]",
    ],
    ids=["empty", "blank", "prose-only", "json-array"],
)
def test_extract_rejects_output_with_no_json_object(text: str) -> None:
    with pytest.raises(ValueError):
        extract_json_object(text)


def test_truncated_output_with_nothing_complete_yields_an_empty_object() -> None:
    # Repaired, but with no usable entries: the caller must treat it as no verdict
    # rather than inventing one.
    assert extract_json_object('{"recommendations": [') == {}


def test_the_repair_status_separates_a_complete_verdict_from_a_cut_off_one() -> None:
    """A repaired envelope means the model's output is *incomplete*.

    The difference is invisible in the parsed object, so it is reported
    explicitly: the caller needs it to know that candidates missing from the
    verdict were never judged rather than excluded.
    """
    complete = json.dumps({"recommendations": [_item(_REAL)]})
    assert extract_json_object_with_status(complete) == (
        {"recommendations": [_item(_REAL)]},
        False,
    )

    cut_off = (
        "{\n"
        '  "recommendations": [\n'
        '    {"standard_number": "' + _REAL + '", "applicability_type": "direct", "confidence": 0.9},\n'
        '    {"standard_number": "' + _PUMP + '", "applic'
    )
    parsed, repaired = extract_json_object_with_status(cut_off)
    assert repaired is True
    # Same recoverability as before (the half-written *key* is dropped, the
    # elements it had already finished are kept) - the new signal is the flag.
    assert [r["standard_number"] for r in parsed["recommendations"]] == [
        _REAL,
        _PUMP,
    ]
    assert "applicability_type" not in parsed["recommendations"][1]


def test_a_repaired_verdict_is_flagged_as_truncated_even_when_finish_reason_is_not(
    ) -> None:
    """A dropped stream can end the body without the provider saying ``length``.

    The verdict is a fragment either way, so it must be flagged and discarded by
    the caller rather than used to exclude candidates the model never reached.
    """
    outcome = parse_reasoning_response(
        '{"recommendations": [{"standard_number": "' + _REAL + '", "confidence": 0.9, "why_it_m',
        [
            CandidateStandard(
                standard_number=_REAL,
                title="title",
                evidence=(CandidateEvidence(chunk_id="c0", snippet="s"),),
            ),
            CandidateStandard(
                standard_number=_PUMP,
                title="title",
                evidence=(CandidateEvidence(chunk_id="c1", snippet="s"),),
            ),
        ],
    )

    assert outcome.truncated is True
    # The parseable part is still validated; it is the *caller* that refuses a
    # fragment. Nothing here invents a designation or a verdict.
    assert [r.standard_number for r in outcome.recommendations] == [_REAL]
    assert all(r.standard_number in {_REAL, _PUMP} for r in outcome.recommendations)


# --- validation is not weakened by the repair ------------------------------
def test_repaired_output_is_still_validated_against_the_candidates() -> None:
    from app.ai.reasoning import CandidateEvidence, CandidateStandard

    candidate = CandidateStandard(
        standard_number=_REAL,
        title="title",
        evidence=(CandidateEvidence(chunk_id="c0", snippet="s"),),
    )
    outcome = parse_reasoning_response(
        '{"recommendations": [{"standard_number": "' + _INVENTED + '", "applic',
        [candidate],
    )
    # Repaired into a parseable object, but the invented designation is still
    # rejected: robustness of the envelope must not weaken grounding.
    assert outcome.recommendations == []
    assert outcome.dropped == [_INVENTED]


# --- provider error handling ------------------------------------------------
class _FakeResponse:
    """Minimal stand-in for ``httpx.Response``."""

    def __init__(self, status_code: int, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or ("no body" if payload is None else json.dumps(payload))

    def json(self):
        if self._payload is None:
            raise ValueError("response body is not JSON")
        return self._payload


class _FakeClient:
    """Replays scripted responses (or raises them) and records every call."""

    def __init__(self, responses) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json})
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        response = self._responses[index]
        if isinstance(response, Exception):
            raise response
        return response


def _ok_payload(text: str = '{"recommendations": []}') -> dict:
    return {
        "model": "test-model",
        "choices": [
            {"message": {"content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _provider(responses):
    settings = Settings(
        llm_provider="openai_compatible",
        llm_model="test-model",
        llm_api_key="test-key",
        llm_base_url="https://example.invalid/v1",
    )
    client = _FakeClient(responses)
    return OpenAICompatibleLLMProvider(settings, client=client), client


def _request() -> LLMRequest:
    return LLMRequest(prompt="p", system_prompt="s", temperature=0.0, max_tokens=100)


def test_rate_limit_fails_fast_without_retrying() -> None:
    provider, client = _provider([_FakeResponse(429, text="Rate limit reached (TPM)")])

    with pytest.raises(LLMRateLimitError) as excinfo:
        provider.generate(_request())

    assert "Rate limit" in str(excinfo.value)
    # A TPM budget resets on a wall-clock window, so retrying inside it is pure
    # waste: exactly one request must be made.
    assert len(client.calls) == 1


def test_rate_limit_error_is_catchable_as_an_app_error() -> None:
    # The service catches AppError to fall back, so this must stay a subclass.
    assert issubclass(LLMRateLimitError, LLMProviderError)
    assert issubclass(LLMRateLimitError, AppError)


def test_server_errors_are_retried_then_reported() -> None:
    provider, client = _provider([_FakeResponse(500, text="upstream boom")])

    with pytest.raises(LLMProviderError):
        provider.generate(_request())

    assert len(client.calls) == 3


def test_transport_errors_are_retried_then_reported() -> None:
    provider, client = _provider(
        [httpx.RequestError("connection reset", request=httpx.Request("GET", "u"))]
    )

    with pytest.raises(LLMProviderError):
        provider.generate(_request())

    assert len(client.calls) == 3


def test_a_transient_server_error_recovers_on_retry() -> None:
    provider, client = _provider(
        [_FakeResponse(503, text="warming up"), _FakeResponse(200, _ok_payload())]
    )

    response = provider.generate(_request())

    assert response.text == '{"recommendations": []}'
    assert response.usage["total_tokens"] == 15
    assert len(client.calls) == 2


def test_client_errors_are_not_retried() -> None:
    provider, client = _provider([_FakeResponse(400, text="bad model")])

    with pytest.raises(LLMProviderError):
        provider.generate(_request())

    assert len(client.calls) == 1


def test_empty_choices_is_reported_as_a_provider_error() -> None:
    provider, _ = _provider([_FakeResponse(200, {"choices": []})])

    with pytest.raises(LLMProviderError):
        provider.generate(_request())


def test_non_json_body_is_reported_as_a_provider_error() -> None:
    provider, _ = _provider([_FakeResponse(200, None, text="<html>oops</html>")])

    with pytest.raises(LLMProviderError):
        provider.generate(_request())


@pytest.mark.parametrize(
    "responses",
    [
        [_FakeResponse(429, text="TPM")],
        [_FakeResponse(500, text="boom")],
        [_FakeResponse(400, text="bad request")],
        [_FakeResponse(200, {"choices": []})],
        [_FakeResponse(200, None, text="<html/>")],
    ],
    ids=["429", "500", "400", "empty-choices", "non-json"],
)
def test_no_error_path_raises_a_type_error(responses) -> None:
    """Regression: AppError(status_code=...) is not a valid constructor call."""
    provider, _ = _provider(responses)
    with pytest.raises(AppError):
        provider.generate(_request())


# --- prompt budget: small payload, unchanged grounding -------------------
# A scope long enough to exercise the clip, and a realistic indexed chunk.
_LONG_SCOPE = (
    "This standard specifies the efficiency classes and performance requirements "
    "for line-operated three-phase a.c. motors. It applies to motors of the output "
    "range stated in the scope clause, and prescribes the test methods used to "
    "determine the losses and the efficiency levels for each rated output and "
    "speed, together with the tolerance limits that a manufacturer must observe "
    "when declaring an efficiency class for a given motor rating and pole count, "
    "including the conditions under which the declared efficiency class applies."
)
_LONG_SNIPPET = (
    "Standard Number: IS 12615:2018\n"
    "Title: Line-Operated Three-Phase a.c. Motors - Efficiency Classes\n"
    + ("This indexed catalogue text repeats the scope and the certification "
       "statement verbatim for the record. " * 20)
)


def _rich_candidate(**overrides) -> CandidateStandard:
    base = {
        "standard_number": "IS 12615:2018",
        "title": "Efficiency classes",
        "scope": _LONG_SCOPE,
        "category": "rotating electrical machines",
        "status": "current",
        "year": 2018,
        "certification": ("BIS certification scheme ISI Mark",),
        "related_standards": ("IS/IEC 60034-1:2022 Rating and performance",),
        "deterministic_rank": 1,
        "similarity_score": 0.61,
        "attribute_score": 0.42,
        "matched_attributes": ("Efficiency class: IE3 (matches)",),
        "unmatched_attributes": (
            "IP enclosure rating: IP55 (not stated in the retrieved record text)",
            "Frequency: 50 Hz (not stated in the retrieved record text)",
        ),
        "matched_terms": ("motor", "induction", "efficiency"),
        "evidence": (
            CandidateEvidence(chunk_id="standard::IS-12615-2018::0", snippet=_LONG_SNIPPET),
        ),
    }
    base.update(overrides)
    return CandidateStandard(**base)


class _RecordingProvider(LLMProvider):
    """Records the prompt the reasoning layer would send, and returns no verdict."""

    provider_name = "prompt-recorder"

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    @property
    def is_configured(self) -> bool:
        return True

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            text=json.dumps({"recommendations": []}),
            provider=self.provider_name,
            model="recorder",
        )


def test_payload_keeps_the_evidence_an_applicability_decision_needs() -> None:
    payload = _rich_candidate().to_payload()

    # Identity and applicability material survive untouched.
    assert payload["standard_number"] == "IS 12615:2018"
    assert payload["title"] == "Efficiency classes"
    assert payload["scope"] == _LONG_SCOPE
    assert payload["certification"] == ["BIS certification scheme ISI Mark"]
    assert payload["related_standards"] == [
        "IS/IEC 60034-1:2022 Rating and performance"
    ]
    assert payload["status"] == "current"
    assert payload["evidence"][0]["chunk_id"] == "standard::IS-12615-2018::0"


def test_payload_clips_only_the_duplicated_snippet() -> None:
    payload = _rich_candidate().to_payload()
    snippet = payload["evidence"][0]["snippet"]

    # The chunk text is a verbatim anchor now, but the scope is NOT clipped.
    assert len(snippet) < len(_LONG_SNIPPET)
    assert snippet.startswith("Standard Number: IS 12615:2018")
    assert payload["scope"] == _LONG_SCOPE


def test_payload_clips_a_pathologically_long_scope() -> None:
    payload = _rich_candidate(scope="x" * 5000).to_payload()
    assert len(payload["scope"]) == _SCOPE_CHARS


def test_deterministic_signals_stay_compact() -> None:
    signals = _rich_candidate().to_payload()["deterministic_signals"]

    assert signals["det_rank"] == 1
    assert signals["cos"] == 0.61
    assert signals["attr"] == 0.42
    for key in ("matched_attributes", "unmatched_attributes", "matched_terms"):
        assert len(signals[key]) <= 3
        assert all(len(item) <= _SIGNAL_CHARS for item in signals[key])


def test_a_many_candidate_prompt_fits_a_small_token_budget() -> None:
    """Ten candidates must stay well inside the provider's per-minute budget."""
    candidates = [_rich_candidate() for _ in range(10)]
    rendered = json.dumps([c.to_payload() for c in candidates], ensure_ascii=False)

    # Per candidate the payload is bounded by the three clip limits, so a
    # ten-candidate prompt cannot blow up however long the source text is.
    assert len(rendered) <= 10 * (
        _SCOPE_CHARS + _SNIPPET_CHARS + 3 * _SIGNAL_CHARS + 400
    )


def test_clipping_removes_the_duplicated_index_text() -> None:
    """The payload must be far smaller than the catalogue text it is built from.

    This is the regression guard for the tokens-per-minute failure: the indexed
    document used to be re-sent inside every evidence snippet.
    """
    candidates = [_rich_candidate() for _ in range(10)]
    rendered = json.dumps([c.to_payload() for c in candidates], ensure_ascii=False)

    # What the prompt cost before clipping: every field sent at full length.
    raw = 0
    for candidate in candidates:
        raw += len(candidate.scope or "") + len(candidate.title or "")
        raw += sum(len(e.snippet) for e in candidate.evidence)
        raw += sum(len(s) for s in candidate.matched_attributes)
        raw += sum(len(s) for s in candidate.unmatched_attributes)
        raw += sum(len(s) for s in candidate.matched_terms)

    assert len(rendered) < raw * 0.6
    # And the duplicated boilerplate really is gone, not merely shortened.
    assert "repeats the scope and the certification statement" not in rendered


def test_grounding_survives_the_smaller_payload() -> None:
    """A verdict parsed from a compact prompt is validated exactly as before."""
    candidate = _rich_candidate()
    good = json.dumps(
        {
            "recommendations": [
                {
                    "standard_number": "is-12615-2018",
                    "applicability_type": "direct",
                    "confidence": 0.9,
                    "why_it_matches": "Defines the requested IE3 class.",
                }
            ]
        }
    )
    accepted = parse_reasoning_response(good, [candidate])
    assert [r.standard_number for r in accepted.recommendations] == ["IS 12615:2018"]

    invented = good.replace("is-12615-2018", "IS 99999:2030")
    rejected = parse_reasoning_response(invented, [candidate])
    assert rejected.recommendations == []
    assert rejected.dropped == ["IS 99999:2030"]


def test_run_ai_reasoning_sends_the_compact_payload() -> None:
    provider = _RecordingProvider()
    run_ai_reasoning(
        provider=provider,
        requirement="Supply of an IE3 induction motor",
        attributes=[{"kind": "efficiency_class", "value": "IE3"}],
        candidates=[_rich_candidate()],
    )
    prompt = provider.requests[0].prompt

    # Identity, scope and a verbatim evidence anchor are all still on the wire.
    assert "IS 12615:2018" in prompt
    assert _LONG_SCOPE[:120] in prompt
    assert "Standard Number: IS 12615:2018" in prompt
    assert "deterministic_signals" in prompt
    # ...while the duplicated tail of the indexed document is gone.
    assert "repeats the scope and the certification statement" not in prompt
