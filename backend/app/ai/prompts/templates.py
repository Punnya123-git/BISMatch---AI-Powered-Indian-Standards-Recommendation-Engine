"""Prompt templates used by the future LLM steps.

Kept in one place so prompts are versioned and reviewable. They are deliberately
not wired into a fake pipeline: they are consumed only once a real LLM provider
and the standards dataset are available.
"""

from __future__ import annotations

REQUIREMENT_EXTRACTION_SYSTEM_PROMPT = (
    "You are an assistant for Indian public procurement. You extract structured "
    "requirements from tender documents. Never invent information that is not "
    "present in the supplied text."
)

REQUIREMENT_EXTRACTION_PROMPT = """\
Extract the technical requirements from the procurement text below.

Return a JSON list of objects with the keys:
  - "requirement": the requirement stated in the document, in plain language
  - "category": a short category label
  - "source_snippet": the verbatim text the requirement came from

Text:
---
{text}
---
"""

#: Operational detail lives in :data:`STANDARD_RECOMMENDATION_PROMPT`. This
#: carries only the principles that must hold regardless of how the prompt is
#: worded, kept short because it is re-sent on every request.
STANDARD_RECOMMENDATION_SYSTEM_PROMPT = (
    "You are an expert technical committee assistant on Indian Standards (BIS) applied to public procurement specifications.\n"
    "You decide whether a candidate standard actually applies to what is being procured, and rank those that do.\n\n"
    "PRINCIPLES:\n"
    "1. Judge every candidate against the procurement target - the specific product, material, equipment, service or specification actually being bought - not against the wider facility or industry it belongs to.\n"
    "2. Do NOT infer that a component is being procured merely because that component could be used within the facility, manufacturing process, plant or system the user described. A component is in scope ONLY if the user explicitly stated it as part of the procurement.\n"
    "3. A standard is 'direct' ONLY when its verified scope covers the procurement target. It is 'related' ONLY when the user explicitly included the related component/specification, or the verified evidence directly supports the relationship.\n"
    "4. Exclude any candidate that is not genuinely relevant. Returning fewer standards, including none, is correct. Never widen the net to produce output.\n"
    "5. NEVER invent standard numbers, titles, scopes, amendments, relationships or certification requirements. Use only the supplied requirement and verified standard evidence - no outside-world assumptions.\n"
    "6. 'confidence' is your own judgement of applicability from the verified evidence. It must NOT track retrieval similarity.\n"
    "7. Return only a valid JSON object matching the requested schema, with no markdown or commentary."
)

STANDARD_RECOMMENDATION_PROMPT = """\
Procurement requirement:
\"\"\"
{requirement}
\"\"\"

Extracted technical attributes from requirement:
{attributes}

Supplied verified candidate standards (retrieved from ChromaDB and verified
against the catalogue; the deterministic similarity/attribute signals are hints,
not decisions):
\"\"\"
{context}
\"\"\"

The ONLY standard designations you may output are exactly these (verbatim):
{allowed_standards}

TASK 1 - ESTABLISH THE PROCUREMENT TARGET (do this first):
State plainly what is actually being procured: the specific product, material,
equipment, service or specification. The target is what the user is buying, NOT
the broader facility, plant, process or industry the purchase sits in. Note the
specifications the user explicitly gave.

TASK 2 - THE APPLICABILITY RULE (applies to every domain):
A standard is applicable only when its verified scope/evidence covers the actual
product, material, equipment, service or specification being procured.

Distinguish what KIND of target it is - this decides most cases:
- FACILITY, PLANT, PROCESS, SITE or INDUSTRY: the target is the facility itself,
  not the equipment installed in it. Equipment merely usable inside it is NOT
  part of the target unless the user stated it, however plausible it is.
- ARTICLE, PRODUCT, EQUIPMENT or MATERIAL: the target is that item, so a
  standard governing the item, or a component that is integral to and supplied
  with it, does concern the target.

"Related" applies only when the user explicitly includes the related
component/specification, or the verified evidence directly supports the
relationship. "Could this component exist inside such a facility?" is NEVER
sufficient evidence of applicability.

TASK 3 - CLASSIFY EACH CANDIDATE:
- "direct": verified scope directly governs the procurement target itself.
- "related": the user explicitly named the related component/specification, or
  verified evidence directly supports the relationship to the target.
- "needs_verification": possibly relevant, but the evidence/scope is too limited
  or ambiguous to confirm applicability without the complete standard text.
- EXCLUDE it entirely when its scope concerns neither the procurement target nor
  a component integral to it - e.g. equipment that merely happens to be usable
  inside a facility the user described.

HARD RULE - DO NOT INFER UNSTATED COMPONENTS:
Do not infer that a component is being procured merely because that component
could be used within the facility, manufacturing process, plant, or system
described by the user.

Example of the rule (illustrative only - the same reasoning applies in every
domain): for a packaged drinking water manufacturing and bottling plant where the
user has NOT stated that pumps or motors are being procured, pump and motor
standards are EXCLUDED, however plausible such equipment is in that industry. If
the user instead writes "packaged drinking water plant including procurement of
submersible pump sets", the pump set is explicitly stated, so pump and motor
standards whose verified scope supports it may be considered.

CONFIDENCE: your own confidence that the standard genuinely applies, judged from
verified scope and evidence - it must NOT mirror retrieval similarity. If
evidence is weak or ambiguous, lower it, use "needs_verification", or exclude.
Never convert high embedding similarity into "applicable".

EXCLUDING IS CORRECT: a small list, or an empty "recommendations" list, is a
valid outcome. Never keep a candidate just to produce output.

Return a JSON object with this exact structure:
{{
  "procurement_target": {{
    "target": "<what is actually being procured>",
    "explicit_specifications": ["<specification the user explicitly stated>"],
    "note": "<any clarification, or empty string>"
  }},
  "recommendations": [
    {{
      "standard_number": "<Exact designation from supplied candidate list>",
      "applicability_type": "direct" | "related" | "needs_verification",
      "confidence": <float 0.0-1.0: genuine applicability, not similarity>,
      "why_it_matches": "<Technical rationale. If the scope does not concern the procurement target, exclude it instead. If evidence is insufficient, state: 'Insufficient evidence to determine applicability; verification is required.'>",
      "matched_requirements": [
        "<Requirement aspect or attribute covered by this standard>"
      ],
      "uncovered_requirements": [
        "<Requirement aspect or attribute stated in requirement but NOT covered by this standard>"
      ],
      "evidence_chunk_ids": [
        "<Copy chunk_id values from the candidate payloads above that support this judgement>"
      ]
    }}
  ]
}}
"""
