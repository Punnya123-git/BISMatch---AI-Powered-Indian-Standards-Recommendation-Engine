"""AI layer: provider abstractions and the reasoning stage that sits above them.

Nothing in this package talks to HTTP, FastAPI or the vector store. It only
defines *interfaces* plus a factory, so the concrete provider can be chosen via
environment variables once the team decides which one to use.

:mod:`app.ai.reasoning` turns verified retrieval candidates into the final
applicability classification and ranking using the configured provider.
"""

from app.ai.llm.base import LLMProvider, LLMRequest, LLMResponse
from app.ai.llm.factory import get_llm_provider
from app.ai.reasoning import (
    CandidateStandard,
    ReasonedStandard,
    ReasoningOutcome,
    parse_reasoning_response,
    run_ai_reasoning,
)

__all__ = [
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "get_llm_provider",
    "CandidateStandard",
    "ReasonedStandard",
    "ReasoningOutcome",
    "parse_reasoning_response",
    "run_ai_reasoning",
]
