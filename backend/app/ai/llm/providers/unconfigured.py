"""Placeholder provider used until a real LLM backend is chosen.

It never fabricates output: any attempt to generate text raises an explicit
configuration error, which the API layer reports to the client.
"""

from __future__ import annotations

from app.ai.llm.base import LLMProvider, LLMRequest, LLMResponse
from app.core.exceptions import ProviderNotConfiguredError


class UnconfiguredLLMProvider(LLMProvider):
    """Reports that no LLM provider has been selected yet."""

    provider_name = "unconfigured"

    def __init__(self, reason: str = "No LLM provider has been configured.") -> None:
        self._reason = reason

    @property
    def is_configured(self) -> bool:
        return False

    def generate(self, request: LLMRequest) -> LLMResponse:
        raise ProviderNotConfiguredError(
            f"{self._reason} Set LLM_PROVIDER, LLM_MODEL and LLM_API_KEY in the "
            "backend environment before requesting generated content."
        )
