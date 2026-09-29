"""LLM provider contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(slots=True)
class LLMRequest:
    """Provider-agnostic LLM call description."""

    prompt: str
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class LLMResponse:
    """Provider-agnostic LLM result."""

    text: str
    provider: str
    model: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class LLMProvider(ABC):
    """Interface every LLM backend must implement."""

    #: Registry key used in settings, e.g. ``"openai_compatible"``.
    provider_name: str = "unconfigured"

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """True when the provider has everything it needs to run."""

    @property
    def model(self) -> str | None:
        return None

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Run a single completion. Raises ``ProviderNotConfiguredError``."""
        raise NotImplementedError

    def describe(self) -> str:
        """Short, human readable description used by /api/health."""
        state = "configured" if self.is_configured else "not configured"
        return f"{self.provider_name} ({state})"
