"""Provider factory: maps ``LLM_PROVIDER`` to an implementation instance."""

from __future__ import annotations

from functools import lru_cache

from app.ai.llm.base import LLMProvider
from app.ai.llm.providers.openai_compatible import (
    OpenAICompatibleLLMProvider,
    PROVIDER_NAME as OPENAI_COMPATIBLE_PROVIDER,
)
from app.ai.llm.providers.unconfigured import UnconfiguredLLMProvider
from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Registry of available providers.
_PROVIDERS: dict[str, type[LLMProvider]] = {
    OPENAI_COMPATIBLE_PROVIDER: OpenAICompatibleLLMProvider,
    # Selectable in its own right, so "unconfigured" is reported as an available
    # option rather than only appearing as the implicit fallback for an unknown
    # name. The reasoning layer checks ``is_configured``, never the registry key.
    UnconfiguredLLMProvider.provider_name: UnconfiguredLLMProvider,
}


def register_llm_provider(name: str, provider_cls: type[LLMProvider]) -> None:
    """Register a provider class so it can be selected by name."""
    _PROVIDERS[name.lower()] = provider_cls


def available_llm_providers() -> list[str]:
    """Names of the providers compiled into this build."""
    return sorted(_PROVIDERS)


def _build_provider(settings: Settings) -> LLMProvider:
    name = (settings.llm_provider or "unconfigured").lower()
    if name == UnconfiguredLLMProvider.provider_name:
        # Constructed with no reason so ``describe()`` stays the neutral default.
        return UnconfiguredLLMProvider()
    provider_cls = _PROVIDERS.get(name)
    if provider_cls is None:
        return UnconfiguredLLMProvider(
            f"LLM provider '{name}' is not implemented in this build."
        )
    return provider_cls(settings)


@lru_cache
def get_llm_provider() -> LLMProvider:
    """Return the configured provider (cached per process)."""
    settings = get_settings()
    provider = _build_provider(settings)
    logger.info("LLM provider: %s", provider.describe())
    return provider


def reset_llm_provider_cache() -> None:
    """Clear the cache (used by tests and after configuration changes)."""
    get_llm_provider.cache_clear()
