"""LLM provider implementations.

Add one module per provider (for example ``openai_compatible.py``,
``azure_openai.py``, ``ollama.py``) and register it in
:mod:`app.ai.llm.factory`. Every provider must implement
:class:`app.ai.llm.base.LLMProvider`.
"""

from app.ai.llm.providers.openai_compatible import (
    OpenAICompatibleLLMProvider,
    PROVIDER_NAME as OPENAI_COMPATIBLE_PROVIDER,
)
from app.ai.llm.providers.unconfigured import UnconfiguredLLMProvider

__all__ = ["OpenAICompatibleLLMProvider", "UnconfiguredLLMProvider"]

