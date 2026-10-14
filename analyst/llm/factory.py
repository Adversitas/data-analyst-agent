from __future__ import annotations

from analyst import settings
from analyst.llm.base import LLMBackend


def make_backend(name: str | None = None, model: str | None = None) -> LLMBackend:
    name = name or settings.BACKEND
    if name == "anthropic":
        from analyst.llm.anthropic_backend import AnthropicBackend

        return AnthropicBackend(model)
    if name == "openai":
        from analyst.llm.openai_backend import OpenAICompatBackend

        return OpenAICompatBackend(model)
    raise ValueError(f"Unknown backend {name!r} (expected 'anthropic' or 'openai')")
