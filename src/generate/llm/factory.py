"""Pick the LLM client from LLM_PROVIDER. Only Groq is implemented; others are config slots."""

from __future__ import annotations

from src.config import Settings, get_settings
from src.generate.llm.base import LLMClient, LLMError


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    provider = settings.llm.provider.lower()
    if provider == "groq":
        if not settings.groq_api_key:
            raise LLMError("GROQ_API_KEY is not set; add it to .env", reason="not_configured")
        from src.generate.llm.groq_client import GroqClient

        return GroqClient(settings.groq_api_key, settings.llm.model, settings.llm.timeout_s)
    if provider in {"gemini", "openai", "anthropic", "mistral"}:
        raise LLMError(f"LLM_PROVIDER={provider} is reserved but not implemented yet; use groq", reason="unsupported_provider")
    raise LLMError(f"Unknown LLM_PROVIDER: {provider}", reason="unsupported_provider")
