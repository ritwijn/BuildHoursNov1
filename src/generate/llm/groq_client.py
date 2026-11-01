from __future__ import annotations

from src.generate.llm.base import LLMError, strip_reasoning


class GroqClient:
    name = "groq"

    def __init__(self, api_key: str, model: str, timeout_s: float):
        from groq import Groq

        self.model = model
        self._client = Groq(api_key=api_key, timeout=timeout_s, max_retries=1)

    def generate(self, messages: list[dict], *, temperature: float, max_tokens: int) -> str:
        import groq

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except groq.APIError as exc:
            raise LLMError(f"Groq request failed: {exc.__class__.__name__}: {exc}", reason=_reason(exc)) from exc
        content = response.choices[0].message.content or ""
        return strip_reasoning(content)


def _reason(exc: Exception) -> str:
    import groq

    for kind, reason in (
        (groq.AuthenticationError, "auth_failed"),
        (groq.PermissionDeniedError, "auth_failed"),
        (groq.NotFoundError, "model_not_found"),
        (groq.RateLimitError, "rate_limited"),
        (groq.APITimeoutError, "timeout"),
        (groq.APIConnectionError, "unreachable"),
        (groq.BadRequestError, "bad_request"),
    ):
        if isinstance(exc, kind):
            return reason
    return "failed"
