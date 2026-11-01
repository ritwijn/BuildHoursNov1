"""Provider-neutral LLM interface: application code only calls generate()."""

from __future__ import annotations

import re
from typing import Protocol

THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL | re.IGNORECASE)


class LLMError(Exception):
    """Timeout, auth, rate limit, bad model id, or missing key."""


class LLMClient(Protocol):
    name: str

    def generate(self, messages: list[dict], *, temperature: float, max_tokens: int) -> str: ...


def strip_reasoning(text: str) -> str:
    """Remove <think>…</think> blocks that reasoning models (e.g. Qwen 3) may emit."""
    return THINK_BLOCK.sub("", text).strip()
