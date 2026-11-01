"""Enforce the answer contract on raw LLM output.

1. Strip reasoning blocks; split the "Source:" line from the body.
2. Citation: keep the model's URL only if it is one of the retrieved source_urls,
   else use the top hit's. Remove any other URLs from the body.
3. Not-found: the model's own "couldn't find" reply is normalized.
4. Advice leak: advice phrasing in the body is replaced by the refusal.
5. Grounding: every % or ₹ figure in the body must appear in some retrieved chunk.
6. Length: keep the first `max_sentences` sentences.
Each check that fires adds a flag, shown in the pipeline view.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.generate.llm.base import strip_reasoning
from src.generate.prompts import NOT_FOUND, REFUSAL
from src.models import Hit

WRAPPER_TAG = re.compile(r"</?\s*(answer|response|output)\s*>", re.IGNORECASE)
URL = re.compile(r"https?://[^\s<>()\[\]\"']+")
SOURCE_LINE = re.compile(r"^\s*\**\s*(source|citation)s?\s*\**\s*:.*$", re.IGNORECASE | re.MULTILINE)
UPDATED_LINE = re.compile(r"^\s*\**\s*last updated.*$", re.IGNORECASE | re.MULTILINE)
FIGURE = re.compile(r"₹\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?%|\b(?:rs\.?|inr)\s?\d[\d,]*(?:\.\d+)?", re.IGNORECASE)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9₹\"'(])")
NOT_FOUND_HINTS = ("couldn't find", "could not find", "not in my sources", "not available in the context",
                   "context does not", "not mentioned in the context", "no information")
ADVICE_PHRASES = ("you should", "i recommend", "we recommend", "i suggest", "good choice", "better option",
                  "worth investing", "you can consider", "consider investing", "should invest", "is a good fund")


@dataclass
class Checked:
    status: str  # answered | not_found | refused
    body: str
    source_url: str
    flags: list[str] = field(default_factory=list)


def postcheck(raw: str, hits: list[Hit], *, max_sentences: int, refusal_url: str) -> Checked:
    flags: list[str] = []
    allowed = [hit.source_url for hit in hits]
    top_url = allowed[0] if allowed else refusal_url

    text = WRAPPER_TAG.sub("", strip_reasoning(raw))
    source_lines = [match.group(0) for match in SOURCE_LINE.finditer(text)]
    cited = [url.rstrip(".,;") for line in source_lines for url in URL.findall(line)]
    cited += [url.rstrip(".,;") for url in URL.findall(SOURCE_LINE.sub("", text))]

    source_url = next((url for url in cited if url in allowed), None)
    if source_url is None:
        source_url = top_url
        flags.append("citation_missing" if not cited else "citation_replaced")
    if len(set(cited)) > 1:
        flags.append("extra_urls_removed")

    body = UPDATED_LINE.sub("", SOURCE_LINE.sub("", text))
    body = URL.sub("", body)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\s*\n\s*", " ", body).strip()
    body = re.sub(r"\(\s*\)|\[\s*\]", "", body).strip()

    lowered = body.lower()
    if not body or any(hint in lowered for hint in NOT_FOUND_HINTS):
        return Checked("not_found", NOT_FOUND, top_url, flags + ["model_not_found"])
    if any(phrase in lowered for phrase in ADVICE_PHRASES):
        return Checked("refused", REFUSAL, refusal_url, flags + ["advice_blocked"])

    ungrounded = ungrounded_figures(body, hits)
    if ungrounded:
        return Checked("not_found", NOT_FOUND, top_url, flags + [f"ungrounded_figures:{','.join(ungrounded)}"])

    sentences = SENTENCE_END.split(body)
    if len(sentences) > max_sentences:
        body = " ".join(sentences[:max_sentences])
        flags.append(f"trimmed_to_{max_sentences}_sentences")
    return Checked("answered", body, source_url, flags)


def ungrounded_figures(body: str, hits: list[Hit]) -> list[str]:
    """% / ₹ figures in body that appear in none of the retrieved chunks."""
    context = _squash(" ".join(hit.text for hit in hits))
    missing = []
    for figure in FIGURE.findall(body):
        number = re.sub(r"[^\d.]", "", figure).strip(".")
        if not number:
            continue
        variants = {number, number.rstrip("0").rstrip(".") if "." in number else number}
        if not any(re.search(rf"(?<![\d.]){re.escape(v)}(?![\d])", context) for v in variants):
            missing.append(figure.strip())
    return missing


def _squash(text: str) -> str:
    """Drop thousands separators so ₹1,000 in a chunk matches 1000 in an answer."""
    return re.sub(r"(?<=\d),(?=\d)", "", text)
