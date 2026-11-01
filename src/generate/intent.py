"""Rules-first intent router (keyword lists live in config/intents.yaml).

Order, first match wins:
  out_of_scope  another AMC is named, or a fund type with no registered scheme
  advice        buy/sell/switch/recommend/"better for me"…
  returns       returns/CAGR/performance…
  factual       expense ratio, exit load, SIP, lock-in, riskometer…
  other         anything else; still goes through RAG and the min_score gate
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.config import Source, load_intents, load_sources


@dataclass
class IntentResult:
    intent: str
    scheme_id: str | None = None
    matched: list[str] = field(default_factory=list)


def classify(text: str, sources: list[Source] | None = None, intents: dict[str, list[str]] | None = None) -> IntentResult:
    sources = sources if sources is not None else load_sources()
    intents = intents if intents is not None else load_intents()
    lowered = _normalize(text)
    scheme_id = match_scheme(lowered, sources)

    other_amcs = _matches(lowered, intents.get("out_of_scope_amcs", []))
    other_amcs += _brand_funds(lowered, intents.get("amc_brands_needing_fund_name", []), intents.get("scheme_terms", []))
    if other_amcs:
        return IntentResult("out_of_scope", None, other_amcs)
    unknown_scheme = _matches(lowered, intents.get("scheme_terms", []))
    if unknown_scheme and scheme_id is None:
        return IntentResult("out_of_scope", None, unknown_scheme)

    # Record the words a pattern matched (e.g. "suggest"), not the regex, for the pipeline view.
    advice_shapes = [m.group(0) for p in intents.get("advice_patterns", []) if (m := re.search(p, lowered))]
    for intent in ("advice", "returns", "factual"):
        matched = _matches(lowered, intents.get(intent, []))
        if intent == "advice":
            matched += [shape for shape in advice_shapes if shape not in matched]
        if matched:
            return IntentResult(intent, scheme_id, matched)
    return IntentResult("other", scheme_id, [])


def match_scheme(text: str, sources: list[Source]) -> str | None:
    """Return the source id of the scheme named in text, preferring the longest match."""
    lowered = _normalize(text)
    best: tuple[int, str] | None = None
    for source in sources:
        if source.type != "scheme":
            continue
        for name in _scheme_names(source):
            if _contains(lowered, name) and (best is None or len(name) > best[0]):
                best = (len(name), source.id)
    return best[1] if best else None


def _scheme_names(source: Source) -> set[str]:
    names = {_normalize(source.title)}
    if source.scheme:
        names.add(_normalize(source.scheme))
    names.update(_normalize(alias) for alias in source.aliases)
    # "flexi cap" alone means HDFC Flexi Cap: every scheme in the corpus is HDFC's.
    names.update(name.removeprefix("hdfc ") for name in list(names) if name.startswith("hdfc "))
    return {name for name in names if name}


FUND_WORDS = ("fund", "elss", "tax saver", "large cap", "mid cap", "flexi cap", "small cap", "multi cap", "index", "etf")


def _brand_funds(text: str, brands: list[str], scheme_terms: list[str]) -> list[str]:
    """Brands like "groww" that are out of scope only when a fund name follows within a few words.

    "Groww ELSS Tax Saver Fund" is the brand's own scheme; "Groww app … HDFC Mid Cap" is not,
    so the words in between may not include "hdfc" or "app".
    """
    words = "|".join(re.escape(_normalize(term)) for term in (*FUND_WORDS, *scheme_terms))
    gap = r"(?:(?!hdfc\b|app\b)[a-z0-9&]+\s+){0,4}?"
    return [
        f"{brand} <fund>"
        for brand in brands
        if re.search(rf"(?<![a-z0-9]){re.escape(_normalize(brand))}\s+{gap}(?:{words})(?![a-z0-9])", text)
    ]


def _matches(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if _contains(text, _normalize(term))]


def _contains(text: str, term: str) -> bool:
    return bool(term) and re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def _normalize(text: str) -> str:
    text = text.lower().replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()
