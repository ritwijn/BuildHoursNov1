"""Second-line advice detection with the language model.

The keyword rules in intent.py catch common advice phrasings instantly, but people ask for
advice in endless ways ("May I know whether to invest?", "Would this suit someone like me?").
For questions the rules let through as factual, one short LLM call decides FACT vs ADVICE.

Fails open: if the call errors or the reply is unclear, the question is treated as factual and
continues to retrieval, where the answer prompt and post-check still block advice wording.
"""

from __future__ import annotations

import re

from src.generate.llm.base import LLMClient, LLMError, strip_reasoning

SYSTEM_PROMPT = """You classify questions sent to a facts-only mutual fund assistant.
Reply with exactly one word:
ADVICE - the user wants a recommendation, opinion or decision: whether/what/when/how much to invest, buy, sell, hold or switch; whether a fund is good, safe, suitable or worth it; which fund is better or best; predictions.
FACT - the user wants a factual attribute or a process: expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark, fund manager, AUM, NAV, who can invest, how to invest or download a statement, what a term means.
If unsure, reply FACT."""

LABEL = re.compile(r"\b(ADVICE|FACT)\b")


def is_advice_request(question: str, llm: LLMClient) -> bool | None:
    """True for advice, False for fact, None if the check couldn't decide."""
    try:
        raw = llm.generate(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
            temperature=0,
            max_tokens=300,
        )
    except LLMError:
        return None
    labels = LABEL.findall(strip_reasoning(raw).upper())
    if not labels:
        return None
    return labels[0] == "ADVICE"
