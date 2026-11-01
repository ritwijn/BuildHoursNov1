"""Prompt and fixed-template copy. The LLM only ever sees retrieved chunks as context."""

from __future__ import annotations

from src.models import Hit

SYSTEM_PROMPT = """You are a facts-only assistant for HDFC Mutual Fund schemes listed on Groww.
Rules:
1. Answer ONLY from the CONTEXT blocks. Never use outside knowledge.
2. Answer the exact fact asked, in at most 3 short sentences.
3. If the CONTEXT does not contain the fact, reply exactly: "I couldn't find that in my sources."
4. Do not give advice, opinions or recommendations. Do not compute, predict or compare returns.
5. Copy numbers exactly as written in the CONTEXT.
6. "Maximum Total Expense Ratio" limits in SID/KIM documents are regulatory caps, not the
   scheme's actual expense ratio. Report an expense ratio only if the CONTEXT states the
   scheme's own figure (e.g. "Expense ratio: 1.02%").
7. For "how do I…" questions, answer with the steps given in the CONTEXT, even if the page's
   title is phrased differently.
8. End with one line: "Source: <url>" where <url> is copied exactly from the source_url of the
   context block you used. Use exactly one URL.
Output format (plain text, no tags or markdown):
The answer in 1-3 sentences.
Source: https://..."""

NOT_FOUND = "I couldn't find that in my sources."
REFUSAL = (
    "I can only share publicly listed scheme facts, not buy/sell or portfolio advice. "
    "To learn how to evaluate mutual funds yourself, see SEBI's investor education material below."
)
RETURNS = (
    "I don't calculate, predict or compare returns. "
    "The scheme's official past performance is published in the HDFC Mutual Fund factsheet below."
)
OUT_OF_SCOPE = (
    "This prototype only covers HDFC Mutual Fund and four of its schemes "
    "(Large Cap, Flexi Cap, ELSS Tax Saver, Mid Cap), so I can't answer that from my sources."
)
EMPTY = "Please type a question about an HDFC Mutual Fund scheme."
LLM_UNAVAILABLE = "The answer service is unavailable right now. Please try again."


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    blocks = [
        f"[{number}] source_url: {hit.source_url}\nsection: {hit.section}\ntext: {hit.text}"
        for number, hit in enumerate(hits, start=1)
    ]
    context = "CONTEXT:\n\n" + "\n\n".join(blocks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\nQUESTION: {question}"},
    ]
