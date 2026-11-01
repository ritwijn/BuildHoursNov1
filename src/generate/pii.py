"""Detect personal details in a question before anything is logged, embedded or sent to an LLM.

Policy is reject, not strip: if any pattern matches, the whole question is refused.
detect_pii returns only the kinds found, never the matched values, so nothing
sensitive can leak into logs through this module.
"""

from __future__ import annotations

import re

PATTERNS: dict[str, re.Pattern[str]] = {
    "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE),
    "aadhaar": re.compile(r"(?<!\d)[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?!\d)"),
    "phone": re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "otp": re.compile(r"\b(?:otp|one[\s-]?time[\s-]?(?:password|code|pin))\b\D{0,20}\d{4,8}\b", re.IGNORECASE),
    "account": re.compile(
        r"\b(?:account|a/c|acct|folio)(?:\s*(?:no\.?|number|#))?\s*[:\-]?\s*[A-Z0-9/]*\d{4,}", re.IGNORECASE
    ),
    "long_number": re.compile(r"(?<![\d.,₹])\d{9,18}(?![\d.,%])"),
}

PII_MESSAGE = (
    "Please don't share personal details such as PAN, Aadhaar, phone, email, OTP or account numbers. "
    "I can only answer general questions about scheme facts."
)


def detect_pii(text: str) -> list[str]:
    """Return the kinds of personal data found (e.g. ["pan"]); empty if none."""
    return [kind for kind, pattern in PATTERNS.items() if pattern.search(text)]
