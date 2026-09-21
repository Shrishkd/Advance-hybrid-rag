"""Output guardrails — Phase 7. PII redaction.

WHY REGEX AND NOT PRESIDIO
--------------------------
The plan named Presidio. It needs spaCy language models and a heavier install, to
protect a corpus of machine learning textbooks that contains almost no personal data.
The realistic leak here is narrow - an email, phone number or card-like number
quoted from a preface or an example dataset - and regex covers that deterministically
and testably. Recorded as a substitution: Presidio's NER would also catch names and
addresses, which this does not.

Groundedness - the output guard that matters most for RAG - is already enforced by
src/generate/citations.py and surfaced in the API's `fabricated` field.
"""

from __future__ import annotations

import re

_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("PHONE", re.compile(r"(?<![\w.])\+?\d{1,3}[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}\b")),
]


def scrub(text: str) -> str:
    """Redact PII-shaped strings.

    >>> scrub("Contact author@example.com or +1 555 123 4567.")
    'Contact [EMAIL REDACTED] or [PHONE REDACTED].'
    >>> scrub("The model reached 0.976 accuracy on 5145 tokens [S1].")
    'The model reached 0.976 accuracy on 5145 tokens [S1].'
    """
    for label, rx in _PATTERNS:
        text = rx.sub(f"[{label} REDACTED]", text)
    return text
